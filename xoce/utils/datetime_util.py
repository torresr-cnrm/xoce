"""
"""

import cftime
import numpy as np
import xarray as xr


# calendar conventions
_dayspermonth      = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_dayspermonth_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# vectorized conversion
_np2datetime = np.vectorize(cftime.DatetimeNoLeap)


def datetime_mean(arr:xr.DataArray, dim):
    """
    Compute the mean of all datetime datas
    """
    data = arr.dropna(dim=dim).data
    datetype = type(data[0])

    ymean = np.mean([d.year for d in data])
    mmean = np.mean([d.month for d in data])
    dmean = np.mean([d.day for d in data])

    return datetype(ymean, mmean, dmean)


def datetime_to_cftime(arr:np.ndarray):
    """
    Loop over an array to convert numpy datetime values in
    cftime datetime type.
    """

    cfarr = np.zeros_like(arr, dtype='O')

    if len(arr.shape) > 1: 
        for i in range(arr.shape[0]):
            cfarr[i] = datetime_to_cftime(arr[i])
    else:
        for i, dt in enumerate(arr):
            dtdate, dttime = str(dt).split('T')
            list_cft = dtdate.split('-') + dttime.split(':')
            cfarr[i] = cftime.DatetimeNoLeap(*[int(float(e)) for e in list_cft])

        return cfarr

    return cfarr


def decode_months_since(arr:np.ndarray, calendar='noleap'):
    """
    Loop over a time-array to decode the datas and return a numpy equivalent 
    array with datetime values.
    """

    units = arr.attrs['units']
    shape = arr.shape
    dats  = arr.data.flatten()

    if not ('since' in units and units.split()[0] == 'months'):
        raise Exception("Unable to decode units: {}.".format(units))
    
    lu1 = units.split()[-2]
    lu2 = units.split()[-1]
    
    y0  = int(lu1.split('-')[0])
    m0  = int(lu1.split('-')[1])
    d0  = int(lu1.split('-')[2])
    h0  = int(lu2.split(':')[0])
    mn0 = int(lu2.split(':')[1])
    s0  = int(lu2.split(':')[2])
    
    years  = y0 + dats // 12
    months = m0 + dats % 12 // 1
    
    dpm    = np.take(_dayspermonth, months.astype('int')-1)
    days   = d0 + (dats % 12 % 1) * dpm // 1
            
    nhours = 24  *(dats % 12 % 1) * dpm % 1
    hours  = h0  + nhours
    minuts = mn0 + nhours % 24 // 1
    secnds = s0  + nhours % 24 %  1 // 1

    if calendar in ['noleap', '365_day']: 
        dtimes = _np2datetime(years, months, days, hours, minuts, secnds)

    return dtimes.reshape(shape)

