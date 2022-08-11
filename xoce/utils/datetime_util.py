"""
"""

import cftime
import numpy as np
import xarray as xr


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

