"""
"""

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
