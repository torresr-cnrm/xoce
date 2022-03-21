"""
"""

import xarray as xr


def time_filtering(ds, dim='time', period=None, inverse=False):
    """
    """
    
    filtered = xr.Dataset(coords=ds.coords)

    for v in ds.variables:
        if not (v in ds.dims):
            vmean = ds[v].mean(dim=dim)
            if inverse:
                filtered[v] = ds[v] - vmean
            else:
                filtered[v] = vmean
    
    return filtered


