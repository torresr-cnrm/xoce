"""
"""

import numpy as np
import xarray as xr

from nemopy.processing import Selector


def merge_coordinates(dataset, coords, diff='relative', tol=1e-6):
    """
    Compare dataset.coords and coords (for instance from a reference mesh).
    The algorithm compare each dimension values and check if the difference
    are inside the tolerance. If so, the dataset coordinate is modified in 
    order to correspond to the reference. 
    Otherwise (difference greater than tolerance) the code '-1' is returned.
    """
    code = 0

    for co in dataset.coords:
        if co in coords:
            cdiff = dataset.coords[co].data - coords[co].data

            rcond = np.abs(coords[co].data) < 1.
            if diff == 'relative':
                cdiff[~rcond] /= coords[co].data[~rcond] 

            conds = np.abs(cdiff) < tol
            datas = dataset.coords[co].data
            datas[conds] = coords[co].data[conds]
            dataset.coords[co] = datas

            if len(datas[conds]) < len(datas):
                code = -1

    return code


def array_diff(da, dim='time', method='forward'):
    """
    Make a diff operation under a DataArray object along one dimension.
    The method argument is set to 'forward' differentiation.
    """
    if method not in ['forward', 'backward']:
        raise Exception("Unknown method '{}'".format(method))
    
    arr = xr.full_like(da, np.nan)

    if method == 'forward':
        dab = da.isel({dim: slice(0,-1,1)})
        daa = da.isel({dim: slice(1,None,1)})
        daa.coords[dim] = dab.coords[dim]

        dif = daa - dab
        lst = dif.isel({dim: [-1]})
        lst.coords[dim] = [da.coords[dim][-1]]
        arr.data = xr.concat([dif, lst], dim=dim).data

    elif method == 'backward':
        dab = da.isel({dim: slice(0,-1,1)})
        daa = da.isel({dim: slice(1,None,1)})
        dab.coords[dim] = daa.coords[dim]

        dif = daa - dab
        fst = dif.isel({dim: [0]})
        fst.coords[dim] = [da.coords[dim][0]]
        arr.data = xr.concat([fst, dif], dim=dim).data

    return arr


def split_dataset(dataset, dim, bounds, drop=False):
    """
    Split dataset and return a list of datasets (one for 
    each band).

    This function is useful for discretizing a dataset
    along one dimension in order to apply some mathematical 
    operation on each band (ex: meridonal or zonal average).
    """

    if dim not in dataset.coords:
        raise Exception("'{}' is not in dataset coordinates: ".format(dim) +
                        "{}".format(list(dataset.coords)))
    
    if not bounds:
        return [xr.Dataset(dataset)]
    
    # init resulting list
    res = list()
    
    # use box clipper processing with only one dim
    clipper = Selector('box-clip', dataset=dataset)
    clipper.set('drop', drop)

    for bnds in bounds:
        clipper.set('box', {dim: (bnds[0], bnds[1])})
        clipped = clipper.execute()
        res.append(clipped)

    return res

