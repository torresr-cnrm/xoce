"""
"""

import numpy as np
import xarray as xr

from xoce.processing.selectors import BoxClipper


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


def concatenate_arrays(larrays, dim, chunks=None):
    """
    Concatenate and rechunk the resulting array if necessary.
    This function simply call the xarray.concat method and 
    rechunk the final array to avoid bad dask usage.
    """
    arr = xr.concat(larrays, dim=dim)
    
    if chunks is None:
        arr = arr.chunk({})
    else:
        arr = arr.chunk(chunks)

    return arr


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
        arr.data = concatenate_arrays([dif, lst], dim=dim, chunks=da.chunks).data

    elif method == 'backward':
        dab = da.isel({dim: slice(0,-1,1)})
        daa = da.isel({dim: slice(1,None,1)})
        dab.coords[dim] = daa.coords[dim]

        dif = daa - dab
        fst = dif.isel({dim: [0]})
        fst.coords[dim] = [da.coords[dim][0]]
        arr.data = concatenate_arrays([fst, dif], dim=dim, chunks=da.chunks).data

    return arr


def array_bnds(da, dim=None):
    """
    Compute DataArray boundaries.
    """
    if isinstance(dim, str):
        dim = (dim,)

    if not set(dim) <= set(da.dims):
        raise Exception("'{}' is not included in DataArray dimension".format(dim))
    elif dim is None:
        dim = da.dims
    
    arr = xr.full_like(da, np.nan)
    
    arr.name = da.name + '_bnds'
    arr = arr.expand_dims({'nbounds': 2**len(dim)}, axis=-1)

    dshape = list(arr.shape)
    dshape[-1] = 0

    dbnds = xr.DataArray(np.ones(tuple(dshape)), dims=arr.dims)

    for d in dim:
        # uptream and downstream boundaries
        dab = da.isel({d: slice(0,-1,1)})
        daa = da.isel({d: slice(1,None,1)})
        for c in daa.coords:
            if d in daa.coords[c].dims:
                dab.coords[c] = daa.coords[c]

        fst = da.isel({d: [0]})
        lst = da.isel({d: [-1]})

        dbnd1 = concatenate_arrays([fst, 0.5*(dab+daa)], dim=d, chunks=da.chunks)
        dbnds = concatenate_arrays([dbnds, dbnd1], dim='nbounds', chunks=da.chunks)

        for c in da.coords:
            if d in da.coords[c].dims:
                dab.coords[c] = da.isel({d: slice(0,-1,1)}).coords[c]
                daa.coords[c] = dab.coords[c]

        dbnd2 = concatenate_arrays([0.5*(dab+daa), lst], dim=d, chunks=da.chunks)
        dbnds = concatenate_arrays([dbnds, dbnd2], dim='nbounds', chunks=da.chunks)

    arr.data = dbnds.data

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
    clipper = BoxClipper(dataset=dataset)
    clipper.set('drop', drop)

    for bnds in bounds:
        clipper.set('box', {dim: (bnds[0], bnds[1])})
        clipped = clipper.execute()
        res.append(clipped)

    return res


def get_dim_axis(dataset, dim_names, skip_notfound=False):
    """
    Return the axis order af a `dataset` (could be both xr.Dataset or xr.DataArray)
    corresponding to the desired dimensions in dim_names. 
    The option skip_notfound can be turn on to avoid raising error if dimension is 
    only in dim_names list.
    """
    axis   = list()
    skiped = list()

    for d in dim_names:
        if d not in dataset.dims:
            if skip_notfound:
                skiped.append(list(dim_names).index(d))
            else:
                raise Exception("'{}' is not in dataset dimensions: ".format(d) +
                                "{}".format(list(dataset.dims)))
        else:
            axis.append(list(dataset.dims).index(d))

    return axis, skiped


def broadcast_like(model, da:xr.DataArray):
    """
    Returns a new xr.DataArray with the exact same shape as `model` 
    and fill with the values of `da`.
    """
    res = xr.full_like(model, 0)
    res.name = da.name
    res.attrs = da.attrs
    
    rshape   = [1] * len(res.shape)
    zaxis, _ = get_dim_axis(res, ['depth'])[0]
    rshape[zaxis] = res.shape[zaxis]
    
    for d in res.dims:
        if d not in da.dims:
            da = da.expand_dims({d: 1})
    da = da.transpose(*list(res.dims))

    res[:] = da[:]
    
    return res


def interp_coord(da, coords, dim, method='linear'):
    """
    Interpolate an array with a new coordinate array.
    Only 1D interpolation is allowed which means pnly one coordinate in 
    the dictionary coords.
    """
    def _delta(a, i):
        slcup = [slice(0, None, 1)] * len(a.shape)
        slcdw = [slice(0, None, 1)] * len(a.shape)
        slcup[i] = slice(1, None, 1)
        slcdw[i] = slice(0, -1, 1)

        slcup = tuple(slcup)
        slcdw = tuple(slcdw)
        
        return a[slcup] - a[slcdw], (slcup, slcdw)
    
    if method not in ['linear']:
        raise Exception("Unknown method '{}'".format(method))

    arr = xr.full_like(da, np.nan)

    if method == 'linear':
        cname  = list(coords.keys())[0]

        # compute delta of coord
        dco = np.array(coords[cname]) - da[cname].data
        
        shp = [1] * len(da.shape)
        for d in da[cname].dims:
            ind = da.dims.index(d)
            shp[ind] = da.shape[ind]

        dco = dco.reshape(shp)

        # compute delta of coordinate upward and downward
        j = da[cname].dims.index(dim)
        cup = np.ones_like(coords[cname])
        cdw = np.ones_like(coords[cname])

        delta_co, slicers = _delta(da[cname].data, j)

        cup[slicers[1]] = delta_co
        cdw[slicers[0]] = delta_co

        cup = cup.reshape(shp)
        cdw = cdw.reshape(shp)

        # compute delta of data upward and downward
        i = da.dims.index(dim)
        dup = xr.full_like(da, 0.)
        ddw = xr.full_like(da, 0.)

        delta_da, slicers = _delta(da.data, i)

        dup[slicers[1]] = delta_da
        ddw[slicers[0]] = delta_da

        # compute final data 
        # array = data + coef_up * (dy/dx)_up - coef_down * (dy/dx)_down
        arr.data = da.data  + dco * ( (np.sign(dco) + 1) / 2.) * dup/cup
        arr.data = arr.data - dco * ( (np.sign(dco) - 1) / 2.) * ddw/cdw

    return arr


def assign_variable(ds, da, name=None, interpolate=True):
    """
    Append a new DataArray `da` to a Dataset `ds`. This function check the 
    dimensions consistency between the dataset and the array.
    If `interpolate` is True, a linear interpolation is used to fit the 
    new array with the dataset dimensions. Otherwise, the array dimensions 
    are overwritten.
    """
    array = da
    for d in da.dims :
        if d in ds.dims and not np.alltrue(da[d].data == ds[d].data):
            if interpolate:
                array = array.interp({d:ds[d].data})
            else:
                array[d].data = ds[d].data
    
    if name is None:
        name = array.name

    ds[name] = array


def check_dims(da, refdims):
    """
    Check if array dimensions and shape fit with a dictionnary
    of reference dimensions.
    """
    status = True
    for d, i in enumerate(da.dims) :
        if d not in refdims :
            status = False
        else:
            if refdims[d] != da.shape[i]:
                status = False

    return status

