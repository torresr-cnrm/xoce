"""
"""

import cftime
import dask
import numpy as np
import xarray as xr

import xoce.utils.dataset_util as xdsutil
import xoce.api.experiment as xp

from ..api.generic import XoceObject, set_attrs
from ..utils.datetime_util import datetime_mean


class AverageFilter(XoceObject):
    """
    Filtering Dataset by averaging all variables along 
    a specified dimension.
    """
    _Parameters = {
        "dim": {'type': str,
                'default': 'time'},
        "period": {'type': list,
                   'default': None},
        "inverse": {'type': bool,
                    'default': False}, 
        "variables": {'type': list,
                      'default': None},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    @set_attrs
    def execute(self):
        ds = self.dataset

        variables = self.variables
        if variables is None:
            variables = ds.variables

        period = self.period
        if period is None:
            period = (ds[self.dim][0], ds[self.dim][-1])

        coords = ds.coords

        if not self.inverse:
            coords = {c: ds.coords[c] for c in ds.coords if c != self.dim}
            cmean  = ds[self.dim].sel({self.dim: slice(*period)}).mean(dim=self.dim)
            newdim = xr.IndexVariable(self.dim, [cmean])
            coords.update({self.dim: newdim})

        filtered = xr.Dataset(coords=coords)

        for v in variables:
            iscalc = isinstance(ds, xp.Experiment) and not ds._calc.is_calculable(v)
            if v not in ds.variables and not iscalc :
                continue

            if not (v in ds.coords) and self.dim in ds[v].dims:
                vmean  = ds[v].sel({self.dim: slice(*period)}).mean(dim=self.dim)

                if self.inverse:
                    narray = ds[v] - vmean.load()
                else:
                    narray = vmean.expand_dims({self.dim:filtered[self.dim]})
                
            elif not self.dim in ds[v].dims:
                narray = ds[v]

            # add filtered variable
            narray.name = v
            xdsutil.assign_variable(filtered, narray)

        return filtered


class ShapiroFilter(XoceObject):
    """
    Smooth some Dataset variables using a 2D shapiro filter.
    """
    _Parameters = {
        "dims": {'type': list,
                 'default': ('x', 'y')},
        "variables": {'type': list,
                      'default': None},
        "periodicity": {'type': str,
                        'default': None},
        "fill_nan": {'type': str, 
                     'default': 'nan'}
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    @set_attrs
    def execute(self):
        ds = self.dataset

        if len(self.dims) != 2:
            raise Exception("ShapiroFilter error: only 2D filtering is allowed. " +
                            "Need to specify 2 dimensions for the parameter 'dims'.")

        variables = self.variables
        if variables is None:
            variables = ds.variables

        filtered = xr.Dataset(coords=ds.coords)

        for v in variables:
            da = ds[v]

            # 0. Get axis indices and check if in DataArray dimensions
            axis  = list()
            for d in self.dims:
                if d not in da.dims:
                    raise Exception("'{}' is not in dataset dimensions: ".format(d) +
                                    "{}".format(list(da.dims)))
                else:
                    axis.append(list(da.dims).index(d))

            iper = None
            if self.periodicity is not None:
                if self.periodicity in self.dims:
                    iper = list(da.dims).index(self.periodicity)
                else:
                    print("ShapiroFilter warning: '{}' not".format(self.periodicity) + 
                          " in 2d filter dimensions ({}).".format(self.dims) +
                          " {}-periodicity ignored".format(self.periodicity))

            # 1. Compute shapiro coefficients
            # -- prepare auto-slicers
            coefs = xr.full_like(da, 1) / (2**(2*len(axis)))

            slicers = [slice(None, None, None)] * len(da.dims)
            frstslc = slice(0, 1, None)
            lastslc = slice(-1, None, None)

            # -- change domain border coefs.
            for a in axis:
                if iper is None or a == iper:
                    slicers[a] = frstslc
                    coefs[tuple(slicers)] = 1/12.
                    slicers[a] = lastslc
                    coefs[tuple(slicers)] = 1/12.
                    slicers[a] = slice(None, None, None)

            # -- change domain corner coefs.
            slicers[axis[0]] = frstslc
            slicers[axis[1]] = frstslc
            coefs[tuple(slicers)] = 1/9.
            if iper is not None:
                coefs[tuple(slicers)] = 1/12.
            
            slicers[axis[0]] = frstslc
            slicers[axis[1]] = lastslc
            coefs[tuple(slicers)] = 1/9.
            if iper is not None:
                coefs[tuple(slicers)] = 1/12.

            slicers[axis[0]] = lastslc
            slicers[axis[1]] = frstslc
            coefs[tuple(slicers)] = 1/9.
            if iper is not None:
                coefs[tuple(slicers)] = 1/12.

            slicers[axis[0]] = lastslc
            slicers[axis[1]] = lastslc
            coefs[tuple(slicers)] = 1/9.
            if iper is not None:
                coefs[tuple(slicers)] = 1/12.

            # 2. Compute filtered array
            mask = np.isnan(da)
            da   = xr.where(mask, np.nan, da)

            # -- prepare auto-slicers
            aslicers = [slice(None, None, None)] * len(da.dims)
            fslicers = [slice(None, None, None)] * len(da.dims)
            beforeslc = slice(0, -1, None)
            afterslc  = slice(1, None, None)
            
            farr  = xr.full_like(da, 0.)
            farr += 4*da

            for a in axis:
                aslicers[a] = beforeslc
                fslicers[a] = afterslc
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)].data

                aslicers[a] = afterslc
                fslicers[a] = beforeslc
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)].data

                aslicers[a] = slice(None, None, None)
                fslicers[a] = slice(None, None, None)
                         
            aslicers[axis[0]] = beforeslc
            aslicers[axis[1]] = beforeslc
            fslicers[axis[0]] = afterslc
            fslicers[axis[1]] = afterslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data
            
            aslicers[axis[0]] = beforeslc
            aslicers[axis[1]] = afterslc
            fslicers[axis[0]] = afterslc
            fslicers[axis[1]] = beforeslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

            aslicers[axis[0]] = afterslc
            aslicers[axis[1]] = beforeslc
            fslicers[axis[0]] = beforeslc
            fslicers[axis[1]] = afterslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

            aslicers[axis[0]] = afterslc
            aslicers[axis[1]] = afterslc
            fslicers[axis[0]] = beforeslc
            fslicers[axis[1]] = beforeslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

            # -- periodicity
            if iper is not None:          
                nper = axis[0]
                if iper == nper:
                    nper = axis[1]

                aslicers[iper] = slice(None, None, None)
                aslicers[nper] = slice(-1, None, 1)
                fslicers[iper] = slice(None, None, None)
                fslicers[nper] = slice(0, 1, 1)
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)].data

                aslicers[iper] = slice(None, None, None)
                aslicers[nper] = slice(0, 1, 1)
                fslicers[iper] = slice(None, None, None)
                fslicers[nper] = slice(-1, None, 1)
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)].data

                aslicers[iper] = beforeslc
                aslicers[nper] = slice(-1, None, 1)
                fslicers[iper] = afterslc
                fslicers[nper] = slice(0, 1, 1)
                farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

                aslicers[iper] = afterslc
                aslicers[nper] = slice(-1, None, 1)
                fslicers[iper] = beforeslc
                fslicers[nper] = slice(0, 1, 1)
                farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

                aslicers[iper] = beforeslc
                aslicers[nper] = slice(0, 1, 1)
                fslicers[iper] = afterslc
                fslicers[nper] = slice(-1, None, 1)
                farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

                aslicers[iper] = afterslc
                aslicers[nper] = slice(0, 1, 1)
                fslicers[iper] = beforeslc
                fslicers[nper] = slice(-1, None, 1)
                farr[tuple(fslicers)] += 1*da[tuple(aslicers)].data

            farr = xr.where(mask, np.nan, farr * coefs)

            # 3. Finally manage new nan (here cause of the algorithm)
            if self.fill_nan == 'old':
                farr = xr.where(np.isnan(farr) & ~mask, da, farr)

            filtered[v] = (farr.dims, farr.data)

        return filtered


class CoarseningFilter(XoceObject):
    """
    Smooth some Dataset variables using coarsening algorithm.
    """
    _Parameters = {
        "dims": {'type': list,
                 'default': ('x', 'y')},
        "factor": {'type': int,
                   'default': 2},
        "variables": {'type': list,
                      'default': None},
        "bnds_dim": {'type': str,
                     'default': 'nbounds'},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    @set_attrs
    def execute(self):
        ds = self.dataset

        dims = self.dims
        if isinstance(self.dims, str):
            dims = [self.dims]

        bndim = self.bnds_dim

        variables = self.variables
        if variables is None:
            variables = [v for v in ds.variables if v not in ds.coords]

        # pre processing: create slicers for latter averaging + compute new coords
        slicers = [slice(i, None, self.factor) for i in range(self.factor)]

        coords = dict()
        for c in ds.coords:
            ncoordinate = xr.DataArray(ds.coords[c])

            for d in dims:
                co = ncoordinate
                coord = 0.
                if d in co.dims:
                    ncells  = co.shape[co.dims.index(d)]
                    nfulls = ncells % self.factor

                    add_mean = co.isel({d: slicers[0]}).isel({d: [-1]})

                    for i, slc in enumerate(slicers):
                        sel = co.isel({d: slc})
                        
                        if i < nfulls:
                            if i != 0:
                                add_mean += co.isel({d: slicers[i]}).isel({d: [-1]})

                        elif nfulls != 0:
                            mean_data = (1/nfulls) * add_mean.data
                            dim_axis = sel.dims.index(d)
                            dat = np.concatenate( (sel.data, mean_data), axis=dim_axis)

                            sel = xr.DataArray(data=dat, name=sel.name, dims=sel.dims)

                        sel[d] = np.arange(len(sel[d]))
                        if c in dims:
                            sel.data  = sel[d].data

                        coord += sel

                if isinstance(coord, (int, float)) and coord == 0:
                    ncoordinate = co
                else:
                    ncoordinate = coord / self.factor

            coords[c] = ncoordinate

        # final check to ensure coherent coordinates
        for c in coords:
            co = coords[c]
            for cc in co.coords:
                if cc in coords:
                    co.coords[cc] = coords[cc]

        # init filtered dataset
        filtered = xr.Dataset(coords=coords)

        for v in variables:
            narray = xr.DataArray(ds[v])

            for d in dims:
                da = narray
                arr = 0.
                if d in da.dims:
                    ncells  = da.shape[da.dims.index(d)]
                    nfulls = ncells %  self.factor

                    add_mean = da.isel({d: slicers[0]}).isel({d: [-1]})

                    for i, slc in enumerate(slicers):
                        sel = da.isel({d: slc})
                        
                        if i < nfulls:
                            if i != 0:
                                if bndim in da.dims:
                                    bnd_axis = da.dims.index(bndim)
                                    slc_list = [slice(None, None, None)] * len(da.dims)
                                    sel_last = sel.isel({d: [-1]})

                                    # minimum only
                                    slc_list[bnd_axis] = slice(0, 1, None)
                                    minv = np.minimum(add_mean[tuple(slc_list)], 
                                                      sel_last[tuple(slc_list)])
                                    add_mean[tuple(slc_list)] = minv * nfulls

                                    # maximum only
                                    slc_list[bnd_axis] = slice(-1, None, None)
                                    maxv = np.maximum(add_mean[tuple(slc_list)], 
                                                      sel_last[tuple(slc_list)])
                                    add_mean[tuple(slc_list)] = maxv * nfulls
                                else:
                                    add_mean += da.isel({d: slicers[i]}).isel({d: [-1]})

                        elif nfulls:
                            mean_data = (1/nfulls) * add_mean.data
                            dim_axis = sel.dims.index(d)
                            dat = np.concatenate( (sel.data, mean_data), axis=dim_axis)

                            if sel.chunks is None:
                                chunks = {}
                            else:
                                chunks = sel.chunks

                            sel = xr.DataArray(data=dat, name=sel.name, dims=sel.dims)

                            if chunks:
                                chunks = list(sel.chunks)
                                dchnks = list(chunks[dim_axis])
                                dchnks[-2] += dchnks[-1]

                                del dchnks[-1]
                                
                                chunks[dim_axis] = tuple(dchnks)
                                chunks = tuple(chunks)

                            if isinstance(sel.data, dask.array.core.Array):
                                sel = sel.chunk(chunks)
                        
                        sel[d] = np.arange(len(sel[d]))
                        if v in dims:
                            sel.data  = sel[d].data

                        if not (bndim in sel.dims):
                            arr += sel
                        elif isinstance(arr, (int, float)) and arr == 0:
                            arr = sel 
                        else:
                            # manage case of bnds variables (need to take min/max bnds)
                            bnd_axis = arr.dims.index(bndim)
                            slc_list = [slice(None, None, None)] * len(arr.dims)
                            
                            # minimum only
                            slc_list[bnd_axis]   = slice(0, 1, None)
                            arr[tuple(slc_list)] = np.minimum(arr[tuple(slc_list)], 
                                                              sel[tuple(slc_list)])

                            # maximum only
                            slc_list[bnd_axis]   = slice(-1, None, None)
                            arr[tuple(slc_list)] = np.maximum(arr[tuple(slc_list)], 
                                                              sel[tuple(slc_list)])

                if isinstance(arr, (int, float)) and arr == 0:
                    narray = xr.DataArray(da)
                else:
                    narray = arr
                    if not (bndim in arr.dims):
                        narray /= self.factor

            # add filtered variable
            narray.name = v
            xdsutil.assign_variable(filtered, narray)

        return filtered

