"""
"""

import cftime
import numpy as np
import xarray as xr

import xoce.utils.dataset_util as xdsutil

from ..api.generic import XoceObject
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


    def execute(self):
        ds = self.dataset
        dimtype = type(ds[self.dim].data[0])

        variables = self.variables
        if variables is None:
            variables = ds.variables

        period = self.period
        if period is None:
            period = (ds[self.dim][0], ds[self.dim][-1])
        
        elif self.dim in ['time', 't'] and ds[self.dim].dtype == 'O':
            # Converting to unregular datetime
            dateinf = [int(ymd) for ymd in period[0].split('-')]
            datesup = [int(ymd) for ymd in period[1].split('-')]
            period = (dimtype(*dateinf), dimtype(*datesup))

        coords = ds.coords

        if not self.inverse:
            coords = {c: ds.coords[c] for c in ds.coords if c != self.dim}
            conds  = (ds[self.dim] >= period[0])
            conds  = conds & (ds[self.dim] <= period[1])
            sliced = ds[self.dim].where(conds)

            # special case for datetime data
            if cftime.datetime in dimtype.__bases__:
                newdim = xr.IndexVariable(self.dim, [datetime_mean(sliced, self.dim)])
            else:
                newdim = xr.IndexVariable(self.dim, [sliced.mean(dim=self.dim).data])
            coords.update({self.dim: newdim})

        filtered = xr.Dataset(coords=coords)

        for v in variables:
            if not (v in ds.coords) and self.dim in ds[v].dims:
                conds  = (ds[v][self.dim] >= period[0])
                conds  = conds & (ds[v][self.dim] <= period[1])
                sliced = ds[v].where(conds)
                vmean  = sliced.mean(dim=self.dim)

                if self.inverse:
                    narray = ds[v] - vmean
                else:
                    narray = vmean.expand_dims({self.dim:filtered[self.dim]})
                
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
        "fill_nan": {'type': str, 
                     'default': 'nan'}
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


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
            
            # 1. Compute shapiro coefficients

            # -- prepare auto-slicers
            coefs = xr.full_like(da, 1) / (2**(2*len(axis)))

            slicers = [slice(None, None, None)] * len(da.dims)
            frstslc = slice(0, 1, None)
            lastslc = slice(-1, None, None)

            # -- change domain border coefs.
            for a in axis:
                slicers[a] = frstslc
                coefs[tuple(slicers)] = 1/12.
                slicers[a] = lastslc
                coefs[tuple(slicers)] = 1/12.
                slicers[a] = slice(None, None, None)

            # -- change domain corner coefs.
            slicers[axis[0]] = frstslc
            slicers[axis[1]] = frstslc
            coefs[tuple(slicers)] = 1/9.
            
            slicers[axis[0]] = frstslc
            slicers[axis[1]] = lastslc
            coefs[tuple(slicers)] = 1/9.

            slicers[axis[0]] = lastslc
            slicers[axis[1]] = frstslc
            coefs[tuple(slicers)] = 1/9.

            slicers[axis[0]] = lastslc
            slicers[axis[1]] = lastslc
            coefs[tuple(slicers)] = 1/9.

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
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)]

                aslicers[a] = afterslc
                fslicers[a] = beforeslc
                farr[tuple(fslicers)] += 2*da[tuple(aslicers)]

                aslicers[a] = slice(None, None, None)
                fslicers[a] = slice(None, None, None)
                         
            aslicers[axis[0]] = beforeslc
            aslicers[axis[1]] = beforeslc
            fslicers[axis[0]] = afterslc
            fslicers[axis[1]] = afterslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)]
            
            aslicers[axis[0]] = beforeslc
            aslicers[axis[1]] = afterslc
            fslicers[axis[0]] = afterslc
            fslicers[axis[1]] = beforeslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)]

            aslicers[axis[0]] = afterslc
            aslicers[axis[1]] = beforeslc
            fslicers[axis[0]] = beforeslc
            fslicers[axis[1]] = afterslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)]

            aslicers[axis[0]] = afterslc
            aslicers[axis[1]] = afterslc
            fslicers[axis[0]] = beforeslc
            fslicers[axis[1]] = beforeslc
            farr[tuple(fslicers)] += 1*da[tuple(aslicers)] 

            farr = xr.where(mask, np.nan, farr * coefs)

            # 3. Finally manage new nan (here cause of the algorithm)
            if self.fill_nan == 'old':
                farr = xr.where(np.isnan(farr) & ~mask, da, farr)

            filtered[v] = (farr.dims, farr.data)

        return filtered

