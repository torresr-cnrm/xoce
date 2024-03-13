"""
"""

import numpy  as np
import xarray as xr
import xoce.utils.dataset_util as xdsutil

from ..api.generic import XoceObject, set_attrs


class BoxClipper(XoceObject):
    """
    Box selection regarding some box on dataset coordinates.
    """
    _Parameters = {
        "box": {'type': dict,
                'default': None},
        "drop": {'type': bool,
                 'default': False},
        "inverse": {'type': bool,
                    'default': False},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    @set_attrs
    def execute(self):
        ds = self.dataset

        for dim in self.box:
            if dim not in ds.coords:
                raise Exception("BoxClipper error: '{}' ".format(dim) + 
                            "is not in dataset coordinates {}".format(list(ds.coords)))

        # init dictionary of clip boundaries
        dict_bnds = dict()

        # coordinates selection
        coords = {c: ds.coords[c] for c in ds.coords if c not in self.box}
        for dim in self.box:
            bounds  = self.box[dim]
            dimtype = type(ds[dim].data[0])

            if bounds is None:
                bounds = (ds[dim][0], ds[dim][-1])
            
            elif dim in ['time', 't']: 
                # Converting to unregular datetime
                if ds[dim].dtype == 'O':
                    dateinf = [int(ymd) for ymd in bounds[0].split('-')]
                    datesup = [int(ymd) for ymd in bounds[1].split('-')]
                elif ds[dim].dtype == '<M8[ns]':
                    dateinf = [bounds[0]]
                    datesup = [bounds[1]]
                bounds = (dimtype(*dateinf), dimtype(*datesup))

            dict_bnds[dim] = bounds

            conds  = (ds[dim] >= bounds[0])
            conds  = conds & (ds[dim] <= bounds[1])
            if self.inverse:
                sliced = ds[dim].where(~conds, drop=self.drop)
            else:
                sliced = ds[dim].where(conds, drop=self.drop)

            if len(ds[dim]) > 1:
                newvar = xr.Variable(ds[dim].dims, sliced.data)
            else:
                newvar = xr.IndexVariable(ds[dim].dims, sliced.data)   
            newvar.attrs = sliced.attrs

            newdim = xr.DataArray(newvar, coords=sliced.coords)
            newdim.name = dim

            coords.update({dim: newdim})

        selected = xr.Dataset()
        for co in coords:
            selected.coords[co] = coords[co]

        conds = True
        for dim in self.box:
            tmp_conds = (ds[dim] >= dict_bnds[dim][0])
            
            if dict_bnds[dim][0] > dict_bnds[dim][1]:
                tmp_conds = tmp_conds | (ds[dim] <= dict_bnds[dim][1])

            conds  = conds & tmp_conds
            conds  = conds & (ds[dim] <= dict_bnds[dim][1])
        
        if self.inverse:
            sliced = ds.where(~conds, drop=self.drop)
        else:
            sliced = ds.where(conds, drop=self.drop)
        
        # add selected box variable
        selected = sliced
        
        return selected



class FieldSelector(XoceObject):
    """
    Select variables to return a reduced dataset. 
    """
    _Parameters = {
        "variables": {'type': list,
                      'default': list()},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    @set_attrs
    def execute(self):
        ds = self.dataset

        for var in self.variables:
            if var not in ds.variables:
                raise Exception("Selector error: '{}' ".format(var) + 
                        "is not in dataset coordinates ()".format(list(ds.variables)))

        newcoords = list()
        selected = xr.Dataset(coords=ds.coords)

        for var in self.variables:
            selected[var] = ds[var]

            for co in selected[var].coords:
                if co not in newcoords:
                    newcoords.append(co)

        # filter coordinates in selected variables
        for co in selected.coords:
            if co not in newcoords:
                del selected.coords[co]
        
        return selected


