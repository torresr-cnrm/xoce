"""
"""

import xarray as xr

from ..api.generic import NemopyObject


class BoxClipper(NemopyObject):
    """
    Box selection regarding some box on dataset coordinates. 
    """
    _Parameters = {
        "box": {'type': dict,
                'default': None},
        "drop": {'type': bool,
                 'default': False},
        "inverse": {'type': bool,
                    'default': False}
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset

        for dim in self.box:
            if dim not in ds.coords:
                raise Exception("BoxClipper error: '{}' ".format(dim) + 
                            "is not in dataset coordinates {}".format(list(ds.coords)))

        # coordinates selection
        coords = {c: ds.coords[c] for c in ds.coords if c not in self.box}
        for dim in self.box:
            conds  = (ds[dim] >= self.box[dim][0])
            conds  = conds & (ds[dim] <= self.box[dim][1])
            if self.inverse:
                sliced = ds[dim].where(~conds, drop=self.drop)
            else:
                sliced = ds[dim].where(conds, drop=self.drop)

            if len(ds[dim]) > 1:
                newdim = xr.Variable(ds[dim].dims, sliced.data)
            else:
                newdim = xr.IndexVariable(ds[dim].dims, sliced.data)   
            coords.update({dim: newdim})

        selected = xr.Dataset(coords=coords)

        conds = ds == ds
        for dim in self.box:
            conds  = conds & (ds[dim] >= self.box[dim][0])
            conds  = conds & (ds[dim] <= self.box[dim][1])
        
        if self.inverse:
            sliced = ds.where(~conds, drop=self.drop)
        else:
            sliced = ds.where(conds, drop=self.drop)
        
        # add selected box variable
        selected = sliced
        
        return selected


class Cutter(NemopyObject):
    """
    Cut one dimension to reduce dataset dims. If the cut value is
    not directly included in dimensions, a linear interpolation is 
    made. 
    Usage example: cut at a specified depth to extract a 2D map. 

    TODO: This processing not finished yet.. 
    """
    _Parameters = {
        "dim": {'type': str,
                'default': 'Depth'},
        "value": {'type': (float, int, str),
                 'default': 0.0},
        "method": {'type': str,
                   'default': 'linear'},
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset
        dim = self.dim
        val = self.value
        
        ldims = list(ds.dims) + [c for c in ds.coords if c not in ds.dims]
        if dim not in ldims:
            raise Exception("Cutter error: '{}' ".format(dim) + 
                        "is not in dataset dimensions or coordinates {}".format(ldims))

        dimtype = type(ds[self.dim].data[0])
        
        if dim in ['time', 't'] and ds[dim].dtype == 'O':
            # Converting to unregular datetime
            ymd = self.value.split('-')
            val = dimtype(*ymd)

        # coordinates selection
        coords = {c: ds.coords[c] for c in ds.coords if c not in self.dim}
        if len(ds[dim]) > 1:
            if val in ds[dim]:
                newval = ds[dim]
            newdim = xr.Variable(ds[dim].dims, newval)
        else:
            newdim = xr.IndexVariable(ds[dim].dims, [val])
        coords.update({dim: newdim})

        selected = xr.Dataset(coords=coords)
        
        return selected