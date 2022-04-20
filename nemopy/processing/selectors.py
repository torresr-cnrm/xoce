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

            newdim = xr.IndexVariable(dim, sliced.data)
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

