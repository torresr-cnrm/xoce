"""
"""

import xarray as xr

from ..api.generic import NemopyObject


class AverageFilter(NemopyObject):
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
                    'default': False}
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset

        if self.period is None:
            period = (ds[self.dim][0], ds[self.dim][-1])

        coords = ds.coords

        if not self.inverse:
            coords = {c: ds.coords[c] for c in ds.coords if c != self.dim}
            conds  = (ds[self.dim] >= period[0])
            conds  = conds & (ds[self.dim] <= period[1])
            sliced = ds[self.dim].where(conds)
            newdim = xr.IndexVariable(self.dim, [sliced.mean(dim=self.dim).data])
            coords.update({self.dim: newdim})

        filtered = xr.Dataset(coords=coords)

        for v in ds.variables:
            if not (v in ds.coords):
                conds  = (ds[v][self.dim] >= period[0])
                conds  = conds & (ds[v][self.dim] <= period[1])
                sliced = ds[v].where(conds)
                vmean  = sliced.mean(dim=self.dim)

                # add filtered variable
                if self.inverse:
                    filtered[v] = ds[v] - vmean
                else:
                    filtered[v] = vmean.expand_dims({'time':1})
        
        return filtered

