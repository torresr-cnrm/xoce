"""
"""

import cftime
import xarray as xr

from ..api.generic import NemopyObject
from ..utils.datetime_util import datetime_mean


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
                    'default': False}, 
        "variables": {'type': list,
                      'default': None},
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
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

                # add filtered variable
                if self.inverse:
                    filtered[v] = ds[v] - vmean
                else:
                    filtered[v] = vmean.expand_dims({'time':1})

        return filtered

