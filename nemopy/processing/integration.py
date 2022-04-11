"""
"""

import xarray as xr

from .processing import Processing


class Integral(Processing):
    """
    Integration of some datas over one specified dimension 
    or the whole space.
    """
    _Parameters = {
        "dims": {'type': list,
                'default': list()},
        "variables": {'type': list,
                      'default': list()},
    }

    def __init__(self, dataset=None, **kargs):
        Processing.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset

        for dim in self.dims :
            if dim not in ds.coords:
                raise Exception("Integral error: '{}' ".format(dim) + 
                            "is not in dataset coordinates {}".format(list(ds.coords)))

        for var in self.variables :
            if var not in ds.variables:
                raise Exception("Integral error: '{}' ".format(var) + 
                        "is not in dataset coordinates ()".format(list(ds.variables)))

        # coordinates selection
        coords = {c: ds.coords[c] for c in ds.coords if c not in self.dims}
        for dim in self.dims:
            dimval = ds[self.dim]
            newdim = xr.IndexVariable(dim, [dimval.mean(dim=dim).data])
            coords.update({dim: newdim})

        integrated = xr.Dataset(coords=coords)
        
        # sanity check for variables list
        if not self.variables :
            variables = [v for v in ds.variables if v not in ds.coords]
        else:
            variables = self.variables
        
        # processing: integration over all dimensions
        volcello = ds['volume']
        for var in variables:
            integrated[var] = (volcello*ds[var]).sum()

        return integrated

