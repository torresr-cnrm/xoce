"""
"""

import numpy as np
import xarray as xr

from ..api.generic import XoceObject


class Integral(XoceObject):
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
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset

        for dim in self.dims:
            if dim not in ds.coords:
                raise Exception("Integral error: '{}' ".format(dim) + 
                            "is not in dataset coordinates {}".format(list(ds.coords)))

        for var in self.variables:
            if var not in ds.variables:
                raise Exception("Integral error: '{}' ".format(var) + 
                        "is not in dataset variables {}".format(list(ds.variables)))

        # coordinates selection
        coords = dict()
        if self.dims:
            coords = {c: ds.coords[c] for c in ds.coords if c not in self.dims}
        integrated = xr.Dataset(coords=coords)
        
        # sanity check for dims and variables list
        if not self.dims:
            dims = None
        else:
            dims = self.dims

        if not self.variables :
            variables = [v for v in ds.variables if v not in ds.coords]
        else:
            variables = self.variables
        
        # processing: integration over all dimensions
        for var in variables:
            if 'depth' in ds[var].dims:
                cell_coefs = ds['volume']
            else:
                cell_coefs = ds['surface']

            # keep nan to get masked arrays
            vals  = (cell_coefs*ds[var]).sum(dim=dims)
            conds = np.isnan(ds[var].sum(dim=dims, skipna=False))
            conds = conds & (ds[var].sum(dim=dims, skipna=True) == 0.)
            integrated[var] = xr.where(conds, np.nan, vals)

        return integrated

