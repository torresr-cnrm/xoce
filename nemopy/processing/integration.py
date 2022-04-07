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
        "method": {'type': str,
                   'default': 'arithmetic',
                   'restricted': ['arithmetic', 'volume']}
    }

    def __init__(self, dataset=None, **kargs):
        Processing.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def execute(self):
        ds = self.dataset

        for dim in self.dims :
            if dim not in ds.coords:
                raise Exception("BoxClipper error: '{}' ".format(dim) + 
                            "is not in dataset coordinates ()".format(list(ds.coords)))

        for var in self.variables :
            if var not in ds.variables:
                raise Exception("BoxClipper error: '{}' ".format(var) + 
                        "is not in dataset coordinates ()".format(list(ds.variables)))

        if self.method == 'volume':
            pass

        return None

