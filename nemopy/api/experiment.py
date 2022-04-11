"""
"""

import numpy as np
import xarray as xr

from ..calc import CalcManager
from ..utils.dataset_util import merge_coordinates


class Experiment:
    def __init__(self, path=None, fmesh=None):
        self.path = path 
        self.fmesh = fmesh

        # protected properties
        self._dataset = xr.Dataset()
        self._mesh = xr.Dataset()
        self._calc = CalcManager(dataset=self)

    
    def __getitem__(self, var):
        if var in self.variables:
            if var in self._dataset.variables:
                return self._dataset[var]
            elif var in self._mesh.variables:
                return self._mesh[var]
        else:
            if self._calc.is_calculable(var):
                return self.calculate(var)
            else:
                raise KeyError("'{}' is not a variable of the experiment.\\".format(var)+
                            "Available variables: {}".format(self.variables))

    def __setitem__(self, var, values):
        # TODO: add test on dimensions
        if isinstance(values, xr.DataArray):
            self._dataset[var] = values
        else:
            raise TypeError("Values should be a DataArray not {}".format(type(values)))

    @property
    def dims(self):
        return dict(self._dataset.dims)

    @property
    def coords(self):
        return self._dataset.coords

    @property
    def variables(self):
        return list(self._dataset.variables) + list(self._mesh.variables)

    def calculate(self, var):
        return self._calc.calculate(var)

    # data loading function
    def load(self, chunks=None):
        ds = xr.open_mfdataset(self.path, chunks=chunks)
        mesh = xr.open_dataset(self.fmesh)

        code_info = merge_coordinates(mesh, ds.coords)
        if code_info == -1:
            print("Warning: mesh and dataset coordinates are not everywhere equal.")

        self._dataset = ds
        self._mesh = mesh
