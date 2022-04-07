"""
"""

import numpy as np
import xarray as xr

from ..utils.dataset_util import compare_coords


class Experiment:
    def __init__(self, path=None, fmesh=None):
        self.path = path 
        self.fmesh = fmesh

        # protected properties
        self._dataset = xr.Dataset()
        self._mesh = xr.Dataset()

    
    def __getitem__(self, v):
        if v in self.variables():
            if v in self._dataset.variables:
                return self._dataset[v]
            elif v in self._mesh.variables:
                return self._mesh[v]
        else:
            raise KeyError("'{}' is not a variable of the experiment.\\".format(v) +
                           "Available variables: {}".format(self.variables))

    @property
    def dims(self):
        return dict(self._dataset.dims)

    @property
    def variables(self):
        return list(self._dataset.variables) + list(self._mesh.variables)

    def calculate(self, var):
        pass

    # data loading function
    def load(self, chunks=None):
        ds = xr.open_mfdataset(self.path, chunks=chunks)
        mesh = xr.open_dataset(self.fmesh)

        diff_coords, _ = compare_coords(mesh.coords, ds.coords)
        for c in diff_coords:
            mdiff = max(np.abs(diff_coords[c])).data
            if 0 < max(np.abs(diff_coords[c])) < 1e-6:
                print("Warning: mesh coordinate '{}' ".format(c) + 
                      "is not exactly the same than initial dataset (max diff = {:.8f}).".format(mdiff))
                print("Mesh coordinates updated.")
                mesh.coords.update({c: ds.coords[c]})

        self._dataset = ds
        self._mesh = mesh
