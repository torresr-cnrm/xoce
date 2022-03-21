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
        self._dataset = None
        self._mesh = None

    @property
    def dims(self):
        return dict(self._dataset.dims)

    @property
    def variables(self):
        return list(self._dataset.variables) + list(self._mesh.variables)

    def load(self):
        ds = xr.open_mfdataset(self.path, chunks={'depth': 10})
        mesh = xr.open_dataset(self.fmesh)

        diff_coords, _ = compare_coords(mesh.coords, ds.coords)
        for c in diff_coords:
            mdiff = max(np.abs(diff_coords[c])).data
            if max(np.abs(diff_coords[c])) < 1e-6:
                print("Warning: mesh coordinate '{}' ".format(c) + 
                      "is not exactly the same than initial dataset (max diff = {:.6}).".format(mdiff))
                print("Mesh coordinates updated.")
                mesh.coords.update({c: ds.coords[c]})

        self._dataset = ds
        self._mesh = mesh

    def average(self, vars, dim, period):
        """
        Explain averaged processing
        """
        pass
        