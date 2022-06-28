"""
"""


import copy
import os
import numpy as np
import xarray as xr

from ..calc import CalcManager
from ..utils.dataset_util import merge_coordinates
from ..utils.io_util import extract_cmip6_variables, get_filename_from_drs
from ..utils.io_util import load_cmip6_output

from . import _VARS_NAME



class Experiment:
    def __init__(self, path=None, fmesh=None):
        self.path = path 
        self.fmesh = fmesh

        # core dataset properties
        self._dims = dict()
        self._coords = dict()
        self._chunks = None                       # dask chunks to split large datasets

        self._arrays = dict()                     # dict-like (could be xr.Dataset) obj
        self._mesh = xr.Dataset()                 # dict-like (should be xr.Dataset) obj
        self._calc = CalcManager(dataset=self)    # instance to compute off-line diag.

        # loading options
        self._unused_dims = list()


    def __getitem__(self, var):
        if var in self.variables:
            if var in self._arrays:
                variable = self._arrays[var]
            elif var in self._mesh.variables:
                variable = self._mesh[var]
            else:
                lres = self.load_variable(var, chunks=self._chunks)
                if lres is not None:
                    variable = self._arrays[var]
                else:
                    raise KeyError("'{}' not found in the experiment.".format(var))
            
            for dim in self._unused_dims:
                if dim in variable.dims:
                    variable = variable.isel({dim:0})
            self.add_variable(var, variable)

            return self._arrays[var]
        else:
            if self._calc.is_calculable(var):
                return self.calculate(var)
            else:
                raise KeyError("'{}' is not a variable of the experiment.".format(var) +
                            "Available variables: {}".format(self.variables))

    def __setitem__(self, var, values):
        # TODO: add test on dimensions
        if isinstance(values, xr.DataArray):
            self.add_variable(var, values)
        else:
            raise TypeError("Values should be a DataArray not {}".format(type(values)))


    # abstract method(s)
    def load(self, chunks=None):
        raise Exception("'load' function not implemented.")


    @property
    def dims(self):
        return self._dims

    @property
    def coords(self):
        return self._coords

    @property
    def variables(self):
        return list(self._arrays) + list(self._mesh)

    def calculate(self, var):
        return self._calc.calculate(var)

    def where(self, conds, other=np.nan, drop=False):
        dataset = xr.Dataset(coords=self.coords)
        for v in self.variables:
            if v not in dataset.dims:
                dataset[v] = self[v].where(conds, other, drop)
        
        return dataset

    def add_variable(self, var, arr, rename_dims=True):
        if var in list(_VARS_NAME[type(self).__name__].keys()):
            newvar = _VARS_NAME[type(self).__name__][var]
        else:
            newvar = var

        if rename_dims:
            rename_dict = dict()
            for vn in _VARS_NAME[type(self).__name__]:
                if vn in list(arr.dims) + list(arr.coords):
                    rename_dict[vn] = _VARS_NAME[type(self).__name__][vn]
            arr = arr.rename(rename_dict)

            if arr.name in _VARS_NAME[type(self).__name__]:
                arr.name = _VARS_NAME[type(self).__name__][arr.name]
        
        self._arrays[newvar] = arr
        
        return newvar, arr

    def load_variable(self, var, chunks=None):
        """
        For Experiment instance using lazy loading and which all DataArrays are
        not directly loaded.
        """
        pass


class SingleDatasetExperiment(Experiment):
    def __init__(self, path=None, fmesh=None):
        super().__init__(path, fmesh)


    # abstract method(s) definition
    def load(self, chunks=None):
        ds = xr.open_mfdataset(self.path, chunks=chunks)
        if self.fmesh:
            mesh = xr.open_dataset(self.fmesh)

        code_info = merge_coordinates(mesh, ds.coords)
        if code_info == -1:
            print("Warning: mesh and dataset coordinates are not everywhere equal.")

        self._arrays = ds
        self._mesh = mesh


class CMIPExperiment(Experiment):
    """
    Experiment data container based on CMIP6 protocole.

    TODO: create a parent class for all experiments to avoid duplicate
          (maybe change the name 'experiment' ??)
    """
    def __init__(self, path=None, fmesh=None):
        super().__init__(path, fmesh)

        # child properties
        self._drs  = dict()         # data reference syntax: variableID_tableID_ .. .nc


    # abstract method(s) definition
    def load(self, chunks=None):
        self._chunks = chunks
        self._drs = load_cmip6_output(self.path)
        if self.fmesh:
            self._mesh = xr.open_dataset(self.fmesh)


    @property
    def variables(self):
        return super().variables + self._drs.get('variable_id', [])

    def load_variable(self, var, chunks=None):
        if not self._drs :
            self._drs = load_cmip6_output(self.path)

        if var not in self._drs['variable_id']:
            raise Exception("No file match `variable_id = {}`".format(var) + 
                            "in directory: {}".format(self.path))
        
        fname = get_filename_from_drs(var, self._drs)
        abspath = os.path.join(self.path, fname)
        ds = xr.open_dataset(abspath, chunks=chunks)

        # update experiment dims and coords
        for d in ds.dims:
            if d not in self.dims:
                self._dims[d] = ds.dims[d]
        
        for c in ds.coords:
            if c not in self.coords:
                newc, datc = self.add_variable(c, ds.coords[c], rename_dims=True)
                self._coords[newc] = datc
        
        # finally link DataArray in a container
        self.add_variable(var, ds[var])
        
        return ds[var]

    def extract_vars(self, variables):
        """
        return only a limited variables list stored in its _drs property.
        """   
        ndrs = extract_cmip6_variables(variables, 'variable_id', self._drs)
        experiment = copy.deepcopy(self)
        experiment._drs = ndrs

        return experiment

