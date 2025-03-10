"""
"""


from codecs import decode
import copy
import os
import numpy as np
import xarray as xr

from ..calc import CalcManager
from ..utils.dataset_util import check_dims, get_dim_axis 
from ..utils.datetime_util import decode_months_since
from ..utils.io_util import extract_cmip6_variables, get_filename_from_drs
from ..utils.io_util import load_cmip6_output

from . import _DIM_COORDINATES, _VARS_NAME



class Experiment:
    def __init__(self, path=None, interpolation='linear'):
        self.path  = path 
        self.fmesh = None

        # core dataset properties
        self._dims   = dict()
        self._coords = dict()
        self._attrs  = dict()
        self._chunks = None                       # dask chunks to split large datasets

        self._arrays = dict()                     # dict-like (could be xr.Dataset) obj
        self._mesh = xr.Dataset()                 # dict-like (should be xr.Dataset) obj
        self._calc = CalcManager(dataset=self)    # instance to compute off-line diag.

        # loading and array options
        self.interpolation = interpolation
        self._dtype        = None                 # numerical data type (eg: float)
        self._decode_times = None                 # decode dim 'times' when loading files
        self._unused_dims  = list()


    def __getitem__(self, var):
        if var in self.variables:
            if var in self.arrays:
                array = self.arrays[var]
            elif var in self._mesh.variables:
                array = self._mesh[var]
            elif var in self.coords:
                array = self.coords[var]
            else:
                lres = self.load_variable(var, chunks=self._chunks, 
                                               decode_times=self._decode_times)
                if lres is not None:
                    array = self.arrays[var]
                else:
                    raise KeyError("'{}' not found in the experiment.".format(var))

        else:
            if not self._calc.is_calculable(var):
                raise KeyError("'{}' is not a variable of the experiment.".format(var) +
                    "Available variables: {}".format(self.variables))
            else:
                array = self.calculate(var)

        # remove unused dim
        for dim in self._unused_dims:
            if dim in array.dims:
                array = array.isel({dim:0})

        # interpolation if needed  
        if not self.interpolation:
            return array
        else:
            method = self.interpolation
            for d in array.dims :
                c = _DIM_COORDINATES.get(d, d)
                if c in self.coords:
                    if not np.all(array[d].data == self.coords[c][d].data):
                        array = array.interp(**{d: self.coords[c]}, method=method,
                                            kwargs={"fill_value": "extrapolate"})

        self.add_variable(var, array)
        array = self.arrays[var]
        
        if self.dtype and array.dtype != self.dtype:
            print(array.name, self.dtype)
            try:
                array = array.astype(self.dtype)
            except TypeError:
                pass

        return array


    def __setitem__(self, var, values):
        # TODO: add test on dimensions
        if isinstance(values, xr.DataArray):
            self.add_variable(var, values)
        else:
            raise TypeError("Values should be a DataArray not {}".format(type(values)))


    # abstract method(s)
    def load(self, chunks={}, replace_dict={}):
        raise Exception("'load' function not implemented.")

    def load_mesh(self, fmesh, replace_dict={}, rename={}):
        if not os.path.isfile(fmesh):
            raise Exception("Mesh file '{}' not found.".format(fmesh))
        
        # open the mesh dataset
        mesh = xr.open_dataset(fmesh)

        # replace some variables values
        for var in replace_dict:
            newvar = replace_dict[var]
            inside = (var in mesh or var in mesh.dims)
            inside = inside & (newvar in mesh or newvar in mesh.dims)
            if inside:
                mesh = mesh.assign({var: mesh[newvar]})

        if rename:
            mesh = mesh.rename(**rename)

        # save object attributes
        self._mesh = mesh
        self.fmesh = fmesh


    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        try:
            np.dtype(value)
        except TypeError as e:
            raise TypeError("data type '{}' not understood.".format(value))

        self._dtype = value

    @property
    def dims(self):
        return self._dims

    @property
    def coords(self):
        return self._coords

    @property
    def attrs(self):
        return {}

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value

    @property
    def variables(self):
        return list( set( list(self._arrays) + list(self._mesh) + list(self.coords) ) ) 


    def calculate(self, var):
        return self._calc.calculate(var)

    def where(self, conds, other=np.nan, drop=False):
        coords = dict()
        for c in self.coords:
            if set(conds.dims) <= set(self.coords[c].dims):
                if drop:
                    coords[c] = self.coords[c].where(conds, drop=drop)
                else:
                    coords[c] = self.coords[c].where(conds, other, drop=drop)
            else:
                coords[c] = self.coords[c]

        dataset = xr.Dataset(coords=coords)
        
        for v in self.variables:
            if v not in dataset.dims:  
                # check dimensions shape and size
                if set(conds.dims) <= set(self[v].dims):
                    indx, skpd = get_dim_axis(self, self[v].dims, skip_notfound=True)
                    
                    var_shpe = np.delete(self[v].shape, skpd)
                    shpe     = np.take(list(self.dims.values()), indx)
                    
                    if np.all(shpe == var_shpe):
                        if drop:
                            arr = self[v].where(conds, drop=drop)
                        else:
                            arr = self[v].where(conds, other, drop=drop)
                        dataset[v] = (arr.dims, arr.data)
                        dataset[v].attrs = arr.attrs

                elif check_dims(self[v], dataset.dims):
                    arr = self[v]
                    dataset[v] = (arr.dims, arr.data)
                    dataset[v].attrs = arr.attrs
        
        return dataset

    def rename(self, name_dict=None, **names):
        if isinstance(name_dict, dict):
            names = name_dict
        
        for name in names:
            if name in self.arrays:
                if isinstance(self.arrays, dict):
                    for v in self.arrays:
                        if name in self.arrays[v]:
                            self.arrays[v] = self.arrays[v].rename(**{name:names[name]})
                else:
                    self.arrays = self.arrays.rename(**{name:names[name]})
            if name in self._coords:
                self._coords[names[name]] = self._coords[name]
                del self._coords[name]

    def rename_dims(self, name_dict=None, **names):
        if isinstance(name_dict, dict):
            names = name_dict
        
        for name in names:
            if isinstance(self.arrays, dict):
                for v in self.arrays:
                    if name in self.arrays[v].dims:
                        self.arrays[v] = self.arrays[v].rename(**{name:names[name]})
            else:
                self.arrays = self.arrays.rename_dims(**{name:names[name]})

        for co in self._coords:
            for name in names:
                if name in self._coords[co].dims:
                    newc = self._coords[co].rename(**{name:names[name]})
                    newc.coords[name] = newc.coords[names[name]]
                    del newc.coords[names[name]]

                    self._coords[name] = newc

    def add_variable(self, var, arr, rename_dims=True, assign=False):
        """
        Add new variable in the private self.arrays dictionary.
        If the variable already exists, the function simply return 
        the desired array. 
        Warning: special treatment are made to get the real variable 
        name..
        """
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

        if newvar not in list(self.arrays):
            self.arrays[newvar] = arr
        elif assign:
            if isinstance(self.arrays, xr.Dataset):
                self.arrays = self.arrays.assign({newvar: arr})
            else:
                self.arrays[newvar] = arr
        else:
            arr = self.arrays[newvar]
        
        return newvar, arr

    def add_coordinate(self, var, arr, rename_dims=True, assign=False):
        """
        Add new coordinate in the private self._coords dictionary.
        If the variable already exists, the function either return the
        existing desired array if assign is False or replace the coordinate
        if assign is True. 
        Warning: special treatment are made to get the real variable 
        name..
        """           

        if var in list(_VARS_NAME[type(self).__name__].keys()):
            newvar = _VARS_NAME[type(self).__name__][var]
        else:
            newvar = var

        if isinstance(arr, np.ndarray):
            arr = xr.DataArray(arr, dims=(newvar))

        if rename_dims:
            rename_dict = dict()
            for vn in _VARS_NAME[type(self).__name__]:
                if vn in list(arr.dims) + list(arr.coords):
                    rename_dict[vn] = _VARS_NAME[type(self).__name__][vn]
            arr = arr.rename(rename_dict)

            if arr.name in _VARS_NAME[type(self).__name__]:
                arr.name = _VARS_NAME[type(self).__name__][arr.name]

        if newvar not in self._coords:
            self._coords[newvar] = arr
        elif assign:
            if isinstance(self._coords, xr.core.coordinates.DatasetCoordinates):
                ds_coords = self._coords.to_dataset().assign_coords({newvar: arr})
                self._coords = ds_coords.coords
            else:
                self._coords[newvar] = arr
        else:
            arr = self._coords[newvar]
        
        # interpolate arrays if need
        if isinstance(self.arrays, xr.Dataset) and newvar in self.arrays.coords:
            self.arrays = self.arrays.interp({newvar: arr}, method=self.interpolation,
                                            kwargs={"fill_value": "extrapolate"})
        
        return newvar, arr

    def load_variable(self, var, chunks=None):
        """
        For Experiment instance using lazy loading and which all DataArrays are
        not directly loaded.
        """
        pass


class SingleDatasetExperiment(Experiment):
    def __init__(self, path=None):
        super().__init__(path)

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, ds):
        self._arrays = ds
        self._coords = ds.coords
        self._dims   = ds.dims

    # abstract method(s) definition
    def load(self, chunks={}, replace_dict={}):
        """Loading output files."""
        try :
            ds = xr.open_mfdataset(self.path)
        except ValueError:
            ds = xr.open_mfdataset(self.path, decode_times=False)
            ds = ds.assign_coords( {'time': decode_months_since(ds['time'])} )
        
        ds = ds.chunk(chunks)

        # replace some variables values
        for var in replace_dict:
            newvar = replace_dict[var]
            inside = (var in ds or var in ds.dims)
            inside = inside & (newvar in ds or newvar in ds.dims)
            if inside:
                ds = ds.assign({var: ds[newvar]})

        # rename some vars, coords or dims
        rename_dict = dict()
        for v, n in _VARS_NAME[type(self).__name__].items():
            if v in ds:
                rename_dict[v] = n
        ds = ds.rename(rename_dict)

        # add into object placeholders
        self._arrays = ds
        self._coords = ds.coords
        self._dims   = ds.dims



class CMIPExperiment(Experiment):
    """
    Experiment data container based on CMIP6 protocole.
    """
    def __init__(self, path=None):
        super().__init__(path)

        # child properties
        self._drs  = dict()         # data reference syntax: variableID_tableID_ .. .nc


    # abstract method(s) definition
    def load(self, chunks={}, replace_dict={}):
        """Loading output files."""

        self._chunks = chunks
        self._drs = load_cmip6_output(self.path)


    @property
    def variables(self):
        return list( set( super().variables + self._drs.get('variable_id', []) ) )

    def load_variable(self, var, chunks={}, decode_times=None):
        if decode_times is False:
            self._decode_times = False
        
        if not self._drs :
            self._drs = load_cmip6_output(self.path)

        if var not in self._drs['variable_id']:
            raise Exception("No file match `variable_id = {}`".format(var) + 
                            "in directory: {}".format(self.path))
        
        var_drs = extract_cmip6_variables([var], 'variable_id', self._drs)
        var_tr  = sorted(var_drs['time_range'])
        var_ind = var_drs['time_range'].index(var_tr[0])

        fname   = get_filename_from_drs(var, {k: [var_drs[k][var_ind]] for k in var_drs})

        abspath = os.path.join(self.path, fname[0])
        ds = xr.open_dataset(abspath, chunks=chunks, decode_times=decode_times)

        # -- concat time_range if necessary
        for tr in var_tr[1:]:
            i       = var_drs['time_range'].index(tr)
            fname   = get_filename_from_drs(var, {k: [var_drs[k][i]] for k in var_drs})

            abspath = os.path.join(self.path, fname[0])
            new_ds  = xr.open_dataset(abspath, chunks=chunks, decode_times=decode_times)

            ds      = xr.concat( (ds, new_ds), dim='time')
        
        # update experiment dims and coords
        for d in ds.dims:
            if d not in self.dims:
                self._dims[d] = ds.dims[d]
        
        for c in ds.coords:
            if c not in self.coords:
                _ = self.add_coordinate(c, ds.coords[c], rename_dims=True)
        
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

