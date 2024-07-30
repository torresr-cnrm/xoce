"""
"""

import h5py
import os
import numpy as np
import warnings
import xoce

from ..api.generic import XoceObject
from ..utils.datetime_util import datetime_to_cftime


def iowrapper(func):
    """Writer decorator."""
    def inner(obj):
        try:
            func(obj)
        except Exception as e:
            # find a way to close file if error occur during reading/writing
            # could be dangerous if file exists..
            pass
    
    return inner

def ioloopwrapper(func):
    """Decorator for detecting."""
    def inner(obj):
        try:
            func(obj)
        except Exception as e:
            # find a way to close file if error occur during reading/writing
            # could be dangerous if file exists..
            pass
    
    return inner


class H5pyWriter(XoceObject):
    """
    Write a dataset (or a list of datasets) in a hdf5 file.
    """
    _Parameters = {
        "path": {'type': str,
                'default': None},
        "filename": {'type': str,
                 'default': None},
        "expname": {'type': str,
                    'default': 'experiment'},
        "variables": {'type': list,
                    'default': list()},
        "reduce_mem": {'type': bool,
                       'default': False},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)

    
    def write(self):
        if not os.path.isdir(self.path):
            raise Exception("WriterError: '{}' is not a directory".format(self.path))

        if not isinstance(self.dataset, (list, tuple, set)):
            datasets = [self.dataset]
        else:
            datasets = list(self.dataset)
        
        realname = os.path.join(self.path, self.filename)
        f = h5py.File(realname, 'w')
        f.attrs['xoce_version'] = xoce.__version__

        for ds in datasets:
            name = ds.attrs.get('name', 'dataset')
            i = 0 
            while name+'{}'.format(i).zfill(2) in f.keys():
                i += 1

            # create one group for each dataset
            grp = f.create_group(name)
            for at in ds.attrs:
                grp.attrs[at] = ds.attrs[at]

            # create coordinates
            for co in ds.coords:
                arr = ds.coords[co]

                if co in ['time', 't']:
                    if arr.dtype == 'O':
                        try:
                            arr[co] = arr.indexes[co].to_datetimeindex()
                            dtype   = h5py.opaque_dtype(arr[co].dtype)
                       
                        except (AttributeError, ValueError):
                            dat     = np.arange(arr.size).reshape(arr.shape)
                            dat     = dat.astype(dtype='datetime64[s]')
                            arr     = arr.assign_coords( {co: datetime_to_cftime(dat)} )

                            arr[co] = arr.indexes[co].to_datetimeindex()
                            dtype   = h5py.opaque_dtype(arr[co].dtype)
                    else:
                        dtype = h5py.opaque_dtype(arr.dtype)
                    arr = arr.astype(dtype)

                try:
                    self._write_variable(co, arr, grp)
                except Exception:
                    warnings.warn("Error occured while writing the variable " +
                                    "'{}' (ignored).".format(co))
                
            # create variables
            for v in ds.variables:
                if not self.variables or v in self.variables:
                    if v in ds.coords:
                        continue

                    arr = ds[v]
                    if self.reduce_mem and arr.dtype == 'float64':
                        arr = arr.astype('float32')
                    
                    try:
                        self._write_variable(v, arr, grp)
                    except Exception:
                        warnings.warn("Error occured while writing the variable " +
                                      "'{}' (ignored).".format(v))

        f.close()


    def _write_variable(self, name, array, hdf_group):
        """Write a new dataset which correspond to a dimension. Note if 
        the dimension already exists, then the function simply return it.
        """

        var = hdf_group.create_dataset(name, array.shape, array.dtype, array.data)

        for at in array.attrs:
            var.attrs[at] = array.attrs[at]
        
        # attach scale for each dimension
        for ind, dim in enumerate(array.dims):
            vdim = self._write_dim(dim, array[dim], hdf_group)
            var.dims[ind].label = dim
            if dim != name:
                var.dims[ind].attach_scale(vdim)

        # store coordinates in attributes
        if list(array.coords):
            acoords = [c for c in array.coords if array.coords[c].shape]
            var.attrs['coordinates'] = ' '.join(acoords)

        return var


    def _write_dim(self, name, array, hdf_group):
        """Write a new dataset which correspond to a dimension. Note if 
        the dimension already exists, then the function simply return it.
        """
        if name in hdf_group:
            var = hdf_group[name]
        else:
            dtype = array.dtype
            datas = array.data
            var = hdf_group.create_dataset(name, array.shape, dtype, datas)

        var.make_scale()

        return var

