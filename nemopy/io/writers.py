"""
"""

import h5py
import os

from ..api.generic import NemopyObject


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


class H5pyWriter(NemopyObject):
    """
    Write a dataset (or a list of datasets) in a hdf5 file.
    """
    _Parameters = {
        "path": {'type': str,
                'default': None},
        "filename": {'type': str,
                 'default': None},
        "expname": {'type': str,
                    'default': 'experiement'},
        "variables": {'type': list,
                    'default': list()},
        "reduce_mem": {'type': bool,
                       'default': False},
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
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
                if co in ds.dims:
                    continue
                arr = ds.coords[co]
                if co in ['time', 't']:
                    if arr.dtype == 'O':
                        arr[co] = arr.indexes[co].to_datetimeindex()
                        dtype = h5py.opaque_dtype(arr[co].dtype)
                    else:
                        dtype = h5py.opaque_dtype(arr.dtype)
                    datas = arr.data.astype(dtype)
                else:
                    dtype = arr.dtype
                    datas = arr.data
                var = grp.create_dataset(co, arr.shape, dtype, datas)
                for at in arr.attrs:
                    var.attrs[at] = arr.attrs[at]
                
                # attach scale for each dimension
                for ind, dim in enumerate(ds[co].dims):
                    vdim = self._write_dim(dim, ds[co][dim], grp)
                    var.dims[ind].attach_scale(vdim)
                
            # create variables
            for v in ds.variables:
                if not self.variables or v in self.variables:
                    if v in ds.coords:
                        continue

                    arr = ds.variables[v]
                    if self.reduce_mem and arr.dtype == 'float64':
                        arr = arr.astype('float32')
                    var = grp.create_dataset(v, arr.shape, arr.dtype, arr.data)

                    for at in arr.attrs:
                        var.attrs[at] = arr.attrs[at]

                    # attach scale for each dimension
                    for ind, dim in enumerate(ds[v].dims):
                        vdim = self._write_dim(dim, ds[v][dim], grp)
                        var.dims[ind].label = dim
                        var.dims[ind].attach_scale(vdim)

        f.close()


    def _write_dim(self, name, array, hdf_group):
        """Write a new dataset which correspond to a dimension. Note if 
        the dimension already exists, then the function simply return it.
        """
        if name in hdf_group:
            return hdf_group[name]

        if name in ['time', 't']:
            if array.dtype == 'O':
                array[name] = array.indexes[name].to_datetimeindex()
                dtype = h5py.opaque_dtype(array[name].dtype)
            else:
                dtype = h5py.opaque_dtype(array.dtype)
            datas = array.data.astype(dtype)
        else:
            dtype = array.dtype
            datas = array.data
        
        var = hdf_group.create_dataset(name, array.shape, dtype, datas)
        var.make_scale()

        return var

