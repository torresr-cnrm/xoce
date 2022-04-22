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

            print('dataset created')
            # create coordinates subgroup
            crds = grp.create_group('coordinates')
            for co in ds.coords:
                arr = ds.coords[co]
                if co in ['time']:
                    dtype = h5py.opaque_dtype(arr.dtype)
                    datas = arr.data.astype(dtype)
                else:
                    dtype = arr.dtype
                    datas = arr.data
                var = crds.create_dataset(co, arr.shape, dtype, datas)
                for at in arr.attrs:
                    var.attrs[at] = arr.attrs[at]

            print('coordinates created')
            # create variables subgroup
            vrs = grp.create_group('variables')
            for v in ds.variables:
                print(v)
                print(ds.variables[v].shape)
                if v in ds.coords:
                    continue
                arr = ds.variables[v]
                nda = arr.data.compute()
                var = vrs.create_dataset(v, arr.shape, arr.dtype, nda)
                for at in arr.attrs:
                    var.attrs[at] = arr.attrs[at]

            print('variables created')
        f.close()

