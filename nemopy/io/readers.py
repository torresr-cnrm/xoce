"""
"""

import h5py
import os
import xarray as xr

from ..api.experiment import Experiment
from ..api.generic import NemopyObject


class H5pyReader(NemopyObject):
    """
    Read an h5py file from nemopy post-processing and return a dataset or an 
    experiment object.
    """
    _Parameters = {
        "path": {'type': str,
                'default': None},
        "filename": {'type': str,
                 'default': None},
    }

    def __init__(self, dataset=None, **kargs):
        NemopyObject.__init__(self, dataset)
        
        # add default processing parameter
        self._set_default_parameters(**kargs)

    
    def read(self):
        if not os.path.isdir(self.path):
            raise Exception("ReaderError: '{}' is not a directory".format(self.path))

        if self.dataset is None:
            datasets = list()
        else:
            datasets = [self.dataset]
        
        realname = os.path.join(self.path, self.filename)
        f = h5py.File(realname, 'r')

        # read one dataset for each group
        for k in f.keys():
            ds = xr.Dataset()
            ds.attrs['name'] = k
            grp = f[k]
            for at in grp.attrs:
                ds.attrs[at] = grp.attrs[at]
        
            # get variables
            for v in grp.keys():
                datas = grp[v][()]
                dims  = list()
                for i, dim in enumerate(grp[v].dims):
                    dname = dim.label
                    if not dname:
                        dname = 'dim_{}'.format(i)
                        if dname in ds.dims:
                            for d in ds.dims:
                                if grp[v].shape[i] == ds.dims[d]:
                                    dname = d
                    dims.append(dname)

                da = xr.DataArray(datas, dims=dims)
                try:
                    ds[v] = da
                except xr.MergeError:
                    pass
            
            datasets.append(ds)
        f.close()
        
        return datasets

