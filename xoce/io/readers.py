"""
"""

import h5py
import os
import xarray as xr

from ..api.experiment import Experiment
from ..api.generic import XoceObject


class H5pyReader(XoceObject):
    """
    Read an h5py file from xoce post-processing and return a dataset or an 
    experiment object.
    """
    _Parameters = {
        "path": {'type': str,
                'default': None},
        "filename": {'type': str,
                 'default': None},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)
        
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
        if not os.path.isfile(realname):
            return datasets

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
                # read data
                datas = grp[v][()]

                # read dimensions
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

                # read coordinates
                if 'coordinates' in grp[v].attrs:
                    coordinates = grp[v].attrs['coordinates'].split()
                    coords = dict()
                    for c in coordinates:
                        cname = c.decode('utf-8')
                        coords[cname] = grp[cname][()]
                else:
                    coords = None

                da = xr.DataArray(datas, coords=coords, dims=dims)
                try:
                    ds[v] = da
                except xr.MergeError:
                    pass
            
            datasets.append(ds)
        f.close()
        
        return datasets

