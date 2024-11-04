"""
"""

import h5py
import os
import numpy  as np
import xarray as xr

from ..api.generic import XoceObject
from ..utils.datetime_util import datetime_to_cftime


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
                
                vcls  = grp[v].attrs.get('CLASS', '')
                if isinstance(vcls, (bytes)):
                    vcls = vcls.decode('utf-8')
                
                if vcls == 'DIMENSION_SCALE':
                    continue
                
                # read data
                datas = grp[v][()]
                if 'datetime' in datas.dtype.__str__():
                    datas = datetime_to_cftime(datas)

                # read dimensions
                dims = list()
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
                        cname = c
                        if isinstance(c, (bytes)):
                            cname = c.decode('utf-8')

                        cdats = grp[cname][()]
                        if 'datetime' in cdats.dtype.__str__():
                            cdats = datetime_to_cftime(cdats)
                        
                        if 'DIMENSION_LABELS' in grp[cname].attrs:
                            cdims = list(grp[cname].attrs['DIMENSION_LABELS'])
                            cdims = [str(d) for d in cdims]
                        else:
                            cdims = [cname]

                        if not isinstance(cdats, (list, np.ndarray)):
                            cdats = np.array([cdats])

                        coords[cname] = xr.DataArray(cdats, name=cname, dims=cdims)

                        if cname not in ds.coords:
                            ds.coords[cname] = coords[cname]

                else:
                    coords = None
                
                if not isinstance(datas, (list, np.ndarray)):
                    datas = np.array([datas])
                    dims  = tuple(list(coords))
                    if len(dims) == 0:
                        dims = (v,)
                
                co2del = list()
                for co in coords:
                    for d in coords[co].dims:
                        if d not in dims:
                            co2del.append(co)
                for co in co2del:
                    del coords[co]

                da = xr.DataArray(datas, coords=coords, dims=dims)
                try:
                    ds[v] = da
                except xr.MergeError:
                    pass
            
            datasets.append(ds)
        f.close()
        
        return datasets

