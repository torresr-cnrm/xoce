"""
Define functions and methods for averaging datasets. 

Note: these functions should probably be a xoce.Processing class
since they have dataset as input and return a new one. Next version ?
"""

import numpy as np
import xarray as xr

from xoce.processing import Selector
from .dataset_util import split_dataset
from .datetime_util import datetime_mean


def zonal_average(dataset, axis='longitude', res_axis='latitude', res_dim='y'):
    """
    """
    averaged = xr.Dataset(coords=dataset.coords)
    
    # if dim is a dataset dimensions: use xarray mean function
    if axis in dataset.dims:
        for v in dataset.variables:
            if axis in dataset[v].dims:
                try:
                    averaged[v] = dataset[v]
                except xr.MergeError:
                    pass
            
                dt = dataset[v].dtype
                if v in ('time', 't') and dt == 'O':
                    averaged[v] = datetime_mean(dataset[v], axis)
                else:
                    averaged[v] = dataset[v].mean(dim=axis)
    
    # if not, dim is a coordinate which could vary with dimensions
    else:
        # TODO: add check on axis and res_axis
        # select only resulting dimension
        fselector = Selector('field', dataset=dataset)
        fselector.set('variables', [res_axis])
        latitudes = fselector.execute()

        varname = "{}_variable".format(res_axis)
        latitudes[varname] = latitudes[res_axis]

        # split original and latitudes datasets
        split_bounds = 0.75*np.arange(226) - 79.5
        bands = list()
        for i in range(len(split_bounds)-1):
            bands.append((split_bounds[i], split_bounds[i+1]))

        splited_ds = split_dataset(dataset, res_axis, bands, drop=False)
        splited_la = split_dataset(latitudes, res_axis, bands, drop=True)

        # compute average on all splited datas
        #    the mean operation should lead to an array of shape ()
        #    therefore, we need to expand dims with parameter 'res_dim'
        #    could we automatically find the good dim name ??
        arrays = list()

        for i in range(len(splited_ds)):
            dims = dataset[res_axis].dims
            da = xr.DataArray(splited_la[i][varname].mean(dim=dims))

            if np.isnan(da):
                da = xr.DataArray(splited_la[i][res_axis].mean(dim=dims))
            if np.isnan(da):
                da.data = np.mean(bands[i])

            da = da.expand_dims({res_dim: 1})
            da.name = res_axis

            ds = splited_ds[i].mean(dim=dims)
            ds.coords[res_axis] = da

            arrays.append(ds)

        # concatenate arrays
        averaged = xr.concat(arrays, ds[res_axis].dims[0])  
    
    return averaged

