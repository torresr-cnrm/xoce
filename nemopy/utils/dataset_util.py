"""
"""

import numpy as np
import xarray as xr


def merge_coordinates(dataset, coords, diff='relative', tol=1e-6):
    """
    Compare dataset.coords and coords (for instance from a reference mesh).
    The algorithm compare each dimension values and check if the difference
    are inside the tolerance. If so, the dataset coordinate is modified in 
    order to correspond to the reference. 
    Otherwise (difference greater than tolerance) the code '-1' is returned.
    """
    code = 0

    for co in dataset.coords:
        if co in coords:
            cdiff = dataset.coords[co].data - coords[co].data

            rcond = np.abs(coords[co].data) < 1.
            if diff == 'relative':
                cdiff[~rcond] /= coords[co].data[~rcond] 

            conds = np.abs(cdiff) < tol
            datas = dataset.coords[co].data
            datas[conds] = coords[co].data[conds]
            dataset.coords[co] = datas

            if len(datas[conds]) < len(datas):
                code = -1

    return code
