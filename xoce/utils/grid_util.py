"""
Function and module for grid management.
"""

import numpy             as np
from   scipy.interpolate import griddata
import warnings


# Grid conventions
_GRIDS_CONVERSION = {
    "Arakawa-C":{
        'U': {
            'longitude': ['F'],
            'latitude': ['T', 'W'],
            'depth':    ['V', 'T', 'F']
        },
        'V': {
            'longitude': ['T', 'W'],
            'latitude': ['F'],
            'depth':    ['U', 'T', 'F']
        },
        'T': {
            'longitude': ['V', 'W'],
            'latitude': ['U', 'W'],
            'depth':    ['U', 'V', 'F']
        }
    }
}


def extract_coords(larrays, classes, grid, convention='Arakawa-C', skiped=None):
    """
    Extract grid coodinates from a list of arrays.  

    Parameters
    ----------
    larrays : Hashable, sequence of Hashable, list
        List of DataArray items where the final coordinates are choosen.
    classes : Hashable, sequence of Hashable, list
        List of classes corresponding to formulas definition of each variable
        contained in larrays.
    grid : str
        Name of the grid point. Should be 'U', 'V' or 'T'.
    convention : str, default: 'Arakawa-C'
        Key name for the convention dictionnary to use.
    skiped : Hashable, seaquence of Hashable, list, dict or DataArray, default : None
        List of coordinates to skip. This is used to avoid computation of already
        coordinate.

    Returns
    -------
    expanded : dict
        Dictionnary with coordinate name as keys and converted grid DataArray as 
        values.

    """

    try:
        conversion_dict = _GRIDS_CONVERSION[convention][grid]
    except KeyError:
        return {}

    if skiped is None:
        skiped = list()

    coords = dict()
    
    for co in conversion_dict:
        if co not in skiped:
            for i, cls in enumerate(classes):
                if 'grid' in dir(cls) and cls.grid in conversion_dict[co]:
                    cdims  = larrays[i][co].dims
                    cdata = larrays[i][co].data

                    coords[co] = (cdims, cdata)
    return coords



def reggrid(xref, yref, values, interpolation='linear', xlim='auto', ylim='auto', nbx=1001, nby=1001):
    """
    Interpolate two dimensional array onto a new regular grid. The regular grid 
    is defined by the min/max of the non-regular grid (xref, yref). 

    Parameters
    ----------
    xref : 1-D [or 2-D] array of floats with shape (n,) [or shape (n,m)]
        Data point coordinate in the 1st direction of the reference grid.
    yref : 1-D [or 2-D] array of floats with shape (m,) [or shape (n,m)]
        Data point coordinate in the 2nd direction of the reference grid.
    values : 2-D array of floats with shape (n,m)
        Data values in the reference grid.
    interpolation : {‘linear’, ‘nearest’, ‘cubic’}, optional
        Interpolation method (see scipy.interpolate.griddata). Default is 'linear'.
    xlim : tuple of int with length 2 or 'auto', optional
        Define the lower and upper bound of the new grid in the 1st direction. 
        Default is 'auto' which calculates bounds as min/max of 'xref'.
    ylim : tuple of int with length 2 or 'auto', optional
        Define the lower and upper bound of the new grid in the 2nd direction.
        Default is 'auto' which calculates bounds as min/max of 'yref'.
    nbx : int, optional
        Number of point in the 1st direction of the interpolation grid. Default is 1001.
    nby : int, optional
        Number of point in the 2nd direction of the interpolation grid. Default is 1001.
    
    Returns
    -------
    xi : 2-D array of floats with shape (n,m)
        Data mesh coordinate in the 1st direction of the interpolate grid.
    yi: 2-D array of floats with shape (m,)
        Data mesh coordinate in the 2nd direction of the interpolate grid.
    interp : 2-D array of floats with shape (n,m)
        Data values interpolated into the new grid.
    """

    # Check dimensions of input arrays
    if len(xref.shape) != len(yref.shape) and (not len(xref.shape) in [1, 2]):
        warnings.warn("Warning (reggrid) - 'xref' and 'yref' should be 1-D or 2-D arrays, no interpolation was done.")
        return (xref, yref), values

    # Compute grid boundaries 
    if xlim == 'auto':
        xmin, xmax = np.min(xref), np.max(xref)
    else:
        xmin, xmax = xlim
    
    if ylim == 'auto':
        ymin, ymax = np.min(yref), np.max(yref)
    else:
        ymin, ymax = ylim
    
    # Get reference grid
    if len(xref.shape) == 1:
        xr, yr = np.meshgrid(xref, yref)
    else:
        xr, yr = xref, yref

    # Create interpolation grid
    x          = np.linspace(xmin, xmax, nbx)
    y          = np.linspace(ymin, ymax, nby)
    xi, yi     = np.meshgrid(x, y)

    # interpolate data values from reference grid (xr, yr) to interpolation grid (xi, yi)
    points = np.stack([xr.flatten(), yr.flatten()], -1)
    values = values.flatten()
    interp = griddata(points=points, values=values, xi=(xi.flatten(), yi.flatten()), method=interpolation)
    interp = interp.reshape(xi.shape)

    return (xi, yi), interp

