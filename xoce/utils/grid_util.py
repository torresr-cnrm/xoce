"""
Function and module for grid management.
"""

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
        List of classes corresponding to formulas definition of ech variable
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

