
def compare_coords(coords, coords_ref):
    """
    Compare two xarray DatasetCoordinates instance.
    """
    tol_coordinates = dict()
    new_coordinates = dict()

    for c in coords:
        if c not in coords_ref:
            new_coordinates[c] = coords[c]

        else:
            tol_coordinates[c] = coords_ref[c] - coords[c]
    
    return tol_coordinates, new_coordinates
