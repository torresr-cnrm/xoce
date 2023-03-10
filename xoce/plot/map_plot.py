"""
"""

import cartopy.feature as cfeature

_LAND_COLROS = {
    'brwn': (0.73, 0.7, 0.65), 
    'dark_brwn': (0.53, 0.48, 0.44),
}

def get_land_mask(rscale, colors):

    if colors in cfeature.COLORS:
        colors = cfeature.COLORS[colors]
    
    elif colors in _LAND_COLROS:
        colors = _LAND_COLROS[colors]
    
    land_mask = cfeature.NaturalEarthFeature(
        'physical', 
        'land', 
        scale=rscale,
        edgecolor='k', 
        facecolor=colors,
    )

    return land_mask

