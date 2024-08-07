"""
"""

import matplotlib
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


_XOCE_COLORLIST = {
    'Groeskamp': [(0.361, 0.310, 0.624), (0.286, 0.529, 0.718), (0.502, 0.761, 0.655),
                  (0.702, 0.871, 0.659), (0.861, 0.905, 0.581), (0.984, 0.992, 0.788),
                  (0.969, 0.878, 0.576), (0.949, 0.698, 0.443), (0.902, 0.455, 0.306),
                  (0.761, 0.294, 0.322), (0.561, 0.106, 0.259),
                  ],

    'Casimir':   [(0.114, 0.122, 0.275), (0.133, 0.157, 0.345), (0.149, 0.180, 0.427),
                  (0.157, 0.208, 0.482), (0.165, 0.235, 0.557), (0.161, 0.259, 0.620),
                  (0.188, 0.278, 0.643), (0.200, 0.302, 0.667), (0.212, 0.318, 0.675),
                  (0.235, 0.341, 0.698), (0.259, 0.365, 0.706), (0.275, 0.357, 0.722),
                  (0.286, 0.455, 0.761), (0.290, 0.498, 0.796), (0.310, 0.541, 0.769),
                  (0.329, 0.561, 0.725), (0.357, 0.573, 0.694), (0.376, 0.596, 0.663),
                  (0.404, 0.616, 0.616), (0.435, 0.635, 0.576), (0.459, 0.647, 0.545),
                  (0.467, 0.647, 0.533), (0.494, 0.675, 0.494), (0.604, 0.718, 0.365),
                  (0.631, 0.729, 0.322), (0.675, 0.749, 0.294), (0.714, 0.765, 0.251),
                  (0.749, 0.780, 0.212), (0.792, 0.792, 0.180), (0.831, 0.804, 0.153),
                  (0.871, 0.824, 0.118), (0.902, 0.843, 0.094), (0.941, 0.859, 0.071),
                 ],

    'Farneti': [(0.024, 0.353, 0.569), (0.000, 0.431, 0.706), (0.000, 0.518, 0.788),
                (0.000, 0.655, 0.890), (0.043, 0.714, 0.918), (0.463, 0.800, 0.953),
                (0.604, 0.859, 0.957), (0.961, 0.761, 0.365), (0.949, 0.663, 0.161),
                (0.937, 0.494, 0.192), (0.922, 0.341, 0.184), (0.839, 0.247, 0.165),
                (0.745, 0.165, 0.180), (0.667, 0.184, 0.173)
                ],

    'OVL': [(0.852, 0.887, 0.969), (0.780, 0.571, 0.928), (0.846, 0.423, 0.969), 
            (0.494, 0.281, 0.714), (0.124, 0.103, 0.264), (0.056, 0.316, 0.530), 
            (0.066, 0.621, 0.785), (0.056, 0.862, 0.958), (0.040, 0.673, 0.617), 
            (0.056, 0.510, 0.276), (0.220, 0.525, 0.183), (0.536, 0.698, 0.178), 
            (0.826, 0.852, 0.178), (0.826, 0.423, 0.183), (0.820, 0.087, 0.183), 
            (0.525, 0.092, 0.194), (0.260, 0.082, 0.183)],

    'Xreds': [(0.941, 0.918, 0.914), (0.906, 0.827, 0.808), (0.875, 0.733, 0.690),
              (0.847, 0.647, 0.580), (0.816, 0.545, 0.455), (0.788, 0.451, 0.341),
              (0.753, 0.349, 0.231), (0.714, 0.243, 0.157), (0.651, 0.133, 0.141),
              (0.580, 0.071, 0.157), (0.498, 0.055, 0.161),
              ],

    'rgbw': [(0.450, 0.008, 0.000), (0.467, 0.051, 0.027), (0.541, 0.157, 0.102),
             (0.631, 0.251, 0.169), (0.714, 0.337, 0.235), (0.800, 0.435, 0.318),
             (0.835, 0.529, 0.361), (0.875, 0.631, 0.420), (0.902, 0.733, 0.494),
             (0.918, 0.80, 0.5490), (0.886, 0.851, 0.596), (0.749, 0.792, 0.494),
             (0.549, 0.702, 0.369), (0.439, 0.639, 0.298), (0.322, 0.580, 0.227),
             (0.231, 0.529, 0.176), (0.137, 0.478, 0.125), (0.086, 0.439, 0.118),
             (0.067, 0.412, 0.157), (0.067, 0.369, 0.212), (0.094, 0.310, 0.243),
             (0.114, 0.251, 0.357), (0.094, 0.282, 0.471), (0.067, 0.337, 0.576),
             (0.090, 0.396, 0.647), (0.125, 0.447, 0.710), (0.180, 0.525, 0.784),
             (0.290, 0.604, 0.839), (0.420, 0.686, 0.890), (0.584, 0.769, 0.925),
             (0.737, 0.855, 0.957), (0.831, 0.910, 0.980)
             ],

    'BlMRd': [(0.094, 0.282, 0.471), (0.420, 0.686, 0.890), (.98, .98, .90), 
              (0.714, 0.337, 0.235), (0.467, 0.051, 0.027),
             ],

    'BlRdAnomaly': [(0.027, 0.118, 0.275), (0.027, 0.184, 0.420), (0.031, 0.322, 0.612),
                    (0.129, 0.443, 0.710), (0.259, 0.573, 0.780), (0.353, 0.627, 0.804),
                    (0.471, 0.749, 0.839), (0.667, 0.863, 0.902), (0.859, 0.961, 1.000),
                    (0.941, 0.988, 1.000), (1.000, 0.988, 0.941), (1.000, 0.878, 0.878),
                    (0.988, 0.733, 0.667), (0.988, 0.573, 0.447), (0.988, 0.573, 0.447),
                    (0.984, 0.416, 0.290), (0.941, 0.235, 0.169), (0.800, 0.094, 0.118),
                    (0.651, 0.059, 0.078), (0.471, 0.039, 0.059), (0.373, 0.000, 0.000),
             ],

    'VoldoireBias': [(.647, .259, .969), (.000, .133, .671), (.000, .180, .859), 
                    (.000, .514, .922), (.000, .651, 1.00), (.000, .812, 1.00),
                     (.557, .863, 1.00), (.784, .969, 1.00), (.960, .990, 1.00),
                     (1.00, .995, .960), (1.00, .992, .804), (1.00, .882, .086),
                     (1.00, .702, .000), (1.00, .475, .000), (1.00, .000, .000),
                     (.904, .145, .141), (.714, .138, .134), (.923, .195, .442)
                    ],

    'blues':  list(plt.get_cmap('bwr')(np.arange(0, 128, 1)))[::-1],
    'reds' :  list(plt.get_cmap('bwr')(np.arange(127, 256, 1))), 

}


def _get_colormap(cmap, colors=None):

    colormap = None
    if isinstance(cmap, str):
        try :
            colormap = plt.get_cmap(cmap)
        except ValueError:
            if cmap.endswith('_r'):
                colors  = _XOCE_COLORLIST.get(cmap[:-2], list())[::-1]
            else:
                colors  = _XOCE_COLORLIST.get(cmap, list())

            if colors:
                colormap = clrs.LinearSegmentedColormap.from_list(cmap, colors)
    
    elif isinstance(colors, list):
        nc = len(colors)
        colormap = clrs.LinearSegmentedColormap.from_list(cmap, colors)

    elif isinstance(cmap, clrs.Colormap):
        return cmap
    
    if colormap is None:
        raise Exception("Colormap '{}' not found.".format(cmap))
    
    return colormap


def get_colorbar(cbnds, cmap='viridis', orientation='vertical', ncolors=None, norm=None,
                 ticklabels=None, return_fig=False, **kargs):
    
    if orientation == 'vertical':
        size  = "20%"
        fsize = (1.5, 4.)
    elif orientation == 'horizontal':
        size  = "75%"
        fsize = (20., 0.25)
    
    colormap = _get_colormap(cmap)
    
    # define boundaries
    if ncolors is not None:
        bnds = np.linspace(cbnds[0], cbnds[-1], ncolors+1)
    else:
        bnds = np.array(cbnds)
    
    if norm is None:
        norm = clrs.BoundaryNorm(bnds, colormap.N)

    # plot colorbar for preview
    ipfig, ipax1 = plt.subplots(ncols=1, figsize=fsize)

    divider = make_axes_locatable(ipax1)
    cax     = divider.append_axes("right", size=size, pad=0.0)
    cax.set_title("{}".format('Colorbar'))

    cbar = ipfig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), 
                          cax=cax, orientation=orientation, **kargs)

    if ticklabels:
        cbar.set_ticklabels(ticklabels)
    
    ipax1.set_visible(False)

    if return_fig:
        return ipfig
    else:
        return cbar

