"""
Functions to plot 2D graphs.
"""

import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_depth_zonal_mean(lats, depth, values, ax, cmap='viridis', vbounds=None, 
                          norm=None, **kargs):
    """ 
    Function to plot a 2D cartography using cartopy.
    """
    # arrange arguments
    if vbounds is None:
        vbounds = (None, None)
    
    # get key arguments
    smooth  = kargs.get('smooth', False)
    title = kargs.get('title', '')
    bcg = kargs.get('bgc', (0.75, 0.75, 0.75))
    yscale = kargs.get('yscale', 'log')
    axes_labels = kargs.get('axes_labels', ['',''])
    cbar_labels = kargs.get('cbar_labels', {})
    colorbar  = kargs.get('colorbar', None)
    add_cbar  = kargs.get('add_cbar', True)
    glines    = kargs.get('gridlines', False)

    # add general information about axis
    if title:
        ax.set_title(title)

    # axes for cartography plot
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    
    # overwrite properties if a colorbar is given
    if colorbar is not None:
        cmap = colorbar.cmap
        norm = colorbar.norm
        vbounds = (colorbar.vmin, colorbar.vmax)
        clevels = len(colorbar._values) - 1
    else:
        clevels = None

    ax.set_title(title)
    ax.set_facecolor(bcg)

    # X axis - latitudes
    ax.set_xlim((-80, 80))
    ax.set_xticks([-60, -30, 0, 30, 60])
    ax.set_xticklabels(['60S', '30S', '0', '30N', '60N'])
    ax.set_xlabel(axes_labels[0])

    # Y axis - depth
    ax.set_yscale(yscale)
    if yscale == 'log':
        ax.set_ylim((10, 6000))
        ax.set_yticks([20, 100, 300, 500, 1000, 3000])
        ax.set_yticklabels([20, 100, 300, 500, 1000, 3000])
    elif yscale == 'linear':
        ax.set_ylim((10, 5500))
        ax.set_yticks([10, 250, 500, 1000, 1500, 2000, 3000, 4000])
    
    ax.invert_yaxis()
    ax.set_ylabel(axes_labels[1])

    # plot data
    if smooth:
        im = ax.contourf(lats, depth, values, cmap=cmap, levels=clevels,
                         vmin=vbounds[0], vmax=vbounds[1])
    else:
        im = ax.pcolormesh(lats, depth, values, cmap=cmap, norm=norm,
                           vmin=vbounds[0], vmax=vbounds[1])

    # create axe for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cax.set_visible(add_cbar)

    if colorbar is not None:
        if colorbar.orientation == 'horizontal':
            cax.set_visible(False)
            
            # position a new axis for the horizontal colorbar
            labs = [a.get_label() for a in ax.figure.axes]
            ilab = 0
            while 'colorbar_id{}'.format(ilab) in labs:
                ilab += 1
            cax  = ax.figure.add_axes([0, 0, 0.1, 0.1], label='colorbar_id{}'.format(ilab))

            posn = ax.get_position()
            cax.set_position([posn.x0 + 0.1*posn.width, posn.y0 - 0.2*posn.height, 
                              0.8 * posn.width        , 0.05*posn.height          ])

            cax.set_visible(add_cbar)
    
        colorbar.ax  = cax
        colorbar.cax = cax

        locs, seq = list(), list()
        if 'locs' in colorbar.locator.__dict__:
            locs = colorbar.locator.locs
        if 'seq' in colorbar.formatter.__dict__:
            seq = colorbar.formatter.seq
        
        formatter = colorbar.formatter
        colorbar.update_normal(colorbar.mappable)

        if list(locs):
            colorbar.set_ticks(locs)
        if list(seq):
            colorbar.set_ticklabels(seq)

        colorbar.formatter = formatter

    elif add_cbar:
        if cbar_labels:
            cbar = plt.colorbar(im, cax=cax, fraction=1, pad=0, 
                                ticks=list(cbar_labels.values()))
            cbar.ax.set_yticklabels(list(cbar_labels.keys()))
        else:
            cbar = plt.colorbar(im, cax=cax, fraction=1, pad=0)

    plt.close()

    