"""
Define functions and methods for plotting datas. 
"""

import numpy as np
import numpy.ma as ma
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def switch_array(datas, isep, axis=0):
    """
    Sort an array to exchange order of values from a specified index.
    Switch is only done on one axis (default is 0).

    In practical, the function is used on geophysical netCDF datasets in order
    to arange correctly longitudes axis.

    # Examples
    ----------

    Let's take an arbitrary 3D array:

    >>> import numpy as np
    >>> a = np.arange(2*3*5).reshape(2, 5, 3)

    We use the function `switch_array` used on axis 1 and with 2 as index from which
    to revert the array:

    >>> b = switch_array(a, 2, axis=1)[:, 1, :]

    This as the same final result than followed lines:

    >>> c = np.ones_like(a)
    >>> c[:, :3, :] = a[:, 2:, :]
    >>> c[:, 3:, :] = a[:, :2, :]
    >>> np.allclose(b[:, 1, :], a[:, 1, :])
    False
    >>> np.allclose(b[:, 1, :], c[:, 1, :])
    True

    """
    if isinstance(datas, ma.MaskedArray):
        newdt = ma.zeros(datas.shape)
        newdt.mask = datas.mask
    else:
        newdt = np.zeros_like(datas) + np.nan
    laxis = datas.shape[axis]

    # prepare slicings: slci (slice for inputs datas), slco (slice for outpurs newdt)
    slci = [slice(None)] * len(datas.shape)
    slco = [slice(None)] * len(datas.shape)

    slci[axis] = slice(isep, laxis)
    slco[axis] = slice(0, laxis-isep)

    newdt[tuple(slco)] = datas[tuple(slci)]

    slci[axis] = slice(0, isep)
    slco[axis] = slice(laxis-isep, laxis)

    newdt[tuple(slco)] = datas[tuple(slci)]

    return newdt


def get_variable_data(datas, var, mesh=None, filtering=True, lamtol=100, 
                      make_grid=False, duplicate_at_origin=False):
    """
    Get variable datas and re arange fields if needed
    (ex: if discontinuity is found in longitudinal data.)

    --
    Warning: algorithm for data filtering is not fully generic yet
    and should improve the way to detect axis on which operate filters.
    Slice object should help for that..
    """

    # get latitude and longitude variables
    if mesh:
        lons = np.array(mesh['glamu'][0])
        lats = np.array(mesh['gphiv'][0])
    else:
        lons = np.array(datas['longitude'])
        lats = np.array(datas['latitude'])

    # get exported variable
    if isinstance(datas, xr.DataArray):
        vals = np.array(datas.data)
    else:
        vals = np.array(datas[var])
    vals.setflags(write=1)

    if filtering:
        # 1. get the last longitude
        lamshp = lons.shape

        i = -1
        isearch = None

        while lamshp[1]+i > 0 and isearch is None:
            if (lons[:, 0] == lons[:, i]).sum() == lamshp[0]:
                isearch = i
            else:
                i -= 1

        lons = lons[:, :isearch]
        lats = lats[:, :isearch]

        if len(vals.shape) > 3 and 1 in vals.shape:
            dat = vals.data
            shp = list(dat.shape)
            dim = list(vals.dims)

            if 1 in shp:
                ax2del = shp.index(1)
                del(shp[ax2del])
                del(dim[ax2del])

                dat = dat.reshape(tuple(shp))
                vals = xr.Variable(tuple(dim), dat, vals.attrs)

        if len(vals.shape) == 2:
            vals = vals[:, :isearch]
        elif len(vals.shape) == 3:
            vals = vals[:, :, :isearch]
        else:
            raise Exception()

        # 2. switch array to ensure longitude continuity
        #   * indgap = indexes where longitude have discontinuity at fixed latitude
        indgap = np.where(np.abs(lons[:, :-1] - lons[:, 1:]) > lamtol)

        # re arange array to sort longitude coordinates (only if one gap is found)
        for i, c in zip(*np.unique(indgap[0], return_counts=True)):
            if c == 1:
                ind = indgap[1][i]
                lats[i] = switch_array(lats[i], ind+1, axis=-1)
                lons[i] = switch_array(lons[i], ind+1, axis=-1)
                if len(vals.shape) == 2:
                    vals[i] = switch_array(vals[i], ind+1, axis=-1)
                elif len(vals.shape) == 3:
                    vals[:, i] = switch_array(vals[:, i], ind+1, axis=-1)

        # 3. final check if there is no other discontinuity
        indgap = np.where(np.abs(lons[:-1, :] - lons[1:, :]) > lamtol)
        inddel = np.unique(indgap[0])
        inddel[::-1].sort()

        for i in inddel:
            lons = np.delete(lons, i+1, axis=0)
            lats = np.delete(lats, i+1, axis=0)
            if len(vals.shape) == 2:
                vals = np.delete(vals, i+1, axis=0)
            elif len(vals.shape) == 3:
                vals = np.delete(vals, i+1, axis=1)


    if lons.shape != lats.shape:
        raise Exception("longitudes and latitudes shapes error.")

    if make_grid and lons.shape == vals.shape[1:]:
        lons = np.insert(lons, 0, lons[:, -1]-360., axis=1)
        lats = np.insert(lats, 0, lats[:, -1], axis=1)

        lons = np.insert(lons, 0, lons[0, :], axis=0)
        lats = np.insert(lats, 0, 2*lats[0, :]-lats[1, :], axis=0)
    
    elif duplicate_at_origin:
        lons = np.concatenate( (np.ones(293).reshape((293, 1))*-180., lons), axis=1)
        lons = np.concatenate( (lons, np.ones(293).reshape((293, 1))*180.), axis=1)
        lats = np.concatenate( (lats[:,   0].reshape((293, 1)), lats), axis=1)
        lats = np.concatenate( (lats, lats[:,  -1].reshape((293, 1))), axis=1)
        vals = np.concatenate( (vals[:,   0].reshape((293, 1)), vals), axis=1)
        vals = np.concatenate( (vals, vals[:,  -1].reshape((293, 1))), axis=1)
    
    return lons, lats, vals


def fill_orca_grid(lons, lats, values, xmap, qmesh, phimax=89.9):
    """
    Fill non-value datas due to the tri-polar ORCA grid.
    This function build a second mesh to fill and add it to the 
    current one.
    """
    wlons, elons = list(), list()
    wlats, elats = list(), list()
    wvals, evals = list(), list()

    nb_lats, nb_lons = lats.shape
    
    # East ORCA pole
    west = lons[-1] < 0
    iepol = np.nanargmin(np.where(~west, lats[-1], np.nan))

    i = 0
    icnds = (lons[-1][iepol-i] >= 0.) and (iepol+1+i < nb_lons)
    while (lats[-1][iepol-i] < phimax) and icnds:
        elons.append([lons[-1][iepol-i], lons[-1][iepol+1+i]])
        elats.append([lats[-1][iepol-i], lats[-1][iepol+1+i]])
        evals.append([values[-1][iepol-i]])

        i+= 1
        icnds = (lons[-1][iepol-i] >= 0.) and (iepol+1+i < nb_lons)

    # West ORCA pole
    iwpol = np.nanargmin(np.where(west, lats[-1], np.nan))
    
    i = 0
    icnds = (lons[-1][iwpol+1+i] <= 0.) and (iwpol-i >= 0)
    while (lats[-1][iwpol-i] < phimax) and icnds:
        wlons.append([lons[-1][iwpol-i], lons[-1][iwpol+1+i]])
        wlats.append([lats[-1][iwpol-i], lats[-1][iwpol+1+i]])
        wvals.append([values[-1][iwpol-i]])
        
        i+= 1
        icnds = (lons[-1][iwpol+1+i] <= 0.) and (iwpol-i >= 0)

    wlons.append([0, 180.])
    wlats.append([phimax, 90.])
    
    # now add the new values in the axes 
    vmin, vmax = qmesh.get_clim()

    if elons:
        xmap.pcolormesh(np.array(elons), np.array(elats), np.array(evals), 
                        cmap=qmesh.cmap, norm=qmesh.norm, 
                        #vmin=vmin, vmax=vmax, cmap=qmesh.cmap, norm=qmesh.norm, 
                        shading='auto', transform=ccrs.PlateCarree())
    if wlons:
        xmap.pcolormesh(np.array(wlons), np.array(wlats), np.array(wvals),
                        cmap=qmesh.cmap, norm=qmesh.norm,
                        #vmin=vmin, vmax=vmax, cmap=qmesh.cmap, norm=qmesh.norm,
                        shading='auto', transform=ccrs.PlateCarree())

    # avoid longitude discontinuity
    if isinstance(xmap.projection, (ccrs.PlateCarree)):
        nlons = np.ones((nb_lats, 2)) * -180.5
        nlons[:, 1] = lons[:, 0]
        nlats = np.ones((nb_lats, 2)) * lats[:, 0].reshape((nb_lats, 1))
        nvals = values[:, 0].reshape((nb_lats, 1))

        xmap.pcolormesh(nlons, nlats, nvals, 
                        cmap=qmesh.cmap, norm=qmesh.norm, 
                        #vmin=vmin, vmax=vmax, cmap=qmesh.cmap, norm=qmesh.norm, 
                        shading='auto', transform=ccrs.PlateCarree())

        nlons = np.ones((nb_lats, 2)) * 180.5
        nlons[:, 0] = lons[:, -1]
        nlats = np.ones((nb_lats, 2)) * lats[:, -1].reshape((nb_lats, 1))
        nvals = values[:, -1].reshape((nb_lats, 1))
        
        xmap.pcolormesh(nlons, nlats, nvals, 
                        cmap=qmesh.cmap, norm=qmesh.norm, 
                        #vmin=vmin, vmax=vmax, cmap=qmesh.cmap, norm=qmesh.norm, 
                        shading='auto', transform=ccrs.PlateCarree())

    return None


def plot_carto(lons, lats, values, xmap, cmap='viridis', vbounds=None, norm=None, 
                **kargs):
    """ 
    Function to plot a 2D cartography using cartopy.
    """
    # arrange arguments
    if vbounds is None:
        vbounds = (None, None)
    
    # get key arguments
    shading = kargs.get('shading', 'auto')
    title = kargs.get('title', '')
    bcg = kargs.get('bgc', (0.75, 0.75, 0.75))
    axes_labels = kargs.get('axes_labels', ['',''])
    cbar_labels = kargs.get('cbar_labels', {})
    colorbar  = kargs.get('colorbar', None)
    add_cbar  = kargs.get('add_cbar', True)
    orca_grid = kargs.get('orca_grid', True)
    glines = kargs.get('gridlines', True)

    # add general information about axis
    if title:
        xmap.set_title(title)

    # axes for cartography plot
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    
    # overwrite properties if a colorbar is given
    if colorbar is not None:
        cmap = colorbar.cmap
        norm = colorbar.norm
        vbounds = (colorbar.vmin, colorbar.vmax)

    xmap.coastlines()
    #xmap.background_patch.set_facecolor(bcg)
	
    if glines:
        xmap.gridlines()

    coef_y = 0.1

    if isinstance(xmap.projection, (ccrs._RectangularProjection, ccrs.Mercator)):
        xmap.set_xticks([-150,-100, -50, 0, 50, 100, 150], crs=ccrs.PlateCarree())
        xmap.set_xticklabels(['150W', '100W', '50W', '0', '50E', '100E', '150E'])

        xmap.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], crs=ccrs.PlateCarree())

        xmap.set_xlabel(axes_labels[0])
        xmap.set_ylabel(axes_labels[1])

        coef_y = 0.15

    # plot data
    if norm is not None:
        qmesh = xmap.pcolormesh(lons[:, :], lats[:, :], values[:, :], 
                                cmap=cmap, norm=norm, shading=shading, 
                                transform=ccrs.PlateCarree())
    else:
        qmesh = xmap.pcolormesh(lons[:, :], lats[:, :], values[:, :], vmin=vbounds[0], 
                                vmax=vbounds[1], cmap=cmap, shading=shading, 
                                transform=ccrs.PlateCarree())

    if orca_grid:
        fill_orca_grid(lons[:, :], lats[:, :], values[:, :], xmap, qmesh)

    if not add_cbar:
        plt.close()
        return
        
    # add an adaptative size color bar
    labs = [ax.get_label() for ax in xmap.figure.axes]
    ilab = 0
    while 'colorbar_id{}'.format(ilab) in labs:
        ilab += 1
    xcb = xmap.figure.add_axes([0, 0, 0.1, 0.1], label='colorbar_id{}'.format(ilab))

    posn = xmap.get_position()
    xcb.set_position([posn.x0 + posn.width + 0.02, posn.y0, 0.01, posn.height])

    if colorbar is not None:
        if colorbar.orientation == 'horizontal':
            cb_position = [posn.x0 + 0.1 * posn.width, 
                           posn.y0 - coef_y * posn.height, 
                           posn.width * 0.8,
                           0.03]
            
            if colorbar.__dict__.get('position', '') == 'top':
                cb_position[1] = posn.y0 + posn.height + 0.03

            xcb.set_position(cb_position)
            xcb.set_xlabel(colorbar.ax.get_xlabel())
        
        else:
            xcb.set_ylabel(colorbar.ax.get_ylabel())

        locs, seq = list(), list()
        if 'locs' in colorbar.locator.__dict__:
            locs = colorbar.locator.locs
        if 'seq' in colorbar.formatter.__dict__:
            seq = colorbar.formatter.seq

        colorbar.ax = xcb
        colorbar.cax = xcb

        formatter = colorbar.formatter

        #colorbar.update_normal(colorbar.mappable)
        cbar = plt.colorbar(colorbar.mappable, cax=xcb, fraction=1, pad=0, 
                            extend=colorbar.extend)
        colorbar.locator.locs
        if list(locs):
            colorbar.set_ticks(locs)
        if list(seq):
            colorbar.set_ticklabels(seq)

        colorbar.formatter = formatter

        if colorbar.__dict__.get('position', '') == 'top':
            xcb.xaxis.tick_top()

    else:
        if cbar_labels:
            cbar = plt.colorbar(qmesh, cax=xcb, fraction=1, pad=0, 
                                ticks=list(cbar_labels.values()))
            cbar.ax.set_yticklabels(list(cbar_labels.keys()))
        else:
            cbar = plt.colorbar(qmesh, cax=xcb, fraction=1, pad=0)

    plt.close()

