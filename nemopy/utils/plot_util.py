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
                        make_grid=False):
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
        #   * indgap = indexes where longitude have discontinuity at latitude fixed
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
    
    return lons, lats, vals


def plot_carto(lons, lats, values, xmap, cmap_name='viridis', vbounds=None, norm=None, 
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

    # add general information about axis
    if title:
        xmap.set_title(title)

    # axes for cartography plot
    cmap = plt.cm.get_cmap(cmap_name)
    xmap.coastlines()
    xmap.gridlines()
    xmap.background_patch.set_facecolor(bcg)


    if isinstance(xmap.projection, (ccrs._RectangularProjection, ccrs.Mercator)):
        xmap.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        xmap.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    # plot data
    carto = xmap.pcolormesh(lons[:, :], lats[:, :], values[:, :], vmin=vbounds[0], 
                            vmax=vbounds[1], cmap=cmap, norm=norm, shading=shading, 
                            transform=ccrs.PlateCarree())

    # add an adaptative size color bar
    xcb = xmap.figure.add_axes([0, 0, 0.1, 0.1])
    xcb.axis('off')

    posn = xmap.get_position()
    xcb.set_position([posn.x0 + posn.width + 0.02, posn.y0, 0.04, posn.height])
    plt.colorbar(carto, ax=xcb, fraction=1, pad=0)

    plt.close()
