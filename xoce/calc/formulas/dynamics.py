"""
Geophysical fluid dynamics computation.
"""

import numpy  as np
import xarray as xr

from xoce.api                     import _DIM_COORDINATES
from xoce.calc.formulas.constants import CONST
from xoce.utils.dataset_util      import array_bnds, array_diff, concatenate_arrays


class uo:
    long_name = 'Sea Water X Velocity'
    standard_name = 'sea_water_x_velocity'
    units = 'm s-1'
    grid  = 'U'

    def calculate():
        raise Exception('Undefined')
        

class vo:
    long_name = 'Sea Water Y Velocity'
    standard_name = 'sea_water_y_velocity'
    units = 'm s-1'
    grid  = 'V'

    def calculate():
        raise Exception('Undefined')


class poh:
    long_name     = 'Hydrostatic pressure under the geopotential surface'
    standard_name = 'sea_water_hydrostatic_pressure_under_geopotential_surface'
    units         = 'Pa'
    grid          = 'T'

    def calculate(rho, e3t):
        rdz_up  = (rho*e3t).isel({'depth': range(0, len(rho['depth'])-1)})
        rdz_up.coords['depth'] = rho['depth'][1:]

        poh_up  = xr.concat( [0.*rho.isel({'depth':0}), rdz_up.cumsum('depth')], 
                                                                            dim='depth')

        poh = CONST.g * (poh_up + rho*e3t/2)

        return poh.transpose(*rho.dims)


class po:
    long_name     = 'Absolute sea water pressure'
    standard_name = 'sea_water_absolute_pressure'
    units         = 'Pa'
    grid          = 'T'

    def calculate(pso, poh):
        return pso + poh


class hpgi:
    long_name     = 'Hydrostatic pressure gradient in the i-direction'
    standard_name = 'hpgi'
    units         = 'Pa m-1'
    grid          = 'T'

    def calculate(rho, e1t, e3t, zos):
        rdz_up  = (rho*e3t).isel({'depth': range(0, len(rho['depth'])-1)})
        grd_up  = array_diff(rdz_up, dim='x', method='centered') / (2*e1t)
        grd_up.coords['depth'] = rho['depth'][1:]

        hpg_up  = xr.concat( [0.*rho.isel({'depth':0}), grd_up.cumsum('depth')], 
                                                                            dim='depth')

        # compute d(rho * e3t/2)/dy
        depthb = e3t/2 - array_diff(zos, dim='x', method='forward')
        deptha = e3t/2 + array_diff(zos, dim='x', method='backward')

        dab = (rho * depthb).isel({'x': slice(0,-1,1)})
        daa = (rho * deptha).isel({'x': slice(1,None,1)})

        fst = (rho * rho['depth']).isel({'x': [0]})
        lst = (rho * rho['depth']).isel({'x': [-1]})

        co  = _DIM_COORDINATES.get('x', 'x')

        fst.data *= 0.
        lst.data *= 0.

        fst.coords[co] = (2*daa.coords[co][-1] - daa.coords[co][-2])
        lst.coords[co] = (2*dab.coords[co][0]  - dab.coords[co][1] )

        dab = dab.transpose(*rho.dims)
        daa = daa.transpose(*rho.dims)
        lst = lst.transpose(*rho.dims)
        fst = fst.transpose(*rho.dims)

        dab = concatenate_arrays([lst, dab], dim='x', chunks=rho.chunks)
        daa = concatenate_arrays([daa, fst], dim='x', chunks=rho.chunks)

        # compute the hydrostatic pressure gradient
        hpg = CONST.g * (hpg_up + (daa - dab) / (2*e1t))

        return hpg.transpose(*rho.dims)


class hpgj:
    long_name     = 'Hydrostatic pressure gradient in the j-direction'
    standard_name = 'hpgj'
    units         = 'Pa m-1'
    grid          = 'T'

    def calculate(rho, e2t, e3t, zos):
        rdz_up  = (rho*e3t).isel({'depth': range(0, len(rho['depth'])-1)})
        grd_up  = array_diff(rdz_up, dim='y', method='centered') / (2*e2t)
        grd_up.coords['depth'] = rho['depth'][1:]

        hpg_up  = xr.concat( [0.*rho.isel({'depth':0}), grd_up.cumsum('depth')], 
                                                                            dim='depth')

        # compute d(rho * e3t/2)/dy
        depthb = e3t/2 - array_diff(zos, dim='y', method='forward')
        deptha = e3t/2 + array_diff(zos, dim='y', method='backward')

        dab = (rho * depthb).isel({'y': slice(0,-1,1)})
        daa = (rho * deptha).isel({'y': slice(1,None,1)})

        fst = (rho * rho['depth']).isel({'y': [0]})
        lst = (rho * rho['depth']).isel({'y': [-1]})

        co  = _DIM_COORDINATES.get('y', 'y')

        fst.data *= 0.
        lst.data *= 0.

        fst.coords[co] = (2*daa.coords[co][-1] - daa.coords[co][-2])
        lst.coords[co] = (2*dab.coords[co][0]  - dab.coords[co][1] )

        dab = dab.transpose(*rho.dims)
        daa = daa.transpose(*rho.dims)
        lst = lst.transpose(*rho.dims)
        fst = fst.transpose(*rho.dims)

        dab = concatenate_arrays([lst, dab], dim='y', chunks=rho.chunks)
        daa = concatenate_arrays([daa, fst], dim='y', chunks=rho.chunks)

        # compute the hydrostatic pressure gradient
        hpg = CONST.g * (hpg_up + (daa - dab) / (2*e2t))

        return hpg.transpose(*rho.dims)


class hpg:
    long_name     = 'Hydrostatic pressure gradient'
    standard_name = 'hpg'
    units         = 'Pa m-1'
    grid          = 'T'

    def calculate(rho, e1t, e2t):
        grdx = hpgi.calculate(rho, e1t)
        grdy = hpgj.calculate(rho, e2t)

        return (grdx, grdy)


class spgi:
    long_name     = 'Surface pressure gradient in the i-direction'
    standard_name = 'spgi'
    units         = 'Pa m-1'
    grid          = 'W'

    def calculate(pso, e1t):

        return array_diff(pso, dim='x', method='centered') / (2*e1t)


class spgj:
    long_name     = 'Surface pressure gradient in the j-direction'
    standard_name = 'spgj'
    units         = 'Pa m-1'
    grid          = 'W'

    def calculate(pso, e2t):

        return array_diff(pso, dim='y', method='centered') / (2*e2t)


class spg:
    long_name     = 'Surface pressure gradient'
    standard_name = 'spg'
    units         = 'Pa m-1'
    grid          = 'W'

    def calculate(rho, e1t, e2t):
        grdx = spgi.calculate(rho, e1t)
        grdy = spgj.calculate(rho, e2t)

        return (grdx, grdy)


