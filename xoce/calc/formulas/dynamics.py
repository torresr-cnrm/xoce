"""
Geophysical fluid dynamics computation.
"""

import numpy  as np
import xarray as xr

from xoce.calc.formulas.constants import CONST
from xoce.utils.dataset_util      import array_diff


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


class pso:
    long_name     = 'Surface pressure at the reference geopotential surface'
    standard_name = 'sea_water_pressure_at_the_reference_geopotential_surface'
    units         = 'Pa'
    grid          = 'W'

    def calculate(zos, rau0=CONST.rho0, p_ref=0.):
        return p_ref + CONST.g * rau0 * zos


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

    def calculate(rho, e1t, e3t):
        rdz_up  = (rho*e3t).isel({'depth': range(0, len(rho['depth'])-1)})
        grd_up  = array_diff(rdz_up, dim='x', method='centered') / (2*e1t)
        grd_up.coords['depth'] = rho['depth'][1:]

        hpg_up  = xr.concat( [0.*rho.isel({'depth':0}), grd_up.cumsum('depth')], 
                                                                            dim='depth')

        grd = array_diff(rho*e3t/2, dim='x', method='centered') / (2*e1t)
        hpg = CONST.g * (hpg_up + grd)

        return hpg.transpose(*rho.dims)


class hpgj:
    long_name     = 'Hydrostatic pressure gradient in the j-direction'
    standard_name = 'hpgj'
    units         = 'Pa m-1'
    grid          = 'T'

    def calculate(rho, e2t, e3t):
        rdz_up  = (rho*e3t).isel({'depth': range(0, len(rho['depth'])-1)})
        grd_up  = array_diff(rdz_up, dim='y', method='centered') / (2*e2t)
        grd_up.coords['depth'] = rho['depth'][1:]

        hpg_up  = xr.concat( [0.*rho.isel({'depth':0}), grd_up.cumsum('depth')], 
                                                                            dim='depth')

        grd = array_diff(rho*e3t/2, dim='y', method='centered') / (2*e2t)
        hpg = CONST.g * (hpg_up + grd)

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

    def calculate(zos, e1t, rau0=CONST.rho0):
        return CONST.g * rau0 * array_diff(zos, dim='x', method='centered') / (2*e1t)


class spgj:
    long_name     = 'Surface pressure gradient in the j-direction'
    standard_name = 'spgj'
    units         = 'Pa m-1'
    grid          = 'W'

    def calculate(zos, e2t, rau0=CONST.rho0):
        return CONST.g * rau0 * array_diff(zos, dim='y', method='centered') / (2*e2t)


class spg:
    long_name     = 'Surface pressure gradient'
    standard_name = 'spg'
    units         = 'Pa m-1'
    grid          = 'W'

    def calculate(rho, e1t, e2t, rau0=CONST.rho0):
        grdx = spgi.calculate(rho, e1t, rau0)
        grdy = spgj.calculate(rho, e2t, rau0)

        return (grdx, grdy)

