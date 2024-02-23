"""
Geophysical fluid dynamics computation.
"""

import xarray as xr

from xoce.calc.formulas.constants import CONST
from xoce.calc.formulas.thermo    import rho
from xoce.utils.dataset_util      import array_bnds, array_diff


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

    def calculate(rho):
        # init a 3D array with the first layer hydrostatic pressure
        poh_0 = CONST.g * rho * rho['depth'][0]
        poh   = xr.DataArray(name='poh', data=poh_0.data, coords=rho.coords, dims=rho.dims)
        
        dbnds = array_bnds(rho['depth'], dim='depth')
        
        # then compute the sea mass pressure above the cell
        dz_up      = xr.full_like(rho['depth'], 0.)
        rho_up     = rho.isel({'depth': range(0, len(rho['depth'])-1)})
        dz_up[:-1] = (dbnds[:, 1] - rho['depth'])[:-1].data
        poh_up     = CONST.g * (rho_up * dz_up).cumsum('depth')
        poh_up['depth'] = rho['depth'][1:]

        poh_up     = xr.concat( [0.*rho.isel({'depth':0}), poh_up], dim='depth')

        # finally add the missing cell sea water mass
        return poh + poh_up + CONST.g * (rho['depth'] - dbnds[:,0]) * rho


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

    def calculate(rho, e1t):
        # init a 3D array with the first layer hydrostatic pressure gradient
        poh_0 = CONST.g * rho.isel({'depth': 0}) * rho['depth'][0]
        hpg_0 = array_diff(poh_0, dim='x', method='centered') / (2*e1t)

        hpg   = hpg_0.expand_dims({'depth': rho['depth']}, axis=0)
        hpg.coords['depth'] = rho['depth']
                
        # ..then compute the pressure gradient above the cell center
        dbnds = array_bnds(rho['depth'], dim='depth')

        dz_dw      = xr.full_like(rho['depth'], 0.)
        dz_up      = xr.full_like(rho['depth'], 0.)

        dz_dw[:-1] = (dbnds[:, 1] - rho['depth'])[:-1].data
        dz_up[:-1] = (rho['depth'] - dbnds[:, 0])[:-1].data

        for k in range(1, len(rho['depth']), 1):
            poh    = CONST.g * rho.isel({'depth': k-1}) * dz_dw[k-1]
            poh    = poh + CONST.g * rho.isel({'depth': k}) * dz_up[k]
            hpg[k] = hpg[k-1] + array_diff(poh, dim='x', method='centered') / (2*e1t) 

        return hpg.transpose(*rho.dims)


class hpgj:
    long_name     = 'Hydrostatic pressure gradient in the j-direction'
    standard_name = 'hpgj'
    units         = 'Pa m-1'
    grid          = 'T'

    def calculate(rho, e2t):
        # init a 3D array with the first layer hydrostatic pressure gradient
        poh_0 = CONST.g * rho.isel({'depth': 0}) * rho['depth'][0]
        hpg_0 = array_diff(poh_0, dim='y', method='centered') / (2*e2t)

        hpg   = hpg_0.expand_dims({'depth': rho['depth']}, axis=0)
        hpg.coords['depth'] = rho['depth']
                
        # ..then compute the pressure gradient above the cell center
        dbnds = array_bnds(rho['depth'], dim='depth')

        dz_dw      = xr.full_like(rho['depth'], 0.)
        dz_up      = xr.full_like(rho['depth'], 0.)

        dz_dw[:-1] = (dbnds[:, 1] - rho['depth'])[:-1].data
        dz_up[:-1] = (rho['depth'] - dbnds[:, 0])[:-1].data

        for k in range(1, len(rho['depth']), 1):
            poh    = CONST.g * rho.isel({'depth': k-1}) * dz_dw[k-1]
            poh    = poh + CONST.g * rho.isel({'depth': k}) * dz_up[k]
            hpg[k] = hpg[k-1] + array_diff(poh, dim='y', method='centered') / (2*e2t) 

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


