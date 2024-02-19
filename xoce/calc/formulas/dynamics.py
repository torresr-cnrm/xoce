"""
Geophysical fluid dynamics computation.
"""

import xarray as xr

from xoce.calc.formulas.constants import CONST
from xoce.calc.formulas.thermo    import rho
from xoce.utils.dataset_util      import array_bnds


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


class psg:
    long_name     = 'Surface pressure at a reference geopotential surface'
    standard_name = 'sea_water_pressure_at_geopotential_surface'
    units         = 'Pa'
    grid          = 'W'

    def calculate(so, bigthetao, zos, pso=101325):
        sso = so.isel({'depth': 0})
        sct = bigthetao.isel({'depth': 0})

        return pso + CONST.g * rho.calculate(sso, sct, depth=0.) * zos


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

    def calculate(psg, poh):
        return psg + poh