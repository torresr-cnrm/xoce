"""
Mechanical energy computation in [J/kg] or [m2/s2]
"""

from xoce.calc.formulas.constants import CONST


class ke:
    long_name = 'kinetic energy per unit of volume'
    standard_name = 'kinetic_energy_vol'
    units = 'kg m-1 s-2'

    def calculate(uo, vo, wo=None, rho0=CONST.rho0):
        if wo is None:
            ke = 0.5*rho0*(uo**2. + vo**2.)
        else:
            ke = 0.5*rho0*(uo**2. + vo**2. + wo**2.)
        
        return CONST.rho0*ke


class pe:
    long_name = 'potential energy per unit of volume'
    standard_name = 'potential_energy_vol'
    units = 'kg m-1 s-2'

    def calculate(rho, N2, rho_star):
        return -0.5*CONST.g**2/(CONST.rho0*N2) * (rho_star)**2.

    