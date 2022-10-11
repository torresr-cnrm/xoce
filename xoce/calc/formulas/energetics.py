"""
Mechanical energy computation in [J/kg] or [m2/s2]
"""

from xoce.calc.formulas.constants import CONST


class ke:
    long_name = 'kinetic energy per unit of volume'
    standard_name = 'kinetic_energy_vol'
    units = 'kg m-1 s-2'
    grid  = 'T'

    def calculate(uo, vo, wo=None, rho0=CONST.rho0):
        return 0.5*rho0*(uo**2. + vo**2.)
        

class pe:
    long_name = 'potential energy per unit of volume'
    standard_name = 'potential_energy_vol'
    units = 'kg m-1 s-2'
    grid  = 'T'

    def calculate(rho, N2, rho_star):
        return -0.5*CONST.g**2/(CONST.rho0*N2) * (rho_star)**2.

    