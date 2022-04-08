"""
Mechanical energy computation in [J/kg] or [m2/s2]
"""

from nemopy.calc.formulas.constants import CONST


class ke:
    long_name = 'kinetic energy'
    standard_name = 'kinetic_energy'
    units = 'm2 s-2'

    def calculate(uo, vo, wo=None):
        if wo is None:
            ke = 0.5*(uo**2. + vo**2.)
        else:
            ke = 0.5*(uo**2. + vo**2. + wo**2.)
        
        return ke


class pe:
    long_name = 'potential energy'
    standard_name = 'potential_energy'
    units = ''

    def calculate(rho, N2):
        return -0.5*CONST.g/N2 * rho**2.

    