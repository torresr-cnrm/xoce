"""
Mechanical energy computation in [J/kg] or [m2/s2]
"""

from .constants import g


def kinetic_energy(ds):
    if 'wo' not in ds:
        ke = 0.5*(ds['uo']**2. + ds['vo']**2.)
    else:
        ke = 0.5*(ds['uo']**2. + ds['vo']**2., ds['wo']**2.)

    ds['ke'] = ke


def potential_energy(ds):
    pe = -0.5*g/ds['N2'] * ds['rho']**2.
    
    ds['pe'] = pe
    