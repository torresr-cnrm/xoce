"""
"""

import gsw
import numpy as np
import xarray as xr

from nemopy.calc.formulas.constants import CONST


class rho:
    long_name = 'In-situ density'
    standard_name = 'density'
    units = 'kg m-3'
    unit_long = 'kilogrammes per cube meter'

    def calculate(so, bigthetao, depth):
        p = CONST.p0 + CONST.g*CONST.rho0*depth
        return gsw.density.rho(so*10**-3, bigthetao, p/10**4)


class bigthetao:
    long_name = 'Conservative temperature'
    standard_name = 'bigthetao'
    units = 'degrees_C'
    unit_long = 'degrees Celsius'

    def calculate(so, thetao):
        return gsw.conversions.CT_from_pt(so*10**-3, thetao)


class N2:
    """
    N^2 calculation from NEMO 3.6 (SUBROUTINE bn2 in eosbn2.F90) 
    """ 
    long_name = 'Brunt Vaisala frequency squared'
    standard_name = 'Nsquared'
    units = 's-1'
    unit_long = 'per second'

    def calculate(thetao, so, depth, e3t):
        T = thetao
        S = so
        d = depth

        n2 = xr.full_like(T, 1)
        rw = (e3t[1:])/(d[:-1].data-d[1:])
        rwd = rw.data

        A, B = N2.eos_ts_coefs(thetao, so, depth)
        while len(rwd.shape) < len(A.shape):
            rwd = rwd.reshape(list(rwd.shape)+[1])

        alpha = (1 - rw) * A[1:] + (rwd * A[:-1]).data
        beta  = (1 - rw) * B[1:] + (rwd * B[:-1]).data

        n2[1:] = CONST.g * (alpha * (T[:-1].data-T[1:]) - beta * (S[:-1].data-S[1:]))
        n2[1:] /= (e3t[1:])

        return n2


    def eos_ts_coefs(thetao, so, depth):
        """
        from NEMO v3.6
        """
        # default value: Vallis 2006
        a0      = 1.6550e-1    # thermal expansion coeff.
        b0      = 7.6554e-1    # saline  expansion coeff.
        lambda1 = 5.9520e-2    # cabbeling coeff. in T^2
        lambda2 = 5.4914e-4    # cabbeling coeff. in S^2
        mu1     = 1.4970e-4    # thermobaric coeff. in T
        mu2     = 1.1090e-5    # thermobaric coeff. in S
        nu      = 2.4341e-3    # cabbeling coeff. in theta*salt
        
        rho0  = 1026.
        teos  = thetao - 10.   # pot. temperature anomaly (t-T0)
        seos  = so - 35.       # abs. salinity anomaly (s-S0)

        zn    = a0 * ( 1. + lambda1*teos + mu1*depth ) + nu*seos
        za = zn * (1/rho0)
        
        zn    = b0 * ( 1. - lambda2*seos - mu2*depth ) - nu*teos
        zb  = zn * (1/rho0)
        
        return za, zb



# DEPRECIATED - old function for calculation of N2
def Nsquared(ds):
    p = CONST.p0 + CONST.g*CONST.rho0*ds['depth']
    lat = ds['latitude']

    # eventually reshape p-array and lat-array
    dims  = {d: ds.dims[d] for d in ds.dims if d in ds['so'].dims}
    
    ldims = [dims[d] for d in ds['so'].dims if d not in ds['depth'].dims]
    ldims += [1]
    p = np.tile(p.data, tuple(ldims))
    p = p.reshape(ds['so'].shape)

    ldims = [dims[d] for d in ds['so'].dims if d not in ds['latitude'].dims]
    ldims += [1]
    lat = np.tile(lat.data, tuple(ldims))
    lat = lat.reshape(ds['so'].shape)

    ds['Nsquared'] = gsw.stability.Nsquared(ds['so']*10**-3, ds['bigthetao'], p/10**4, lat)

