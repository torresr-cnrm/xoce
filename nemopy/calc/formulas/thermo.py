"""
"""

import gsw
import numpy as np
import xarray as xr

from nemopy.calc.formulas.constants import CONST
from nemopy.utils.dataset_util import array_diff


class rho:
    long_name = 'In-situ density'
    standard_name = 'density'
    units = 'kg m-3'
    unit_long = 'kilogrammes per cube meter'

    def calculate(so, bigthetao, depth):
        p = CONST.p0 + CONST.g*CONST.rho0*depth
        return gsw.density.rho(so*10**-3, bigthetao, p/10**4)


class rho_star:
    long_name = 'Density anomalies from a reference value'
    standard_name = 'density_an'
    units = 'kg m-3'
    unit_long = 'kilogrammes per cube meter'

    def calculate(rho, rho_ref=CONST.rho0):
        return rho - rho_ref


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

        n2 = xr.full_like(T, 1)
        n2 = xr.where(np.isnan(T), np.nan, n2)
        rw = (0.5*e3t)/array_diff(depth, dim='depth', method='backward')

        A, B = N2.__eos_ts_coefs(thetao, so, depth)

        # reshape rw to match EOS coefs dimensions (create a function for this ?)
        index = list()
        for co in A.dims:
            if co not in rw.dims:
                rw = rw.expand_dims({co: A[co].size})
            if co == 'depth':
                index.append(slice(1,None,1))
            else:
                index.append(slice(0,None,1))
        
        # select thermal/haline coefficient and change 'depth' coordinates
        A1 = A.isel({'depth': slice(0,-1,1)})
        A1.coords['depth'] = A.coords['depth'][1:]
        A1 = xr.concat([A.isel({'depth': [0]}), A1], dim='depth')

        B1 = B.isel({'depth': slice(0,-1,1)})
        B1.coords['depth'] = B.coords['depth'][1:]
        B1 = xr.concat([B.isel({'depth': [0]}), B1], dim='depth')

        alpha = (1 - rw) * A1 + (rw * A)
        beta  = (1 - rw) * B1 + (rw * B)

        dT = array_diff(T, dim='depth', method='backward')
        dS = array_diff(S, dim='depth', method='backward')

        # calculate n2 for all depth except surface (index 0)
        bn2 = CONST.g * ( alpha * dT - beta * dS) / e3t
        
        # filter top / bottom boundaries
        # -- backward : need to filter the bottom
        cbnds = (n2['depth'] < n2['depth'][-1])
        n2 = xr.where(cbnds, bn2, 0)

        return n2


    def __eos_ts_coefs(thetao, so, depth):
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
        
        teos  = thetao - 10.   # pot. temperature anomaly (t-T0)
        seos  = so - 35.       # abs. salinity anomaly (s-S0)

        zn    = a0 * ( 1. + lambda1*teos + mu1*depth ) + nu*seos
        za = zn * (1/CONST.rho0)
        
        zn    = b0 * ( 1. - lambda2*seos - mu2*depth ) - nu*teos
        zb  = zn * (1/CONST.rho0)
        
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

