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
    long_name = 'In-situ density anomalies from a reference value'
    standard_name = 'density_an'
    units = 'kg m-3'
    unit_long = 'kilogrammes per cube meter'

    def calculate(rho, rho_ref=CONST.rho0):
        return rho - rho_ref


class prd:
    long_name = 'In-situ dimensionless density anomalies from a reference value'
    standard_name = 'prd'
    units = ''
    unit_long = ''

    def calculate(rho, rho_ref=CONST.rho0):
        return (rho / rho_ref) - 1


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
        n2.name = 'N2'

        rw = xr.full_like(T, 1)
        rw.name = 'rw'
        rwd = (0.5*e3t)/array_diff(depth, dim='depth', method='backward')

        # filtering unwanted dim of len 1
        while 1 in rwd.shape:
            dim = rwd.dims[rwd.shape.index(1)]
            rwd = rwd.isel({dim: 0})

        # fill the rw array
        for k in range(len(rwd)):
            slicer = [slice(0, None, 1)] * len(rw.dims)
            slicer[rw.dims.index('depth')] = slice(k, k+1, 1)
            rw[tuple(slicer)] = rwd[k]

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
        bn2 = -1*CONST.g * ( alpha * dT - beta * dS) / e3t
        
        # filter top / bottom boundaries
        # -- backward : need to filter the bottom
        slicers = [slice(0, None, 1)] * len(bn2.shape)
        idepth  = bn2.dims.index('depth')
        slicers[idepth]  = slice(-2, -1, 1)
        slc = tuple(slicers)
        
        bn2[slc] = xr.where(np.isnan(bn2[slc]), np.nan, 0.)

        # -- null value at surface
        slicers[idepth]  = slice(0, 1, 1)
        slc = tuple(slicers)

        bn2[slc] = xr.where(np.isnan(bn2[slc]), np.nan, 0.)

        return bn2


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


class M2:
    """
    Horizontal stratification M^2 
    """ 
    long_name = 'Horizontal stratification'
    standard_name = 'Msquared'
    units = ''
    unit_long = ''

    def calculate(prd, e1t, e2t):
        grdx = (array_diff(prd, dim='x'))
        grdy = (array_diff(prd, dim='y'))

        m2 = ((grdx/e1t)**2. + (grdy/e2t)**2.)**0.5
        m2.name = 'M2'

        return m2


class slpi:
    """
    Isopycnal slope (or neutral surfaces slope) in the i-direction
    from NEMO 3.6 (ldfslp.F90)
    """
    long_name = 'Isopycnal slope in the i-direction'
    standard_name = 'slpi'
    units = ''
    unit_long = ''

    def calculate(prd, N2, mlotst, depth, e1t, e3w_1d):
        gradx = (array_diff(prd, dim='x'))
        gradx.assign_coords(coords=prd.coords)

        zbw = -0.5/9.81 * N2 * ( 2 * prd + 2. )
        zai = gradx / e1t    

        zbi = np.minimum(zbw, -100. * np.abs(zai))
        zbi = np.minimum(zbi, -7e3 / e3w_1d * np.abs(zai))
            
        slpi = (zai / (zbi + 1e-20))
        slpi.name = 'wslpi'

        # special treatment for the mixed layer = slope just bellow the OML * coef
        nanmask = np.isnan(slpi)
        omlmask = (mlotst > depth)

        depth = N2['depth'].copy()
        depth = depth.assign_coords({'depth': np.arange(depth.shape[0])})
        conds = (depth['depth'] == omlmask.sum('depth'))
        conds = conds.assign_coords({'depth': N2['depth'].values})

        omln2  = N2.where(conds).sum('depth')
        omlprd = prd.where(conds).sum('depth')

        zbw = -0.5/9.81 * omln2 * ( 2 * omlprd + 2. )

        zai = (gradx / e1t).where(conds).sum('depth') 

        zbi = np.minimum(zbw, -100. * np.abs(zai))
        zbi = np.minimum(zbi, -7e3 / e3w_1d * np.abs(zai))

        ze3w = e3w_1d
        for dim in slpj.dims:
            if dim not in ze3w.dims :
                ze3w = ze3w.expand_dims({dim: slpj[dim].shape[0]})
        ze3w = ze3w.transpose(*slpj.dims)
        ze3w = ze3w.where(conds).sum('depth')

        omlcoef = (depth - 0.5 * ze3w) / xr.where(mlotst < 5., mlotst, 5.)

        omlslp = (zai / (zbi + 1e-20)) * omlcoef
        omlslp = omlslp.assign_coords({'depth': slpi.coords['depth']})

        slpi = xr.where(omlmask, omlslp, slpi)
        slpi = xr.where(nanmask, np.nan, slpi)
        
        return slpi


class slpj:
    """
    Isopycnal slope (or neutral surfaces slope) in the j-direction
    from NEMO 3.6 (ldfslp.F90)
    """
    long_name = 'Isopycnal slope in the j-direction'
    standard_name = 'slpj'
    units = ''
    unit_long = ''

    def calculate(prd, N2, mlotst, depth, e2t, e3w_1d):
        grady = (array_diff(prd, dim='y'))
        grady.assign_coords(coords=prd.coords)

        zbw = -0.5/9.81 * N2 * ( 2 * prd + 2. )
        zaj = grady / e2t    

        zbj = np.minimum(zbw, -100. * np.abs(zaj))
        zbj = np.minimum(zbj, -7e3 / e3w_1d * np.abs(zaj))
            
        slpj = (zaj / (zbj + 1e-20))
        slpj.name = 'wslpj'

        # special treatment for the mixed layer = slope just bellow the OML * coef
        nanmask = np.isnan(slpj)
        omlmask = (mlotst > depth)

        depth = N2['depth'].copy()
        depth = depth.assign_coords({'depth': np.arange(depth.shape[0])})
        conds = (depth['depth'] == omlmask.sum('depth'))
        conds = conds.assign_coords({'depth': N2['depth'].values})

        omln2  = N2.where(conds).sum('depth')
        omlprd = prd.where(conds).sum('depth')

        zbw = -0.5/9.81 * omln2 * ( 2 * omlprd + 2. )

        zaj = (grady / e2t).where(conds).sum('depth') 

        zbj = np.minimum(zbw, -100. * np.abs(zaj))
        zbj = np.minimum(zbj, -7e3 / e3w_1d * np.abs(zaj))
        
        ze3w = e3w_1d
        for dim in slpj.dims:
            if dim not in ze3w.dims :
                ze3w = ze3w.expand_dims({dim: slpj[dim].shape[0]})
        ze3w = ze3w.transpose(*slpj.dims)
        ze3w = ze3w.where(conds).sum('depth')

        omlcoef = (depth - 0.5 * ze3w) / xr.where(mlotst < 5., mlotst, 5.)

        omlslp = (zaj / (zbj + 1e-20)) * omlcoef
        omlslp = omlslp.assign_coords({'depth': slpj.coords['depth']})

        slpj = xr.where(omlmask, omlslp, slpj)
        slpj = xr.where(nanmask, np.nan, slpj)

        return slpj


class slp:
    """
    Isopycnal slope (or neutral surfaces slope) vector
    """
    long_name = 'Isopycnal slope vector'
    standard_name = 'slp'
    units = ''
    unit_long = ''

    def calculate(prd, N2, mlotst, depth, e1t, e2t, e3w_1d):
        sx = slpi.calculate(prd, N2, mlotst, depth, e1t, e3w_1d)
        sy = slpj.calculate(prd, N2, mlotst, depth, e2t, e3w_1d)

        return (sx, sy)

        