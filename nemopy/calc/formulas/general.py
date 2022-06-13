"""
General stratified fluid formulas
"""

from pandas import array
from nemopy.utils.dataset_util import array_diff


class volume:
    long_name = 'Volume'
    standard_name = 'volcello'
    units = 'm3'

    def calculate(e1t, e2t, e3t):
        return e1t * e2t * e3t


class surface:
    long_name = 'Surface'
    standard_name = 'areacello'
    units = 'm2'

    def calculate(e1t, e2t):
        return e1t * e2t


class e3t:
    long_name = 'Cell depth at T-grid'
    standard_name = 'e3t'
    units = 'm'

    def calculate(e3t_1d):
        return e3t_1d


class slp:
    long_name = 'Vector slope of neutral surfaces'
    standard_name = 'slope'
    units = ''

    def calculate(slpx, slpy):

        return (slpx, slpy)


class slpx:
    long_name = 'X component of the slope of neutral surfaces'
    standard_name = 'xslope'
    units = ''

    def calculate(rho, e1u, e3t):
        grdx = array_diff(rho, dim='x') / e1u
        grdz = array_diff(rho, dim='depth') / e3t

        return grdx/grdz


class slpy:
    long_name = 'Y component of the slope of neutral surfaces'
    standard_name = 'yslope'
    units = ''

    def calculate(rho, e2v, e3t):
        grdy = array_diff(rho, dim='y') / e2v
        grdz = array_diff(rho, dim='depth') / e3t

        return grdy/grdz

