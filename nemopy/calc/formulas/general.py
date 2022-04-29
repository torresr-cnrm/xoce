"""
General fluid formulas
"""



class volume:
    long_name = 'Volume'
    standard_name = 'volcello'
    units = 'm3'

    def calculate(e1t, e2t, e3t):
        return e1t * e2t * e3t


class e3t:
    long_name = 'Cell depth at T-grid'
    standard_name = 'e3t'
    units = 'm'

    def calculate(e3t_1d):
        return e3t_1d
    