"""
General fluid formulas
"""



class volume:
    long_name = 'Volume'
    standard_name = 'volcello'
    units = 'm3'

    def calculate(e1t, e2t, e3t):
        return e1t * e2t * e3t
    