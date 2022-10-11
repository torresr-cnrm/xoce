"""
Geophysical fluid dynamics computation.
"""

class uo:
    long_name = 'Sea Water X Velocity'
    standard_name = 'sea_water_x_velocity'
    units = 'm s-1'
    grid  = 'U'

    def calculate():
        raise Exception('Undefined')
        

class vo:
    long_name = 'Sea Water Y Velocity'
    standard_name = 'sea_water_y_velocity'
    units = 'm s-1'
    grid  = 'V'

    def calculate():
        raise Exception('Undefined')

