from .filters import AverageFilter, ShapiroFilter, CoarseningFilter
from .integration import Integral
from .selectors import BoxClipper, FieldSelector


def Filter(name, dataset=None, *args, **kargs):
    if name == 'average':
        return AverageFilter(dataset, *args, **kargs)
    elif name == 'shapiro':
        return ShapiroFilter(dataset, *args, **kargs)
    elif name == 'coarsening':
        return CoarseningFilter(dataset, *args, **kargs)

def Selector(name, dataset=None, *args, **kargs):
    if name == 'box-clip':
        return BoxClipper(dataset, *args, **kargs)
    elif name == 'field':
        return FieldSelector(dataset, *args, **kargs)
    
