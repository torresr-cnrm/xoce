from .filters import AverageFilter, ShapiroFilter
from .integration import Integral
from .selectors import BoxClipper, FieldSelector, IndexSelector


# TODO: create a generic function for all processing ??
def Filter(name, dataset=None, *args, **kargs):
    if name == 'average':
        return AverageFilter(dataset, *args, **kargs)
    elif name == 'shapiro':
        return ShapiroFilter(dataset, *args, **kargs)

def Selector(name, dataset=None, *args, **kargs):
    if name == 'box-clip':
        return BoxClipper(dataset, *args, **kargs)
    elif name == 'field':
        return FieldSelector(dataset, *args, **kargs)
    elif name == 'index':
        return IndexSelector(dataset, *args, **kargs)
    
