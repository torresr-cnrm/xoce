from .filters import AverageFilter
from .selectors import BoxClipper


# TODO: create a generic function for all processing ??
def Filter(name, dataset=None, *args, **kargs):
    if name == 'average':
        return AverageFilter(dataset, *args, **kargs)

def Selector(name, dataset=None, *args, **kargs):
    if name == 'box-clip':
        return BoxClipper(dataset, *args, **kargs)
