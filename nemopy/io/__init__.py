from .readers import H5pyReader
from .writers import H5pyWriter


def Reader(name, dataset=None, *args, **kargs):
    if name == 'hdf5':
        return H5pyReader(dataset, *args, **kargs)

def Writer(name, dataset=None, *args, **kargs):
    if name == 'hdf5':
        return H5pyWriter(dataset, *args, **kargs)
