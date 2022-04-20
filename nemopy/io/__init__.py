from .writers import H5pyWriter


def Writer(name, dataset=None, *args, **kargs):
    if name == 'hdf5':
        return H5pyWriter(dataset, *args, **kargs)
