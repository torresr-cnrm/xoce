import os
import sys 
sys.path.append(os.getcwd())

import xoce
from xoce.io import Writer


def test_hdf5_writer():
    """
    Simple test to check the Writer interface
    """
    err = 0
    datadir = os.path.join(os.getcwd(), 'tests', 'data')
    outfile = 'hdf5_writer_output.hdf5'

    exp = xoce.CMIPExperiment(path=datadir)
    exp.load()

    try:
        writer = Writer('hdf5', dataset=exp)
        writer.set('path', datadir)
        writer.set('filename', outfile)
        writer.write()
    except Exception:
        err = 1
    
    os.remove(os.path.join(datadir, outfile))

    assert err == 0

