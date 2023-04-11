"""
"""

import os


_cmip_ids = ['variable_id', 'table_id', 'source_id', 'experiment_id', 'member_id', 
'grid_label', 'time_range']


def load_cmip6_output(output):
    """
    Function for reading an `output` directory which .nc files inside. This returns 
    a dictionnary object with all information about files following CMIP6 Data Reference 
    Syntax (DRS). 
    
    For instance: {'variable_id': ['tos', 'areacella'],
                   'table_id': ['Omon', 'Ofx'],
                   'source_id': ['source', 'source'],
                        .                .
                        .                .
                   }
                   
    Note that for now the function do not follow exactly the directory structure 
    template from CMIP6 DRS. We assume a single directory where all .nc output files
    are stored.
    
    See https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html for going further.
    
    """
    cmip_dict = {cid: list() for cid in _cmip_ids}

    if not os.path.isdir(output):
        raise FileNotFoundError('No such directory: {}'.format(output))
    
    ncfiles = []
    for f in os.listdir(output):
        if os.path.isfile(os.path.join(output, f)) and f.endswith('.nc'):
            ncfiles.append(f[:-3])

    for f in ncfiles:
        append_cmip6_variable(f, cmip_dict)

    return cmip_dict


def variable_exists(fname, cmip_dict):
    """
    Check if the variable stored in fname already exists in the dictionnary
    """
    fids = fname.split('_')
    bids = [False] * len(fids)

    index = -1
    for i, fid in enumerate(fids):
        if index == -1:
            if fid in cmip_dict[_cmip_ids[i]]:
                index = cmip_dict[_cmip_ids[i]].index(fid)
                bids[i] = True
        else:
            if fid == cmip_dict[_cmip_ids[i]][index]:
                bids[i] = True

    return sum(bids) == len(bids)


def append_cmip6_variable(fname, cmip_dict, override=False):
    """
    Append a CMIP6 variable to the existing dictionnary
    """
    if variable_exists(fname, cmip_dict) and not override:
        return

    file_ids = fname.split('_')
    if len(file_ids) >= 6:
        for i, fid in enumerate(file_ids[:6]) :
            cmip_dict[_cmip_ids[i]].append(fid)

        if len(file_ids) > 6:
            cmip_dict[_cmip_ids[6]].append(file_ids[6])
        else:
            cmip_dict[_cmip_ids[6]].append('')

    else:
        print("Warning: file '{}' does not respect ".format(fname) + 
            "CMIP6 Data Reference Syntax.")


def extract_cmip6_variables(cmip_ids, cmip_key, cmip_dict, override=False):
    """
    Return a reduced dictionary with only wanted variables
    """
    if cmip_key not in cmip_dict:
        raise KeyError("Cannot find key '{}' in cmip dictionary".format(cmip_key))

    indexes = list()
    for i, val in enumerate(cmip_dict[cmip_key]):
        if val == cmip_ids or val in cmip_ids:
            indexes.append(i)

    new_dict = dict()
    for key in cmip_dict:
        values = new_dict.setdefault(key, list())
        for i in indexes:
            values.append(cmip_dict[key][i])
    
    if override:
        cmip_dict = new_dict
    
    return new_dict


def get_filename_from_drs(variable_id, cmip_dict):
    """
    Return real filename from CMIP6 DRQ dict.
    """
    inds = [i for i, x in enumerate(cmip_dict['variable_id']) if x == variable_id]
    
    if not inds:
        print("Variable '{}' not in Data Reference Syntax.".format(variable_id))
        return []

    filenames = list()
    for ivar in inds:
        varids = []
        for id in _cmip_ids:
            if cmip_dict[id][ivar] != '':
                varids.append(cmip_dict[id][ivar])
        
        fname = '_'.join(varids)
        fname += '.nc'

        filenames.append(fname)

    return filenames

