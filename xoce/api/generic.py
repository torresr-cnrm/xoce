"""
"""

import xoce


class NKeyError(Exception):
    """
    Custom KeyError for XoceObject.
    """
    def __init__(self, key, obj):
        msg  = "'{}' is not a parameter for {}. ".format(key, type(obj).__name__)
        msg += "Allowed keys are: {}".format(list(obj._Parameters.keys()))
        super().__init__(msg)

class NTypeError(Exception):
    """
    Custom TypeError for XoceObject.
    """
    def __init__(self, key, val, obj):
        ktype = obj._Parameters[key].get('type', None)
        msg  = "'{}' for {} object should be of ".format(key, type(obj).__name__)
        msg += "type {}, not {}".format(ktype.__name__, type(val).__name__)
        super().__init__(msg)      


class XoceObject:
    """
    Generic abstract class for interfacing xoce objects as 
    processing and io classes.
    """
    _Parameters = dict()

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    # generic parameter setter
    def set(self, param, value):
        if param not in self._Parameters:
            raise NKeyError(param, self)
        elif not isinstance(value, self._Parameters[param]['type']):
            try:
                ptype = self._Parameters[param]['type']
                if type(ptype) is tuple:
                    ptype = ptype[0]
                self.__dict__[param] = ptype(value)
            except Exception:
                raise NTypeError(param, value, self)
        else:
            self.__dict__[param] = value

    def _set_default_parameters(self, **kargs):
        for param in self._Parameters:
            if param in kargs:
                self.set(param, kargs[param])
            else:
                self.__dict__[param] = self._Parameters[param]['default']


def set_attrs(func):
    """
    Processing decorator. This function add 'attrs' in processed xr.Dataset.
    """

    def processing(xobject):
        processed = func(xobject) # processed is a xarray.Dataset instance

        # keep input dataset or experiment attrs
        processed.attrs.update(xobject.dataset.attrs)

        # add processing attrs
        classname = type(xobject).__name__

        old_proc = processed.attrs.get('xoce_processing', '')
        if old_proc:
            new_attrs = {'xoce_processing': "{} + {}".format(old_proc, classname)}
        else:
            new_attrs = {'xoce_processing': classname}
            
        for p in xobject._Parameters:
            name = '{}_{}'.format(classname, p)
            attr = xobject.__dict__.get(p, '')
            new_attrs[name] = str(attr)

        new_attrs['xoce_version'] = xoce.__version__
        processed.attrs.update(new_attrs)

        return processed

    return processing

