"""
"""


class NKeyError(Exception):
    """
    Custom KeyError for NemopyObject.
    """
    def __init__(self, key, obj):
        msg  = "'{}' is not a parameter for {}. ".format(key, type(obj).__name__)
        msg += "Allowed keys are: {}".format(list(obj._Parameters.keys()))
        super().__init__(msg)

class NTypeError(Exception):
    """
    Custom TypeError for NemopyObject.
    """
    def __init__(self, key, val, obj):
        ktype = obj._Parameters[key].get('type', None)
        msg  = "'{}' for {} object should be of ".format(key, type(obj).__name__)
        msg += "type {}, not {}".format(ktype, type(val))
        super().__init__(msg)      


class NemopyObject:
    """
    Generic abstract class for interfacing nemopy objects as 
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
            raise NTypeError(param, value, self)
        else:
            if param in self.__dict__:
                self.__dict__[param] = value

    def _set_default_parameters(self, **kargs):
        for param in self._Parameters:
            if param in kargs:
                self.__dict__[param] = kargs[param]
            else:
                self.__dict__[param] = self._Parameters[param]['default']