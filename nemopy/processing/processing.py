"""
"""

class Processing:
    """
    Generic abstract class for processing xarray Dataset.
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
        if param in self._Parameters and param in self.__dict__:
            self.__dict__[param] = value

    def _set_default_parameters(self, **kargs):
        for param in self._Parameters:
            if param in kargs:
                self.__dict__[param] = kargs[param]
            else:
                self.__dict__[param] = self._Parameters[param]['default']
