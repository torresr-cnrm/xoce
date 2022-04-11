"""
Python module for equation and formulas management.
"""

import importlib.util
import inspect
import os


class CalcManager:
    def __init__(self, dataset):
        self._dataset   = dataset
        self._functions = dict()

        # load all formulas
        self._load_formulas()


    @property
    def functions(self):
        return list(self._functions.keys())


    def calculate(self, variable):
        """
        Calculate a variable by its name.
        """
        if variable in self._dataset.variables:
            return self._dataset[variable]
        
        if variable not in self._functions:
            raise Exception("Unknow formula: '{}'".format(variable))

        # get class, function and its signature
        clss = self._functions[variable]
        func = clss.calculate
        prms = inspect.signature(func).parameters

        args = list()
        for p in prms:
            if p in self._dataset.variables:
                args.append(self._dataset[p])
            elif not (prms[p].default == inspect._empty):
                args.append(prms[p].default)
            else:
                args.append(self.calculate(p))

        # TODO: make a test on arguments shape and dimensions here ?? 
        # use calculate method which is defined for all functions
        darray = func(*tuple(args))
        
        # finally add variable attrs
        for attr in clss.__dict__:
            if not attr.startswith('__') and attr != 'calculate':
                darray.attrs[attr] = clss.__dict__[attr]
        self._dataset[variable] = darray

        return darray


    def is_calculable(self, variable):
        """
        Check if a variable is calculable. 
        """
        return True


    def feed(self, **kargs):
        """
        Add new variables in dataset.
        """
        for k in kargs:
            # TODO: make a test if dimensions are OK with current dataset
            self._dataset[k] = kargs[k]

    
    def _load_formulas(self):
        """
        Loop over all files in '../formulas' directory in order
        to inspect classes. Each class correspond to a physical
        equation.
        """
        root  = os.path.dirname(__file__)
        fpath = os.path.join(os.path.realpath(root), 'formulas')

        flist = [f for f in os.listdir(fpath) if f.endswith('.py')]
        for f in flist:
            modname, _ = os.path.splitext(f)
            full_filename = os.path.join(fpath, f)
            spec = importlib.util.spec_from_file_location(modname, full_filename)
            module = importlib.util.module_from_spec(spec)

            # load module in file f
            spec.loader.exec_module(module)

            allmembers = inspect.getmembers(module)
            clsmembers = [m[1] for m in allmembers if inspect.isclass(m[1])]

            for cls in clsmembers:
                if cls.__name__ not in self._functions:
                    self._functions[cls.__name__] = cls

