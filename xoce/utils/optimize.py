"""
Define some optimization functions and methods. 

The functions `runge_kunta_1` and `runge_kutta_4` are developped so that they can 
be used to solve a system of Ordinary Differential Equations (ODEs) of order k=3
and dimension n.
"""

import numpy as np


def runge_kutta_1(f, y0, dy0, t, args=()):
    """Explicit Runge-Kutta method of order 1.

    This method can be used to solve ODE systems (of order 3 only) along a time
    axis ``t``.

    Parameters
    ----------
    f : callable
        Right-hand side of the system. The calling signature is ``fun(y, dy, t, *args)``.
        Here ``t`` is a scalar, and ``y`` and ``dy`` are ndarrays with same shapes.
        These ndarrays can have shapes of dimension 1, 2 or greater.
    t : array_like, shape (l,)
        Times times where computed the integration.
    y0 : array_like
        Initial state for the function to integrate.
    dy0 : array_like
        Initial state for the first derivative.
    *args : all arguments to pass throw the callable function ``f``.


    Attributes
    ----------
    n : int
        Number of equations (n = mul(y0.shape))
    t : float
        Current time.
    y : ndarray
        Current state.
    dy : ndarray
        Current first derivative.
    """
    n  = len(t)
 
    y  = np.repeat(np.expand_dims(y0, y0.ndim),   n, axis=-1)
    dy = np.repeat(np.expand_dims(dy0, dy0.ndim), n, axis=-1)
        
    yi  = y0
    dyi = dy0
    
    timesel = [slice(None)] * y.ndim
    
    for i in range(n - 1):
        timesel[-1] = slice(i+1, i+2, 1)
        
        ti = t[i]
        dt = t[i+1] - t[i]

        yip1  = yi  + dt * dyi
        dyip1 = dyi + dt * f(yi, dyi, ti, *args)
        
        y[tuple(timesel)]  = np.expand_dims(yip1, yi.ndim)
        dy[tuple(timesel)] = np.expand_dims(dyip1, dyi.ndim)
                
        yi  = yip1
        dyi = dyip1
        
    return y, dy


def runge_kutta_4(f, y0, dy0, t, args=()):
    """Explicit Runge-Kutta method of order 4.

    This method can be used to solve ODE systems (of order 3 only) along a time
    axis ``t``.
     
    Parameters
    ----------
    f : callable
        Right-hand side of the system. The calling signature is ``fun(y, dy, t, *args)``.
        Here ``t`` is a scalar, and ``y`` and ``dy`` are ndarrays with same shapes.
        These ndarrays can have shapes of dimension 1, 2 or greater.
    t : array_like, shape (l,)
        Times times where computed the integration.
    y0 : array_like
        Initial state for the function to integrate.
    dy0 : array_like
        Initial state for the first derivative.
    *args : all arguments to pass throw the callable function ``f``.


    Attributes
    ----------
    n : int
        Number of equations (n = mul(y0.shape))
    t : float
        Current time.
    y : ndarray
        Current state.
    dy : ndarray
        Current first derivative.
    """
    n  = len(t)
 
    y  = np.repeat(np.expand_dims(y0, y0.ndim),   n, axis=-1)
    dy = np.repeat(np.expand_dims(dy0, dy0.ndim), n, axis=-1)
        
    yi  = y0
    dyi = dy0
    
    timesel = [slice(None)] * y.ndim
    
    for i in range(n - 1):
        timesel[-1] = slice(i+1, i+2, 1)
        
        ti = t[i]
        dt = t[i+1] - t[i]
        
        k1 = [dyi, f(yi, dyi, ti, *args)]
        k2 = [dyi + k1[1]*dt/2, f(yi + k1[0]*dt/2, dyi + k1[1]*dt/2, ti + dt/2, *args)]
        k3 = [dyi + k2[1]*dt/2, f(yi + k2[0]*dt/2, dyi + k2[1]*dt/2, ti + dt/2, *args)]
        k4 = [dyi + k3[1]*dt,   f(yi + k3[0]*dt,   dyi + k3[1]*dt,   ti + dt,   *args)]

        yip1  = yi  + (dt/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        dyip1 = dyi + (dt/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
                
        y[tuple(timesel)]  = np.expand_dims(yip1, yi.ndim)
        dy[tuple(timesel)] = np.expand_dims(dyip1, dyi.ndim)
                
        yi  = yip1
        dyi = dyip1
        
    return y, dy

