# General imports
from math import inf, exp
import numpy as np

def to_numpy(func):
  """
  Decorator function to convert functions
  to Numpy format
  """

  def numpy_func(w):
    return func(*w)

  return numpy_func

#----------------------------------------#
#------------ Ejercicio 1 ---------------#
#----------------------------------------#

# Apartado 1.1
def gradient_descent(w, lr, grad_f, f, max_iters, epsilon=-inf):
    """
    Arguments:
    - w: Initial point
    - lr: Learning rate
    - grad_f: Gradient of function to minimize
    - f: Function to minimize
    - max_iters: Max iterations before stopping
    - epsilon: Threshold to stop when reached

    Returns:
    - Minimun found
    - Num of iterations needed
    """
    iters = 0

    while iters < max_iters and f(w) < epsilon:
        w = w - lr * grad_f(w)
        iters += 1

    return w, iters

# Apartado 1.2
# Function to minize: E(u,v) = (ue^v - 2ve^(-u))^2

@to_numpy
def E(u, v):
    return (u*exp(v) - 2*v*exp(-u))**2

"""Partial derivate of E with respect to u"""
def Eu(u, v):
    return 2 * (u*exp(v) - 2*v*exp(-u)) * (exp(v) + 2*v*exp(-u))

"""Partial derivate of E with respect to v"""
def Ev(u, v):
    return 2 * (u*exp(v) - 2*v*exp(-u)) * (u*exp(v) - 2*exp(-u))

"""Gradient of E"""
@to_numpy
def gradE(u, v):
    return np.array([Eu(u, v), Ev(u, v)])
