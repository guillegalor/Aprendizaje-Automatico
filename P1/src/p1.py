# General imports
from math import inf
from sympy import *

def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""

  def numpy_func(w):
    return func(*w)

  return numpy_func

#----------------------------------------#
#------------ Ejercicio 1 ---------------#
#----------------------------------------#

# Apartado 1.1
def gradient_descent(w, lr, grad_f, f, max_iters, epsilon):
    iters = 0

    while iters < max_iters and f(w) < epsilon:
        w = w - lr * grad_f(w)
        iters += 1

    return w, iters

# Apartado 1.2
# Function to minize: E(u,v) = (ue^v - 2ve^(-u))^2

def E(w):
    return
