# General imports
from math import inf, exp, sin, cos, pi
import numpy as np

#----------------------------------------#
#---------- Auxiliar Functions ----------#
#----------------------------------------#

def to_numpy(func):
    """
    Decorator function to convert functions
    to Numpy format
    """

    def numpy_func(w):
        return func(*w)

    return numpy_func

def wait():
    input("\n--- Pulsar tecla para continuar ---\n")

#----------------------------------------#
#------------ Ejercicio 1 ---------------#
#----------------------------------------#

# Apartado 1.1
def gradient_descent(w, lr, f, grad_f, max_iters, epsilon=-inf):
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

    while iters < max_iters and f(w) > epsilon:
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

def main_1_2():
    w = (1, 1)
    lr = 0.1
    minimun, iters = gradient_descent(w, lr, E, gradE, 10000000, 10**(-14))

    # TODO: Mostrar expresión del gradiente
    print ('GRADIENTE DESCENDENTE')
    print ('Ejercicio 1')
    print ('Numero de iteraciones: ', iters)
    print ('Coordenadas obtenidas: (', minimun[0], ', ', minimun[1],')')

    input("\n--- Pulsar tecla para continuar ---\n")

# Apartado 1.3
# Function to minize: f(x,y) = (x-2)^2 + 2(y+2)^2 + 2sin(2 \pi x)sin(2 \pi y)
@to_numpy
def f(x, y):
    return (x-2)**2 + 2*(y +2)**2 + 2*sin(2*pi*x)*sin(2*pi*y)
