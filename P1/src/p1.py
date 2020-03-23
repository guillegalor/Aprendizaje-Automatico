# General imports
from math import inf, exp, sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

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

    print ('GRADIENTE DESCENDENTE')
    print ('Ejercicio 1')
    print ('Numero de iteraciones: ', iters)
    print ('Coordenadas obtenidas: (', minimun[0], ', ', minimun[1],')')

    wait()

# Apartado 1.3
# Function to minize: f(x,y) = (x-2)^2 + 2(y+2)^2 + 2sin(2 \pi x)sin(2 \pi y)
@to_numpy
def f(x, y):
    return (x-2)**2 + 2*(y +2)**2 + 2*sin(2*pi*x)*sin(2*pi*y)

def fx(x, y):
    return 2*(2*pi*cos(2*pi*x)*sin(2*pi*y) + x - 2)

def fy(x, y):
    return 4*(pi*sin(2*pi*x)*cos(2*pi*y) + y + 2)

@to_numpy
def gradf(x, y):
    return np.array([fx(x, y), fy(x, y)])

def gd_graph(w, lr, f, grad_f, max_iters):
    iters = 0
    values_f = []

    while iters < max_iters:
        values_f.append(f(w))
        w = w - lr * grad_f(w)
        iters += 1

    plt.plot(range(max_iters), values_f, 'bo')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.show()

def main_1_3a():
    w = (1, 1)

    print ('Resultados ejercicio 2\n')
    print ('\nGrafica con learning rate igual a 0.01')
    gd_graph(w, 0.01, f, gradf, 50)
    wait()
    print ('\nGrafica con learning rate igual a 0.1')
    gd_graph(w, 0.1, f, gradf, 50)
    wait()

def main_1_3b():
    lr = 0.01

    w = (2.1, -2.1)
    w, _ = gradient_descent(w, lr, f, gradf, 50)
    print ('Punto de inicio: (2.1, -2.1)\n')
    print ('(x,y) = (', w[0], ', ', w[1],')\n')
    print ('Valor minimo: ', f(w))

    wait()

    w = (3.0, -3.0)
    w, _ = gradient_descent(w, lr, f, gradf, 50)
    print ('Punto de inicio: (3.0, -3.0)\n')
    print ('(x,y) = (', w[0], ', ', w[1],')\n')
    print ('Valor minimo: ',f(w))

    wait()

    w = (1.5, 1.5)
    w, _ = gradient_descent(w, lr, f, gradf, 50)
    print ('Punto de inicio: (1.5, 1.5)\n')
    print ('(x,y) = (', w[0], ', ', w[1],')\n')
    print ('Valor minimo: ',f(w))

    wait()

    w = (1.0, -1.0)
    w, _ = gradient_descent(w, lr, f, gradf, 50)
    print ('Punto de inicio: (1.0, -1.0)\n')
    print ('(x,y) = (', w[0], ', ', w[1],')\n')
    print ('Valor mínimo: ',f(w))

    wait()
