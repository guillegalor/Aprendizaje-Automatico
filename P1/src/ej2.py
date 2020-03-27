# General imports
import numpy as np
import random

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set seed for random
random.seed(1)

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

#------------ Ejercicio 1 ---------------#

def readData(file_x, file_y):
    # Load files with data
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []

    # We only keep elements with class 1 or 5
    # (but we store the class as -1 and 1 respectively )
    for i in range(0,datay.size):
        if datay[i] == 5 or datay[i] == 1:
            if datay[i] == 5:
                y.append(1)
            else:
                y.append(-1)
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y

# Gets de error
def Err(x,y,w):
    wN = np.linalg.norm(x.dot(w) - y)**2
    return wN/len(x)

# Gradient of the error function
def dErr(x, y, w):
    return 2/len(x) * (x.T.dot(x.dot(w) - y))

def sgd(x, y, lr=0.01, max_iters=1000, batch_size=32):
    """
    Stochastic gradient descent applied to linear regression
    - x              : Data (array of arrays with features of each element)
    - y              : Labels for each element
    - lr             : Learning rate
    - max_iters      : Max iterations to be done
    - minibatch_size : Batch size
    """

    w = np.zeros(3)
    iters = 0
    idxs = [i for i in range(len(x))]
    batch_start = 0

    while iters < max_iters:
        # Shuffles the batch if it is reseted
        if batch_start == 0:
            random.shuffle(idxs)

        # Gets indexes and updates w
        batch_idxs = idxs[batch_start: batch_start + batch_size]
        w = w - lr * dErr(x[batch_idxs], y[batch_idxs], w)

        # Updates iterations and batch
        iters += 1
        batch_start += batch_size
        if batch_start >= len(x):
            batch_start = 0

    return w

def pseudoinverse(x, y):
    """
    Pseudoinverse algorithm applied to linear regression
    - x              : Data (array of arrays with features of each element)
    - y              : Labels for each element
    """

    # Gets singular value decomposition
    u, s, v = np.linalg.svd(x)
    # Compute the inverse of the matrix D (only use the diagonal part of the matrix d)
    # avoiding 0 values (or really close to 0)
    d = np.diag([0 if np.allclose(p, 0) else 1/p for p in s])
    # Compute the pseudoinverse of X
    pseudoinverse = v.T.dot(d).dot(d).dot(v).dot(x.T)
    return pseudoinverse.dot(y)

# Data reading
x, y = readData('../data/X_train.npy', '../data/y_train.npy')
# Test data reading
x_test, y_test = readData('../data/X_test.npy', '../data/y_test.npy')
print(x.shape, y.shape)

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')

# Gradiente descendente estocástico
w = sgd(x, y)
print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

w = pseudoinverse(x, y)
print('\nBondad del resultado para pseudo-inversa:')
print("  Ein:  ", Err(x, y, w))
print("  Eout: ", Err(x_test, y_test, w))

wait()

#------------ Ejercicio 2 ---------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size, size, (N, d))

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

X = simula_unif(1000, 2, 1)
plt.plot([a[0] for a in X], [a[1] for a in X], 'bo')
plt.show()

# b) Mapa de etiquetas
@to_numpy
def f(x1, x2):
    return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)

# Compute the labels of 90% of elements
y = f(X[:900].T)
# Generate the remaining 10% randomly
y_random = np.random.choice([-1, 1], 100)
# Join everything
y = np.hstack((y, y_random))

# Generates the label map and
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'g']))
plt.show()

# c) Ajustar un modelo de regresión lineal utilizando el gradiente descendiente estocástico
X = np.array([[1, x[0], x[1]] for x in X])

w = sgd(X,y)
print ('Bondad del resultado en el experimento para grad. descendente estocástico:\n')
print ("Ein: ", Err(X, y, w))

# Ejecutar el experimento a)-c) 1000 veces
