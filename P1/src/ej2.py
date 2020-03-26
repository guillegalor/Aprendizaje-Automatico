# General imports
import numpy as np
import matplotlib.pyplot as plt
import random

# Set seed for random
random.seed(1)

def wait():
    input("\n--- Pulsar tecla para continuar ---\n")

#----------------------------------------#
#------------ Ejercicio 2 ---------------#
#----------------------------------------#

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

# Stochastic gradiente descent
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
        iterations += 1
        batch_start += batch_size
        if batch_start >= len(x):
            batch_start = 0

    return w

# Data reading
x, y = readData('../data/X_train.npy', '../data/y_train.npy')
# Test data reading
x_test, y_test = readData('../data/X_test.npy', '../data/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico
w = sgd(x, y)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x, y, w))
print ("Eout: ", Err(x_test, y_test, w))

wait()
