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
    y = np.array(y, np.float64)[:, None]

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

    w = np.zeros((x.shape[1], 1))
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

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')

# Gradiente descendente estocástico
w_sgd = sgd(x, y)
print ('Bondad del resultado para grad. descendente estocástico:\n')
print ("Ein: ", Err(x, y, w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))

w_psi = pseudoinverse(x, y)
print('\nBondad del resultado para pseudo-inversa:')
print("  Ein:  ", Err(x, y, w_psi))
print("  Eout: ", Err(x_test, y_test, w_psi))

wait()

print('Rectas de regresión obtenidas utilizando los algoritmos SGD y Pseudo-Inversa:')
# Plots
x_1 = x[np.where(y == -1)[0]]
x_5 = x[np.where(y == 1)[0]]

scatter = plt.scatter(x_1[:, 1], x_1[:, 2], c='r', edgecolor='k', label='1')
scatter = plt.scatter(x_5[:, 1], x_5[:, 2], c='b', edgecolor='k', label='5')
plt.xlabel('Valor medio nivel gris')
plt.ylabel('Simetría vertical')

plt.plot(x[:,1], (-w_sgd[1]*x[:,1] - w_sgd[0])/w_sgd[2], ls='--' ,label='SGD')
plt.plot(x[:,1], (-w_psi[1]*x[:,1] - w_psi[0])/w_psi[2], ls='--' ,label='Pseudoinverse')
plt.legend()

plt.show()

wait()

#------------ Ejercicio 2 ---------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size, size, (N, d))

# EXPERIMENTO
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

print ('Mapa de puntos')
X = simula_unif(1000, 2, 1)
plt.plot([a[0] for a in X], [a[1] for a in X], 'bo')
plt.show()
wait()

# b) Mapa de etiquetas
@to_numpy
def f(x1, x2):
    return np.sign((x1 - 0.2)**2 + x2**2 - 0.6)

# Compute the labels of 90% of elements
y = f(X[:900].T)
# Generate the remaining 10% randomly
y_random = np.random.choice([-1, 1], 100)
# Join everything
y = np.hstack((y, y_random))[:, None]

print('Mapa de clases')
# Generates the label map and
scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=ListedColormap(['r', 'g']))
plt.show()
wait()

# c) Ajustar un modelo de regresión lineal utilizando el gradiente descendiente estocástico
# Creates feature vector
X = np.array([[1, x[0], x[1]] for x in X])

w = sgd(X,y)
print ('Bondad del resultado en el experimento para grad. descendente estocástico: \n')
print ("Ein: ", Err(X, y, w))

# Plots
x_a = X[np.where(y == -1)[0]]
x_b = X[np.where(y == 1)[0]]

scatter = plt.scatter(x_a[:, 1], x_a[:, 2], c='r', edgecolor='k', label='-1')
scatter = plt.scatter(x_b[:, 1], x_b[:, 2], c='g', edgecolor='k', label='1')

print('Ejemplo regresión lineal')
plt.plot(x[:,1], (-w[1]*x[:,1] - w[0])/w[2], ls='--')
plt.legend()

plt.show()
wait()

# d) Ejecutar el experimento a)-c) 1000 veces

# Agroup data generation in one unique function
def data_genarator(linear=True):
    X = simula_unif(1000, 2, 1)
    y = f(X[:900].T)
    y_random = np.random.choice([-1, 1], 100)
    y = np.hstack((y, y_random))[:, None]
    if (linear):
        X = np.array([[1, x[0], x[1]] for x in X])
    else:
        X = np.array([[1, x[0], x[1], x[0]*x[1], x[0]*x[0], x[1]*x[1]] for x in X])

    return X, y

def experiment(linear=True):
    x, y = data_genarator(linear)
    x_test, y_test = data_genarator(linear)
    w = sgd(x,y)

    # Training error
    Ein = Err(x, y, w)
    # Test error
    Eout = Err(x_test, y_test, w)

    return Ein, Eout, w

# Shows plot with no linear regression
_, _, w = experiment(linear=False)
x_a = X[np.where(y == -1)[0]]
x_b = X[np.where(y == 1)[0]]

scatter = plt.scatter(x_a[:, 1], x_a[:, 2], c='r', edgecolor='k', label='-1')
scatter = plt.scatter(x_b[:, 1], x_b[:, 2], c='g', edgecolor='k', label='1')

xmin, xmax = X[:, 1].min() - .5, X[:, 1].max() + .5
ymin, ymax = X[:, 2].min() - .5, X[:, 2].max() + .5
xp, yp = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
F = -w[0] - w[1] * xp - w[4] * xp * xp
G = w[2] * yp + w[3] * xp * yp + w[5] * yp * yp
plt.contour(xp, yp, (G - F), levels = [0])

plt.legend()

print('Ejemplo regresión no lineal')
plt.show()
wait()

# Perform the 1000 experiments
Eins = []
Eouts = []
for _ in range(1000):
    Ein, Eout, _ = experiment()
    Eins.append(Ein)
    Eouts.append(Eout)

Ein_medio = sum(Eins)/1000
Eout_medio = sum(Eouts)/1000

print ('Errores Ein y Eout medios tras 1000reps del experimento lineal:\n')
print ("Ein media: ", Ein_medio)
print ("Eout media: ", Eout_medio)

# Perform the 1000 no linear experiments
Eins = []
Eouts = []
for _ in range(1000):
    Ein, Eout, _ = experiment(linear=False)
    Eins.append(Ein)
    Eouts.append(Eout)

Ein_medio = sum(Eins)/1000
Eout_medio = sum(Eouts)/1000

print ('Errores Ein y Eout medios tras 1000reps del experimento no lineal:\n')
print ("Ein media: ", Ein_medio)
print ("Eout media: ", Eout_medio)
