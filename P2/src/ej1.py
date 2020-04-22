# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Guillermo Galindo Ortuño
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    input("\n--- Pulsar ENTER para continuar ---\n")

#----------------------------------------#

# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b

# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])

plt.plot([a[0] for a in x], [a[1] for a in x], 'bo')
plt.title('Nube de puntos uniforme')
plt.show()
wait()

x = simula_gaus(50, 2, np.array([5,7]))

plt.plot([a[0] for a in x], [a[1] for a in x], 'bo')
plt.title('Nube de puntos gaussiana')
plt.show()
wait()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def genera_datos(intervalo, num_datos, dim):
    """
    Genera recta y datos uniformes en un intervalo.
    Argumentos:
        * intervalo : intervalo en el que están todos los datos
        * num_datos : numero de vectores de datos
        * dim       : dimensión de cada vector de datos
    Returns:
        * a, b : Parametros de la recta (ax + b)
        * X    : conjunto de datos
        * y    : conjunto de etiquetas
    """

    a, b = simula_recta(intervalo)

    X = simula_unif(num_datos, dim, intervalo)
    y = np.empty((num_datos,))
    for i in range(num_datos):
        y[i] = f(X[i, 0], X[i, 1], a, b)

    return a, b, X, y

intervalo = [-50,50]
N = 100

a, b, X, y = genera_datos(intervalo, N, 2)

# Muestra las etiquetas
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=ListedColormap(['r', 'g']))
plt.plot([-50, 50], [a*(-50 ) + b, a*(50) + b])
plt.title('Nube de puntos clasificada sin ruido')
plt.show()

wait()

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido
y_noise = y.copy()

# Modifica un 10% aleatorio de cada etiqueta
for label in {-1, 1}:
  y_lab = np.nonzero(y == label)[0]
  y_rand = np.random.choice(y_lab, math.ceil(0.1*len(y_lab)), replace=False)
  y_noise[y_rand] = -y_noise[y_rand]

plt.scatter(X[:, 0], X[:, 1], c=y_noise.flatten(), cmap=ListedColormap(['r', 'g']))
plt.plot([-50, 50], [a*(-50 ) + b, a*(50) + b])
plt.title('Nube de puntos clasificada con ruido')
plt.show()

wait()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01

    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
                cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')

    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

clasificadores = [
    lambda x: x[:, 1] - a*x[:, 0] - b,
    lambda x: (x[:, 0]-10)**2 + (x[:, 1]-20)**2 - 400,
    lambda x: 0.5*(x[:, 0]+10)**2 + (x[:, 1]-20)**2 - 400,
    lambda x: 0.5*(x[:, 0]-10)**2 - (x[:, 1]+20)**2 - 400,
    lambda x: x[:, 1] - 20*x[:, 0]**2 - 5*x[:, 0] + 3
]

titulos = [
    ' Recta    ',
    ' Elipse 1 ',
    ' Elipse 2 ',
    ' Elipse 3 ',
    ' Parábola ',
]

for clasif, titulo in zip(clasificadores, titulos):
    plot_datos_cuad(X, y_noise, clasif, title=titulo)

wait()

###############################################################################
###############################################################################
###############################################################################

# Datos con coordenada homogénea
X_h = np.hstack((np.ones((N,1)), X))

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, labels, max_iter, vini):
    """
    Calcula el hiperplano solución a un problema de clasificación binaria.
    Argumentos:
        * datos    : matriz con los datos (con coordenada homogénea)
        * label    : vector de etiquetas
        * max_iter : máximo de iteraciones permitidas
        * vini     : valor inicial
    Return:
        * w        : vector de pesos
        * iters    : número de iteraciones
    """

    w  = vini.copy()

    for i in range(max_iter):
        w_old = w.copy()

        for item, label in zip(datos, labels):
            if signo(w.dot(item)) != label:
                w += label * item
        if np.all(w == w_old):
            return w, i+1

    return w, i+1

# Parameters
max_iters = 1000
num_coords = np.shape(X_h)[1]

# Initial vector 0
vini_0 = np.zeros(num_coords)
w_0, it_0 = ajusta_PLA(X_h, y, max_iters, vini_0)

# Random initializations
iterations = []
for i in range(0,10):
    w, iters = ajusta_PLA(X_h, y, max_iters, np.random.rand(num_coords))
    iterations.append(iters)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

wait()

# Ahora con los datos del ejercicio 1.2.b

# Initial vector 0
vini_0 = np.zeros(num_coords)
w_0, it_0 = ajusta_PLA(X_h, y_noise, max_iters, vini_0)

# Random initializations
iterations = []
for i in range(0,10):
    w, iters = ajusta_PLA(X_h, y_noise, max_iters, np.random.rand(num_coords))
    iterations.append(iters)

print('Valor medio de iteraciones necesario para converger (datos con ruido): {}'.format(np.mean(np.asarray(iterations))))

wait()

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def ErrRL(x, y, w):
    return np.mean(np.log(1 + np.exp(-y.dot(x.dot(w).T))))

def gradErrLR(x, y, w):
    return -np.mean((y.dot(x))/(1 + np.exp(y.dot(x.dot(w).T))))

def sgdRL(datos, labels, eta=0.01, batch_size=1):
    """
    Algoritmo de regresión logística utilizando
    el SGD
    Argumentos:
        * datos      : matriz con los datos (con coordenada homogénea)
        * label      : vector de etiquetas
        * eta        : tasa de aprendizaje
        * batch_size : tamaño del batch
    Return:
        * w          : vector de pesos
    """

    N, dim = datos.shape
    w = np.zeros(dim)
    min_reached = False
    idxs = np.arange(N)
    batch_start = 0

    while not min_reached:
        w_old = w.copy()
        if batch_start == 0:
            np.random.shuffle(idxs)

        batch_idxs = idxs[batch_start: batch_start + batch_size]
        w = w - eta*gradErrLR(datos[batch_idxs], labels[batch_idxs][None, :], w)

        min_reached = np.linalg.norm(w - w_old) <= 0.01

        if batch_start >= N:
            batch_start = 0

    return w

intervalo = [0,2]
d = 2
N = 100

a, b, X, y = genera_datos(intervalo, N, d)
X_h = np.hstack((np.ones((N, 1)), X))

w = sgdRL(X_h, y)

wait()

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

#CODIGO DEL ESTUDIANTE
N_test = 1000
X_test = simula_unif(N_test, 2, intervalo)
X_test_h = np.hstack((np.ones((N_test, 1)), X_test))

y_test = np.empty((N_test, ))
for i in range(N_test):
    y_test[i] = f(X_test[i, 0], X_test[i, 1], a, b)

print("Error: {}".format(ErrRL(X_test_h, y_test[None, :], w)))

wait()

###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('data/X_train.npy', 'data/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('data/X_test.npy', 'data/y_test.npy', [4,8], [-1,1])

#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

wait()

#LINEAR REGRESSION FOR CLASSIFICATION

#CODIGO DEL ESTUDIANTE

wait()

#POCKET ALGORITHM

#CODIGO DEL ESTUDIANTE

wait()

#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
