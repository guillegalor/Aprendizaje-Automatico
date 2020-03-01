# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# Seed
seed = 1

# General imports
import collections, numpy

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# Scikit Learn imports
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#----------------------------------------
#------------ Parte 1 -------------------
#----------------------------------------

# Reads iris dataset
iris = datasets.load_iris()

# Gets features and labels
X = iris.data[:, -2:]  # Take only last 2 features
y = iris.target

# Creates figure for plotting
fig, ax = plt.subplots()
plt.clf()

# Gets limits for the plot
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# Plot the training points
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'g', 'b']),
            edgecolor='k')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.legend(handles=scatter.legend_elements()[0], labels=['Flower 0', 'Flower 1', 'Flower 2'])
# plt.show()

#----------------------------------------
#------------ Parte 2 -------------------
#----------------------------------------

# Gets training and test sets
X_train = []
X_test = []
y_train = []
y_test = []

print(X)

# Dictionary to determine if an element goes to test or training set
keys= dict.fromkeys(y)
counter = {key: 0 for key in keys}

for features, label in zip(X, y):
    if counter[label] != 4:
        X_train.append(features)
        y_train.append(label)
    else:
        X_test.append(features)
        y_test.append(label)

    counter[label] = (counter[label] + 1) % 5

print(collections.Counter(y_train))
print(collections.Counter(y_test))
