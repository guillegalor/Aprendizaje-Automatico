# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

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

plt.show()
