import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
data = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = data.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

RESHAPED = 784
X_train = X_train.reshape(X_train.shape[0], RESHAPED)
X_test = X_test.reshape(X_test.shape[0], RESHAPED)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print("Test\n",cm)
def VisualizingDataset(X_, Y_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    for i, label in enumerate(np.unique( Y_)):
        plt.scatter(X1[Y_ == label], X2[Y_ == label],color = ListedColormap(("red", "green"))(i),label = label)
def VisualizingResult(model, X_):
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    X1_range = np.arange(start= X1.min()-1, stop= X1.max()+1,step = 0.01)
    X2_range = np.arange(start= X2.min()-1, stop= X2.max()+1,step = 0.01)
    X1_matrix, X2_matrix = np.meshgrid(X1_range, X2_range)
    X_grid= np.array([X1_matrix.ravel(), X2_matrix.ravel()]).T
    Y_grid= model.predict(X_grid).reshape(X1_matrix.shape)
    plt.contourf(X1_matrix, X2_matrix, Y_grid, alpha = 0.5,cmap = ListedColormap(("red","green")))

score = classifier.score(X_test,Y_test)
print("accuracy=%.2f%%" , score * 100)


