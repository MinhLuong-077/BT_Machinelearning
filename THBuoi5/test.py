import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
# Importing the dataset
dataset_train = pd.read_csv('Social_Network_Ads_Train.csv')
dataset_test = pd.read_csv('Social_Network_Ads_Test.csv')
X_train = dataset_train.iloc[:, [3, 4]].values
y_train = dataset_train.iloc[:, -1].values
X_test = dataset_test.iloc[:, [3, 4]].values
y_test = dataset_test.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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
classifier = KNeighborsClassifier(n_neighbors = 1,metric='cosine')
classifier=classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Test\n",cm)
VisualizingResult(classifier, X_train)
VisualizingDataset(X_train, y_train)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(loc='upper right')
plt.show()

VisualizingResult(classifier, X_test)
VisualizingDataset(X_test, y_test)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(loc='upper right')
plt.show()