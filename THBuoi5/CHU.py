import tensorflow as tf
data = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = data.load_data()
X_train = X_train / 255.0
X_test = X_test/ 255.0
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)
score = classifier.score(X_test,Y_test)
print("Accuracy= " , score)

