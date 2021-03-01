import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
dataset = pd.read_csv("mmc2_Sample.csv",sep=";")

# Thay thế kí tự đặc biệt ở các dòng có dấu :"?"
dictionary = {'?':'','NO':'0','YES':'1','no':'0','yes':'1'}
dataset.replace(dictionary,regex = False, inplace = True)

#Use float(x) with "NaN" as x to create a NaN value
nan_value = float("NaN")

#Call df.replace with an empty string as to_replace and a NaN value as value to replace all empty strings with NaN values
dataset.replace("", nan_value, inplace=True)


# Dropping all the rows with nan values
dataset.dropna( inplace=True)

#print df
dataset

X = dataset.iloc[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21]].values
Y = dataset.iloc[:, -1].values




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers =[('encoder',OneHotEncoder(),[5,14,15,16])],remainder='passthrough')


X_trans = np.array(ct.fit_transform(X))





from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_trans, Y, train_size = 0.8, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
list=[ 'poly', 'rbf', 'sigmoid', 'precomputed','linear']

#Phương pháp SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbfS' )
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, y_pred)
print(cm1)

scores = cross_val_score(classifier, X_trans, Y, cv=3)
print('K-fold cross validation score of SVM : ',scores)

from sklearn.metrics import accuracy_score
Y_pred = classifier.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("SVM score", score )

#phương pháp random forest
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 10, criterion ='entropy', random_state = 0)
classifier1.fit(X_train, Y_train)

cm = confusion_matrix(Y_test, classifier.predict(X_test))
print(cm) 

scores = cross_val_score(classifier, X_trans, Y, cv=3)
print('K-fold cross validation score of random forest: ',scores)


Y_pred = classifier1.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("random forest score", score )