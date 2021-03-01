import pandas as pd
# Importing the dataset
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('mmc2_Sample.csv',delimiter=";")

dataset.drop(dataset.columns[[0,1,2,3,4,5,7,10,11,12,13,14,15,16,18,19]], axis=1, inplace=True)
dataset.drop(dataset[dataset['PKD_section']=='?'].index, inplace=True)
dataset=dataset.replace("?",0)
dataset=dataset.replace("YES",1)
dataset=dataset.replace("NO",0)
y = dataset.iloc[:, -1].values
del dataset['FRAUD']
dataset=pd.get_dummies(dataset,columns=(['LEGAL_FORM_group','PKD_section','TYPE_OF_VEHICLE']))
#PKD section NET ASSETS, LEGAL FORM group,TYPE OF VEHICLE,NUMBER OF EMPLOYEES,CUSTOMER FOR
X = dataset.iloc[:].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
list=[ 'poly', 'rbf', 'sigmoid']
from sklearn.svm import SVC
for a in list:
    classifier = SVC(kernel = a)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred)
    print("Test\n",cm1)
    print("Accuracy= ",classifier.score(X_test, y_test))
    scores = cross_val_score(classifier, X, y, cv=3)
    print('K-fold cross validation score '+ a +' : ',scores)
list=['gini', 'entropy']
from sklearn.ensemble import RandomForestClassifier
for a in list:
    classifier1 = RandomForestClassifier(n_estimators = 10, criterion = a, random_state = 0)
    classifier1.fit(X_train, y_train)
    y_pred = classifier1.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Test\n",cm)
    print("Accuracy= ",classifier1.score(X_test, y_test))
    scores = cross_val_score(classifier1, X, y, cv=3)
    print('K-fold cross validation score '+ a +' : ',scores)
