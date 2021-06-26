import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score

import elm 

from mlxtend.classifier import StackingCVClassifier

import time
import sys


print('Enter Classifier/s: (Example: nb,svm)')
print('svm = Support Vector Machine')
print('nb = Naive Bayes')
print('rf = Random Forest')
print('knn = K-Nearest Neighbours')
print('elm = Extreme Learning Machine')
print('mlp = Multi-Layer Perceptron Neural Network')

user_input = input()


contains_elm = False
models = []

# Importing the required rows from Dataset 
# dataTable_DWT_With_New_Filter
dataset = pd.read_csv('Extracted_Features.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding the Dependent Variable
labelEncoder = LabelEncoder()
Y = labelEncoder.fit_transform(Y)


splits = 5
accuracy_sum = 0
sensitivity_sum = 0
specificity_sum = 0

time_on_average = 0

kf = KFold(n_splits=splits, shuffle=True)

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]

	# Applying Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.fit_transform(X_test)
	Y_pred = []



	if ',' in user_input:
		# multiple classifiers
		user_list = user_input.split(',')

		for entry in user_list:
			if entry == 'nb':
				models.append(GaussianNB())

			elif entry == 'svm':
				models.append(SVC(kernel = 'rbf', random_state = 0, gamma='scale', shrinking=True, probability=False, cache_size=500))

			elif entry == 'elm':
				contains_elm = True
				models.append(elm.ELM(hid_num=1000))

			elif entry == 'rf':
				models.append(RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0))

			elif entry == 'knn':
				models.append(KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2, weights='uniform'))

			elif entry == 'mlp':
				models.append(MLPClassifier(hidden_layer_sizes=(500,100), activation='relu', solver='adam', random_state=1, max_iter=1000))

			else:
				print('Incorrect Classifier Entry!')
				print('Quitting ... ')
				exit()



		if contains_elm:
			lr = LogisticRegression()
			sclf = StackingCVClassifier(classifiers=models, meta_classifier=lr, random_state=1)
			sclf.fit(X_train, Y_train)
			Y_pred = sclf.predict(X_test)
		else:
			lr = LogisticRegression()
			sclf = StackingCVClassifier(classifiers=models, use_probas=True, meta_classifier=lr, random_state=1)
			sclf.fit(X_train, Y_train)
			Y_pred = sclf.predict(X_test)



	else:
		# single classifiers
		if user_input == 'nb':
			classifier = GaussianNB()
			classifier.fit(X_train, Y_train)
			Y_pred = classifier.predict(X_test)

		elif user_input == 'svm':
			classifier = SVC(kernel = 'rbf', random_state = 0, gamma='scale', shrinking=True, probability=False, cache_size=500)
			classifier.fit(X_train, Y_train)
			Y_pred = classifier.predict(X_test)

		elif user_input == 'elm':
			classifier = elm.ELM(hid_num=1000)
			classifier.fit(X_train, Y_train)
			Y_pred = classifier.predict(X_test)

		elif user_input == 'rf':
			classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
			classifier.fit(X_train, Y_train)
			Y_pred = classifier.predict(X_test)

		elif user_input == 'knn':
			classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2, weights='uniform')
			classifier.fit(X_train, Y_train)		
			Y_pred = classifier.predict(X_test)

		elif user_input == 'mlp':
			classifier = MLPClassifier(hidden_layer_sizes=(500,100), activation='relu', solver='adam', random_state=1, max_iter=1000)
			classifier.fit(X_train, Y_train)
			Y_pred = classifier.predict(X_test)		


		else:
			print('Incorrect Classifier Entry!')
			print('Quitting ... ')
			exit()



	#Making the Confusion Matrix
	confusionMatrix = confusion_matrix(Y_test, Y_pred)
	tn, fp, fn, tp = confusionMatrix.ravel()
	sensitivity = tp/(tp + fn)
	specificity = tn/(tn + fp)

	accuracy_sum += accuracy_score(Y_test, Y_pred)
	sensitivity_sum += sensitivity
	specificity_sum += specificity

	print(confusionMatrix)
	print("accuracy: {}".format(accuracy_score(Y_test, Y_pred)))
	print("sensitivity: {}".format(sensitivity))
	print("specificity: {}".format(specificity))
	print("summation: {}".format(tn+fp+fn+tp))
	print("\n")


print("\n")
print("FINAL Classifier Scores")
print("accuracy: {}".format(accuracy_sum/splits))
print("sensitivity: {}".format(sensitivity_sum/splits))
print("specificity: {}".format(specificity_sum/splits))






