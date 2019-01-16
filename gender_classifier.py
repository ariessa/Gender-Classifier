from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score

# Classifiers

# 1 DecisionTreeClassifier
clf_tree = tree.DecisionTreeClassifier()

# 2 KNeighborsClassifier
clf_neigh = KNeighborsClassifier(n_neighbors = 3)

# 3 LogisticRegression
clf_logReg = LogisticRegression()

# 4 NaiveBayes
clf_gnb = GaussianNB()

# 5 SupportVectorMachine (SVM)
clf_svm = svm.SVC()


# Data set [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	[166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
	[159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


# Train models
clf_tree = clf_tree.fit(X, Y)
clf_neigh = clf_neigh.fit(X, Y)
clf_logReg = clf_logReg.fit(X, Y)
clf_gnb = clf_gnb.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)


# Predictions
prediction_clf = clf_tree.predict(X)
prediction_neigh = clf_neigh.predict(X)
prediction_logReg = clf_logReg.predict(X)
prediction_gnb = clf_gnb.predict(X)
prediction_svm = clf_svm.predict(X)

# Accuracy results
result_tree = accuracy_score(Y, prediction_clf)
result_neigh = accuracy_score(Y, prediction_neigh)
result_logReg = accuracy_score(Y, prediction_logReg)
result_gnb = accuracy_score(Y, prediction_gnb)
result_svm = accuracy_score(Y, prediction_svm)

print("Result for DecisionTreeClassifier: ", result_tree)
print("Result for KNeighborsClassifier: ", result_neigh)
print("Result for Logistic Regression: ", result_logReg)
print("Result for NaiveBayes: ", result_gnb)
print("Result for SVM: ", result_svm)

# Find best result
print("The comparison doesn't consider DecisionTreeClassifier and SVM")

if (result_neigh > result_logReg) and (result_neigh > result_gnb):
	print("The most accurate is KNeighborsClassifier, with : ", result_neigh)

elif (result_logReg > result_neigh) and (result_logReg > result_gnb):
	print("The most accurate is LogisticRegression: ", result_logReg)

else:
	print("The most accurate is NaiveBayes: ", result_gnb)
