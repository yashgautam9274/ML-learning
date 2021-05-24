from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading dataset
iris = datasets.load_iris()

# Printing description and features
#print(iris.DESCR)
features = iris.data
labels = iris.target
#print(features[0],labels[0])

# -Features
#        - sepal length in cm
#        - sepal width in cm
#        - petal length in cm
#        - petal width in cm
# - class:
#                 - Iris-Setosa
#                - Iris-Versicolour
#                - Iris-Virginica

# Training the classifiers
clf = KNeighborsClassifier()
clf.fit(features,labels)

pred = clf.predict([[1,1,1,1]])
print(pred)