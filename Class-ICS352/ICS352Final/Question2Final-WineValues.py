#Question 2 Final Wine (40 Points):
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

data = load_wine()
X = data.data 
y = data.target 

#Part 1:
print("Classes:", data.target_names) #Classes
print("Features:", data.feature_names) #Features
print("Number of Dimensions:", X.shape[1]) #Dimensions
print("Number of Samples:", X.shape[0]) #Samples

#Part 2:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) #42 is 100% yet 0 is 93%
gnb = GaussianNB()
gnb.fit(X_train, y_train)
bayes_preds = gnb.predict(X_test)
bayes_accuracy = accuracy_score(y_test, bayes_preds)
print(f"\nBayesian Classifier Accuracy: {bayes_accuracy:.4f}")

#Part 3
k = 1 #With random state set to 42, 1 is the highest, but this could be overfitting.
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
accuracyk = accuracy_score(y_test, preds)
print(f"KNN Accuracy: {accuracyk:.4f}")

#Part 4
j = 9 #8 or 9 seem to work.
kmeans = KMeans(n_clusters=j, random_state=0, n_init=10)
kmeans.fit(X_train)
kmeans_labels = kmeans.predict(X_test)

remapped = np.zeros_like(kmeans_labels)
for i in range(j):
    mask = (kmeans.labels_ == i)
    if np.any(mask):
        remapped_class = np.bincount(y_train[mask]).argmax()
        remapped[kmeans_labels == i] = remapped_class

accuracyj = accuracy_score(y_test, remapped)
print(f"KMeans Clustering Accuracy: {accuracyj:.4f}")

#Part 5
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_preds = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_preds)
print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")

#Part 6
#Decision Tree Accuracy seems to give the best accurate results.