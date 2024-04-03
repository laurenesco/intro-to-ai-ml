import sklearn
import urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Loading the dataset
data = load_wine()
X_, y = data.data, data.target
target_names = data.target_names

print(X_.shape)

# Standardize all features
X = StandardScaler().fit_transform(X_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5,3), random_state=1)

# We use the fit() function to train the neural network
clf.fit(X_train, y_train)

print(clf.predict_proba(X_test[2:3,:]))
print(clf.predict(X_test[2:3,:]))
print(clf.score(X_test, y_test))
