import urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Loading the dataset
data = load_diabetes()
X, y = data.data, data.target

# The original X has 30 features
print(X.shape)

data_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

regr = MLPRegressor(hidden_layer_sizes = (3,5), random_state=1).fit(X_train, y_train)

prediction_result = regr.predict(X_test[2:3,:])
print(prediction_result)

error = regr.score(X_test, y_test)/len(y_test)
print(error)
