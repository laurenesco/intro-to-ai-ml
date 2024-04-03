import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and partition the dataset
s_df = pd.read_csv('sonar.all-data')

X = s_df.iloc[:, :59]
y = s_df.iloc[:, 60]

# Create partitions of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)

# Create a perceptron and assign values to various hyperparameters
perceptron = Perceptron(random_state=1, max_iter=50, tol=0.005)
perceptron.fit(X_train, y_train)

# We predict with our built perceptron
yhat_train_perceptron = perceptron.predict(X_train)
yhat_test_perceptron = perceptron.predict(X_test)

print("Perceptron: Accuracy for training is %.2f" % (accuracy_score(y_train, yhat_train_perceptron)))
print("Perceptron: Accuracy for testing is %.2f" % (accuracy_score(y_test, yhat_test_perceptron)))