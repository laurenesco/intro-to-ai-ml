import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and partition the dataset
s_df = pd.read_csv('sonar.all-data')

X = s_df.iloc[:, :59]
y = s_df.iloc[:, 60]

# Create partitions of training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)

# Build an MLP network
mlp = MLPClassifier(solver='sgd', max_iter=50, verbose=True, random_state=1,
                    learning_rate_init=.1, hidden_layer_sizes=(60, 100, 2))
mlp.fit(X_train, y_train)

# Make predictions with our new classifier
yhat_train_mlp = mlp.predict(X_train)
yhat_test_mlp = mlp.predict(X_test)

print("Multilayer Perceptron: Accuracy for training is %.2f" % (accuracy_score(y_train, yhat_train_mlp)))
print("Multilayer Perceptron: Accuracy for testing is %.2f" % (accuracy_score(y_test, yhat_test_mlp)))