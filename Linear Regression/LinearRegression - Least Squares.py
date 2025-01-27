### Linear regression from scratch - Least Squares ###

# Imports 
import numpy as np

# Create the LinearRegression class
class LinearRegression():
    
    def __init__(self):
        self.weights = None
        
    # Put the analytical least squares solution inside the fit method
    def fit(self, X, y):
        X_left = np.linalg.inv(X.T.dot(X))
        X_right = X.T.dot(y)
        self.weights = X_left.dot(X_right)
        
    # Use the learned weights from the fit method to make predictions
    def predict(self, X):
        return np.dot(X, self.weights)

    
# Test the above
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y, true_coefs = make_regression(n_samples=200, n_features=1, noise=20, coef=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=28)

model = LinearRegression()

model.fit(X_train, y_train)
y_hat = model.predict(X_test)

weights = model.weights

line = X.dot(weights) 

plt.scatter(X, y, alpha=.5)
plt.plot(X, line, color="red")
plt.show()