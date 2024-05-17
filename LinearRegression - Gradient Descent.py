### Linear regression from scratch ###

# Begin by creating a linear regression class using numpy
import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        
    # The fit method needs to contain gradient descent to find the best weights and biases
    def fit(self, X, y):
        n_rows, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.max_iter):   
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (2/n_rows) * np.dot(X.T, (y_pred - y))
            db = (2/n_rows) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    # The predict method takes the current values of the weights and biases and predicts the on the input X        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    # Returns all coefficients of the model
    def coefs(self):
        return self.weights, self.bias
    
    
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y, true_coefs = make_regression(n_samples=200, n_features=1, noise=10, coef=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

# Create an instance of the linear regression class above with the default parameters
model = LinearRegression()

# Fit the model and make predictions
model.fit(X, y)
y_preds = model.predict(X_train)

weights, bias = model.coefs()

line = weights*X + bias

plt.scatter(X, y, alpha=.5)
plt.plot(X, line, color="red")
plt.show()