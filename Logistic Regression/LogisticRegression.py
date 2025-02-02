### Logistic regression from scratch - Gradient Descent ###

# Begin with imports
import numpy as np

# Create the LogisticRegression class
class LogisticRegression():
    def __init__(self, lr=0.01, max_iter=100, threshold=.5):
        self.lr = lr
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.losses = []
        self.threshold = threshold
        
    # Define the sigmoid function
    def sigmoid(self, x):
        A = 1 / (1 + np.exp(-x))
        return A
        
    # Define the loss function - binary cross entropy
    def loss_fn(self, y_true, y_pred):
        epsilon = np.repeat(1e-8, y_true.shape[0])  # Include some noise
        y_1 = y_true * np.log(y_pred + epsilon)
        y_2 = (1 - y_true) * np.log((1 - y_pred) + epsilon)
        return -np.mean(y_1 + y_2)
    
    # Method to obtain the probabilistic outputs
    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(z)
        return A
    
    # Define the fit method
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialise the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Perform graident descent
        for _ in range(self.max_iter):
            A = self.feed_forward(X)
            self.losses.append(self.loss_fn(y,A))
            da_dz = A - y   # Sigmoid derivative
            dw = (1/n_samples) * np.dot(da_dz, X)   # Weights gradient
            db = (1/n_samples) * np.sum(da_dz)  # Bias gradient
            
            # Make updates to parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    # Define predict method
    def predict(self, X):
        y_pred = self.feed_forward(X)
        y_pred_clf = [1 if i > self.threshold else 0 for i in y_pred]
        return np.array(y_pred_clf)
    
### Test the custom class ###
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Create synthetic data
X, y = make_classification(n_samples=500, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)

# Create an instance of the logisitic regression class
clf = LogisticRegression()

# Train the model and make predictions
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_preds)
print("Test accuracy: {0:.3f}".format(accuracy))

conf_matrix = confusion_matrix(y_test, y_preds)
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=[0,1])
conf_matrix_display.plot()
plt.show()