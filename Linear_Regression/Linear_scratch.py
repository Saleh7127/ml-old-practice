import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, lambda_=0.1, use_sgd=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_ = lambda_  # Regularization strength (for Ridge)
        self.use_sgd = use_sgd  # If True, use Stochastic Gradient Descent (SGD)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        if self.use_sgd:
            # Use Stochastic Gradient Descent (SGD)
            for _ in range(self.n_iters):
                for i in range(n_samples):
                    random_index = np.random.randint(0, n_samples)
                    X_i = X[random_index:random_index+1]
                    y_i = y[random_index:random_index+1]
                    y_pred = np.dot(X_i, self.weights) + self.bias

                    dw = (1 / n_samples) * np.dot(X_i.T, (y_pred - y_i)) + (self.lambda_ / n_samples) * self.weights
                    db = (1 / n_samples) * np.sum(y_pred - y_i)

                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
        else:
            # Use Batch Gradient Descent
            for _ in range(self.n_iters):
                y_pred = np.dot(X, self.weights) + self.bias
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.lambda_ / n_samples) * self.weights
                db = (1 / n_samples) * np.sum(y_pred - y)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
