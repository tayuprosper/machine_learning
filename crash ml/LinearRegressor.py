import numpy as np
import matplotlib.pyplot as plt


class LinearRegressor:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        N, D = X.shape

        # y = y.reshape(-1, 1)
        self.weights = np.zeros((D, 1))
        self.bias = 0

        # List to track the loss changes
        self.loss_history = []

        for i in range(self.n_iters):

            # Forward Pass
            y_predicted = (X @ self.weights) + self.bias
            loss_error = y_predicted - y

            # MSE 
            mse = (1/N) * np.sum(np.square(loss_error))

            self.loss_history.append(mse)

            # Backward Pass (Use @ for Matrix Multiplication)
            # Shape: (features, samples) @ (samples, 1) -> (features, 1)
            dW = (1 / N) * (X.T @ loss_error)
            # Gradient for bias is the average of errors
            db = (1 / N) * np.sum(loss_error)

            # 5. Update Parameters
            self.weights -= self.lr * dW
            self.bias -= self.lr * db
            
            # Print progress every 100 steps
            if i % 100 == 0:
                mse = np.mean(np.square(loss_error))
                print(f"Iter {i}: Cost {mse:.4f}")

    def predict(self, X):
        # Simple linear transformation
        return (X @ self.weights) + self.bias
   
       


# 1. Create a simple linear dataset: y = 2x + 1
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[3], [5], [7], [9]], dtype=float)

# 2. Initialize and train
model = LinearRegressor(lr=0.01, n_iters=500)
model.fit(X, y)

# 3. Test
test_x = np.array([[5], [6]])
predictions = model.predict(test_x)
print(f"Predictions for 5 and 6: {predictions.flatten()}") 

# visualise the loss
plt.plot(range(model.n_iters), model.loss_history)
plt.xlabel('Iterations')
plt.ylabel('Mean Square Error')
plt.title('Learning curve: Convergence of Gradient Descent')
plt.show()