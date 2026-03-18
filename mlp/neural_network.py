import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, n_features, n_hidden, n_classes, lr= 0.01, n_iters= 1000):
        self.lr = lr
        self.n_iters = n_iters
        
    
        # layer 1 weights and baises (Input -> Hidden )
        self.W1 = np.random.randn(n_features, n_hidden) *0.01
        self.b1 = np.zeros((1, n_hidden))
        
        # Layer 2 Weights and biasis (Hidden -> Output)
        self.W2 = np.random.randn(n_hidden, n_classes) * 0.01
        self.bias1 = np.zeros((1, n_hidden))
        self.bias2 = np.zeros((1, n_classes))
        
        self.lost_history = []
        
    def _relu(self, Z):
        return np.maximum(0,Z)
    
    def _softmax(self, z):
        z_shifted = z - np.max(z,axis=1,keepdims=True)
        exps = np.exp(z_shifted)
        
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def _one_hot(self, y, n_classes):
        y_flat = y.flatten().astype(int)
        
        return np.eye(n_classes)[y_flat]
    
    def _relu_derivative(self, Z):
        # 1. (Z > 0) creates a matrix of True and False
        # 2. .astype(float) instantly converts True to 1.0 and False to 0.0
        return (Z > 0).astype(float)
    
    def _forward_pass(self, X):
        """
        Your Mission: Implement the Forward Pass
        X shape: (N, n_features)
        """
        # --- Layer 1 (Hidden Layer) ---
        # 1. Calculate linear combination for layer 1 (Z1)
        Z1 = X @ self.W1 + self.bias1
        
        # 2. Apply ReLU activation to get the hidden state (A1)
        A1 = self._relu(Z1)
        
        # --- Layer 2 (Output Layer) ---
        # 3. Calculate linear combination for layer 2 (Z2), using A1 as the input!
        Z2 = A1 @ self.W2 + self.bias2
        
        # 4. Apply Softmax activation to get the final probabilities (A2)
        A2 = self._softmax(Z2)
        
        # We must return all these variables because we need them for Backpropagation later
        return Z1, A1, Z2, A2    
          
    def _backward_pass(self,  X, Y, Z1, A1, Z2, A2):
        n_samples, _ = X.shape
        
        # loss for layer 2
        dZ2 = A2 - Y
        
        # gradient for layer two
        dW2 = (1/n_samples) * (A1.T @ dZ2)
        # bias for layer 2
        db2 = (1/n_samples) * np.sum(dZ2, keepdims=True, axis=0)
        
        # Push erro backwards
        # loss for layer one 
        # dZ1 calculates the error that goes to layer 2
        dZ1 = (dZ2 @ self.W2.T) * self._relu_derivative(Z1)
        
        # gradient for layer 1
        dW1 = (1/n_samples) * (X.T @ dZ1)
        # bias for layer 1
        db1 = (1/n_samples) * np.sum(dZ1, axis=0, keepdims=True)
        
        
        return dW1, db1, dW2, db2
        
       
    def fit(self, X, y):
        n_samples, _ = X.shape
        n_classes = len(np.unique(y))
        
        # convert integer lables to One_hot
        Y_one_hot = self._one_hot(y, n_classes)
        
        for i in range(self.n_iters):
            # forward pass
            Z1, A1, Z2, A2 = self._forward_pass(X)
            
            # calculating and storing loss (Categorical cross entropy loss)
            loss = -(1/n_samples) * np.sum(Y_one_hot * np.log(A2 + 1e-9))
            self.lost_history.append(loss)
            
            # backward pass
            dW1, db1, dW2, db2 = self._backward_pass(X, Y_one_hot, Z1,A1,Z2,A2)
            
            # update parameters ( Gradient Descent)
            self.W1 -= self.lr * dW1
            self.bias1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.bias2 -= self.lr * db2
            
    def predict(self, X):
        # we only care about the final output (A2) predictionss
        _,_,_,A2 = self._forward_pass(X)
        return np.argmax(A2, axis=1)
        
def plot_decision_boundary(X, y, model):
    """Generates a color map to visualize the model's decision boundary."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict the class for every point on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and the original data points
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("MLP Decision Boundary")
    plt.show()
    
if __name__ == "__main__":
    print("Generating the Donut Dataset...")
    # 1. Generate non-linear data (factor=0.5 means the inner circle is half the size)
    X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train Your Neural Network
    # We have 2 features (x/y coordinates), 10 hidden neurons, and 2 output classes
    print("Training the Multi-Layer Perceptron...")
    model = NeuralNetwork(n_features=2, n_hidden=10, n_classes=2, lr=0.1, n_iters=2000)
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"\nModel Accuracy on unseen data: {accuracy * 100:.2f}%")

    # 5. Visualize the Magic
    plot_decision_boundary(X, y, model)