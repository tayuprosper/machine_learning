import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class SoftmaxRegressor:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _softmax(self, z):
        z_shifted = z - np.max(z, keepdims=True, axis=1)
        
        exps = np.exp(z_shifted)
        return exps/np.sum(exps, axis=1, keepdims=True)


    def _one_hot(self, y, n_classes):
        y_flat = y.flatten().astype(int)
        
        return np.eye(n_classes)[y_flat]
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # determine the number of unique classes
        n_classes = len(np.unique(y))
        
        # convert y to hot_one encoded matrix (N,K)
        
        Y_one_hot = self._one_hot(y, n_classes)
        
        # initialis the weights and biases
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1,n_classes))
        
        
        # training loop
        for i in range(self.n_iters):
            linear_model = X @ self.weights + self.bias
            
            y_predicted = self._softmax(linear_model)
            
            # loss calcuation
            loss = -(1/n_samples)*np.sum(Y_one_hot * np.log(y_predicted +1e-9))
            self.loss_history.append(loss)
            
            
            error = y_predicted - Y_one_hot
            
            dW = (1/n_samples) * (X.T @ error)
            db = (1/n_samples) * np.sum(error, axis=0, keepdims=True)
            
            # Update Parameters
            self.weights -= self.lr * dW
            self.bias -= self.lr * db
            
    def predict(self,X):
        linear_model = X @ self.weights + self.bias
        probabilities = self._softmax(linear_model)
        
        return np.argmax(probabilities, axis=1)
        
        
if __name__ == "__main__":
    print("Loading Iris dataset...")
    # 1. Load Data (3 classes: Setosa (0), Versicolor (1), Virginica (2))
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 2. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Feature Scaling (Standardization)
    # Senior rule: Calculate mean and std ONLY on the training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # 4. Train Your Custom Model
    print("Training model...")
    model = SoftmaxRegressor(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate on Unseen Data
    predictions = model.predict(X_test)
    
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"\nModel Accuracy on unseen data: {accuracy * 100:.2f}%")

    # 6. Build a 3x3 Confusion Matrix from scratch
    K = len(np.unique(y))
    confusion_matrix = np.zeros((K, K), dtype=int)
    
    for true_label, pred_label in zip(y_test, predictions):
        confusion_matrix[true_label][pred_label] += 1

    print("\n--- Confusion Matrix ---")
    print("                 Predicted")
    print("              0      1      2")
    print(f"Actual 0  |  {confusion_matrix[0][0]:>3}    {confusion_matrix[0][1]:>3}    {confusion_matrix[0][2]:>3}")
    print(f"Actual 1  |  {confusion_matrix[1][0]:>3}    {confusion_matrix[1][1]:>3}    {confusion_matrix[1][2]:>3}")
    print(f"Actual 2  |  {confusion_matrix[2][0]:>3}    {confusion_matrix[2][1]:>3}    {confusion_matrix[2][2]:>3}")
    print("\n(0: Setosa, 1: Versicolor, 2: Virginica)")

    # 7. Plot the Loss Curve
    plt.plot(model.loss_history)
    plt.title("Softmax Regression Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Categorical Cross-Entropy Loss")
    plt.show()