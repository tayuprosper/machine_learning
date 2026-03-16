""" logistic regression is still going to use the linear
 regreession class but instead of predicting continues, it will squish output to be
  either 0 or 1 using  the sigmoid function so new prediction is yhat = sig(XW + b)
    where sig(z) = 1 / (1 + e^-z). Also, the loss function wil not longer be mse because with
  sigmoid, the "loss valley" is no longer a U-shape; it becomes "non-convex" (bumpy)
  . Gradient descent would get stuck in a "fake bottom" so we rather use Binary Cross-Entropy ( Log loss)
  which now is loss = - (1/N) * sum[(ylog(yhat) + (1 -y) log(1- yhat)] and  dloss/dw = (1/N) * X.T * (yhat - y)"""
  
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

class LogisticRegressor:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []  
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
      
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)  # Reshape y to be a column vector
        
        # initialization of weights and bias
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.loss_history = []  
        
        for _ in range(self.n_iters):
            linear_model = (X @ self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            #compute loss and gradients
            # add tiny epsilon to avoid log(0)
            loss = - (1/n_samples) * np.sum(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.loss_history.append(loss)
            
            # Backward pass (Gradient descent)
            dw = (1/n_samples) * (X.T @ (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # upate weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X, threshold=0.5):
        linear_model = (X @ self.weights) + self.bias
        probabilities = self._sigmoid(linear_model)
        return (probabilities >= threshold).astype(int)
        
def evaluate_clinical_model(y_true, y_pred):
    #ensure flat array for easy computation
    
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    
    #-------------------------------------------------------------
    # 1. CCONFUSION METRIX ( Targeting 0 as the 'positive' class )
    #-------------------------------------------------------------
    
    # TRue Positive: They have cancer (0) AND we predict cancer (0)
    # sum all instances wher the patient had cancer and we predicted correctly
    TP = np.sum((y_true == 0) & (y_pred == 0))
    
    # True Negetive:  How many people were health (1) and predicted health (1)
    
    TN = np.sum((y_pred == 1) & (y_true == 1))
    
    # False Negetve: How many people had cancer (0) but we predicted healthy (1)
    
    FN = np.sum((y_pred == 1) & (y_true == 0))
    
    # False positive: How many people were healthy (1) and we predicted cancer (0)
    
    FP = np.sum((y_pred == 1) & (y_true == 0))
    
    print("\n--- Clinical Evaluation ---")
    print(f"Confusion Matrix:")
    print(f"                 Predicted Cancer (0) | Predicted Healthy (1)")
    print(f"Actual Cancer (0) |        {TP}           |        {FN}          ")
    print(f"Actual Healthy(1) |        {FP}           |        {TN}          ")
    
    #----------------------------------------------------------
    # METRICS
    # add epsilon (1e-9) to prevent divsion by zero
    #----------------------------------------------------------
    
    # Presision: out of all predicte cancers, how many were real (total positive)
    # total positive = TP + FN
    
    precision = TP / (TP + FP + 1e-9)
    
    # Recall: Out of all actual cancers, how many did we catch
    recall = TP / (TP + FN + 1e-9)
    
    # F1 score harmonic mean of precision and Recall
    
    f1_score = 2 * ((precision * recall )/(precision + recall))
    
    print(f"\nPrecision (Cancer): {precision * 100:.2f}%")
    print(f"Recall (Cancer)   : {recall * 100:.2f}%")
    print(f"F1-Score          : {f1_score * 100:.2f}%")
    
    return precision, recall, f1_score
  
  

    
          
      
if __name__ == "__main__":
    print("Loading dataset...")
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    
    # Feature scaling (standardization)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    
    # 4. Train Your Model
    print("Training model...")
    # Notice we can use a much higher learning rate (0.1) because the data is scaled!
    model = LogisticRegressor(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate on unseen data
    predictions = model.predict(X_test)
    
    # Flatten predictions to (N,) to match y_test shape for easy comparison
    predictions = predictions.flatten()
    
    # Calculate Accuracy: (Total Correct) / (Total Samples)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Model Accuracy on unseen data: {accuracy * 100:.2f}%")

    # 6. Plot the Loss Curve
    plt.plot(model.loss_history)
    plt.title("Logistic Regression Loss Curve (Binary Cross-Entropy)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    
    evaluate_clinical_model(y_test,predictions)
    
