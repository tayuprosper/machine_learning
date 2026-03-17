# Machine Learning from First Principles

This repository contains custom, from-scratch implementations of fundamental machine learning algorithms. The goal of this project is to build a deep mathematical and engineering intuition for how models learn, optimizing directly via Gradient Descent using strictly Python and NumPy.

No high-level ML frameworks (like Scikit-Learn or PyTorch) are used for the core algorithms, forward/backward passes, or metric calculations. 

## 📁 Repository Structure

* `linear/`
  * Contains the `LinearRegressor` class.
  * **Concepts covered:** Vectorized forward pass, Mean Squared Error (MSE) loss, and Gradient Descent optimization for continuous targets.
* `logistic/`
  * Contains the `LogisticRegressor` class.
  * **Concepts covered:** Sigmoid activation, Binary Cross-Entropy (Log Loss), feature scaling (Standardization), and classification thresholds.
  * Includes a real-world evaluation script tested on the Breast Cancer Wisconsin dataset, featuring custom implementations of a Confusion Matrix, Precision, Recall, and F1-Score metrics.

## 🛠️ Tech Stack

* **Python 3.x**
* **NumPy:** For all vectorized matrix operations and linear algebra.
* **Matplotlib:** For visualizing loss curves and convergence plateaus.
* **Scikit-Learn (Utility only):** Strictly used for fetching datasets and performing standard train/test splits.

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name

    Create and activate a virtual environment (recommended):
    Bash

    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    Install the required dependencies:
    Bash

    pip install numpy matplotlib scikit-learn

🧠 Why Build From Scratch?

In modern MLOps, engineers rely on PyTorch or TensorFlow for production. However, understanding the underlying matrix calculus, the chain rule in backpropagation, and the exact mechanics of loss functions is what allows an engineer to debug vanishing gradients, optimize custom architectures, and transition from a model user to a model builder.
🗺️ Roadmap

    [x] Multiple Linear Regression

    [x] Logistic Regression (Binary Classification)

    [x] Softmax Regression (Multi-class Classification) - In Progress

    [ ] Multi-Layer Perceptron (MLP)


# Note:
## This readme was generated with an LLM. Factcheck important info

