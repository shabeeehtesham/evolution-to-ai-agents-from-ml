# Project 1: Multi-Layer Perceptron (MLP) from Scratch in NumPy

A modular, object-oriented implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using only Python and NumPy. No PyTorch, no TensorFlow, no scikit-learn.

This project is designed to demonstrate a deep, low-level mathematical understanding of neural networks, backpropagation, and matrix calculus.

---

## 🛠️ Architecture Overview

The codebase is organized using a modular, object-oriented structure where each component is isolated into clean, reusable classes matching the structure of modern frameworks:

1. **`Layer_Dense`**: Fully connected layer holding the weight matrix $\mathbf{W}$ and bias vector $\mathbf{b}$.
2. **`Activation_ReLU`**: Rectified Linear Unit activation function ($f(x) = \max(0, x)$).
3. **`Activation_Sigmoid`**: Sigmoid activation function ($f(x) = \frac{1}{1 + e^{-x}}$).
4. **`Activation_Softmax`**: Softmax activation for generating probability distributions over multiple classes.
5. **`Loss_CategoricalCrossentropy`**: Calculates loss for multi-class classification.
6. **`Optimizer_SGD`**: Stochastic Gradient Descent with support for learning rate decay and momentum.
7. **`Activation_Softmax_Loss_CategoricalCrossentropy`**: Optimized combination class that mathematically consolidates the Softmax backward pass and Cross-Entropy derivative into a highly efficient step: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}$.

---

## 📐 Mathematical Derivations & Backprop

During training, we calculate the gradients of the loss function $\mathcal{L}$ with respect to the weights $\mathbf{W}$ and biases $\mathbf{b}$ of each layer using the Chain Rule.

For a single dense layer computing $\mathbf{z} = \mathbf{X}\mathbf{W} + \mathbf{b}$, given the incoming gradient of the loss with respect to the output layer, $\frac{\partial \mathcal{L}}{\partial \mathbf{z}}$ (denoted as `dvalues` in code):

### 1. Gradient of Loss w.r.t. Weights
The weight updates depend on the activations from the previous layer ($\mathbf{X}$):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{X}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$$

### 2. Gradient of Loss w.r.t. Biases
The bias update is simply the sum of gradients across the batch dimension:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$$

### 3. Gradient of Loss w.r.t. Inputs
The gradient propagated back to the prior layer is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \mathbf{W}^T$$

---

## 🚀 How to Run

Open the Jupyter notebook [neural_network_scratch.ipynb](neural_network_scratch.ipynb) in your Jupyter environment, Google Colab, or VS Code:

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook projects/1_neural_network_scratch/neural_network_scratch.ipynb
   ```
2. **VS Code**: Simply open the `neural_network_scratch.ipynb` file and click **Run All** in the top menu.

The notebook executes a training loop on a generated high-dimensional spiral dataset. If `torchvision` is installed, it will automatically attempt to load the real MNIST digits dataset.

---


## 💡 Key Takeaways for Recruiters
- **Numerical Stability**: Softmax uses input shift subtraction ($\mathbf{x} - \max(\mathbf{x})$) to prevent floating-point overflow. Sigmoid uses value clipping to prevent underflow.
- **Optimized Backward Passes**: Demonstrates the mathematical simplification of merging Softmax + Categorical Cross-Entropy, reducing backpropagation computational complexity.
- **Momentum-based SGD**: Implements velocity tracking ($v = \beta v - \eta \nabla_\theta \mathcal{L}$) which accelerates convergence by damping oscillations.
