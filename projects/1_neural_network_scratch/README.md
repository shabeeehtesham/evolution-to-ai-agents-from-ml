# Project 1: Multi-Layer Perceptron (MLP) from Scratch in NumPy

A modular, object-oriented implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using only Python and NumPy. No PyTorch, no TensorFlow, no scikit-learn.

This project is designed to demonstrate a deep, low-level mathematical understanding of neural networks, backpropagation, and matrix calculus, explained in a simple and intuitive way.

---

## 💡 The Intuitive "Teacher-Student" Analogy

If you are new to neural networks, the math of how they learn can seem complex. Think of the training process as a **Teacher and a Student** in a classroom:

1. **The Student (The Network)**: The student is trying to guess the correct answer (e.g., is this hand-written digit a 3 or an 8?). The student's brain has millions of tiny dial knobs (**weights** and **biases**) that control how they think. At first, all these knobs are set randomly.
2. **The Forward Pass (The Guess)**: The student looks at the input question and turns their knobs to make a guess. Because the knobs are random, the first guess is probably terrible.
3. **The Loss Function (The Grade)**: The teacher grades the student's answer. The **Loss** is a score representing how far off the student's guess was from the truth. A high loss means a bad grade.
4. **The Backward Pass (Backpropagation - The Feedback)**: Instead of just saying "you're wrong," the teacher walks backward through the student's reasoning path. The teacher calculates exactly how changing each dial knob would have helped get a better score. 
5. **The Optimizer (Learning and Adjusting)**: The student adjusts each knob slightly in the direction the teacher suggested. Then, they try a new question. After practicing on thousands of questions, the student turns the knobs to the perfect positions and gets an A+!

---

## 🛠️ Architecture Overview

The codebase is organized using a modular, object-oriented structure where each component is isolated into clean, reusable classes matching the structure of modern frameworks:

1. **`Layer_Dense`**: A fully connected layer holding the weights (strengths of connection) and biases (how easy it is for a neuron to fire).
2. **`Activation_ReLU`**: The activation function that acts as a gate. If a neuron's value is negative, it shuts it off (outputs 0). If it's positive, it passes it through unchanged.
3. **`Activation_Sigmoid`**: An activation function that squashes any input value to a range between `0` and `1`. Perfect for binary classification.
4. **`Activation_Softmax`**: Converts raw network outputs (logits) into clean probabilities that sum up to `100%`.
5. **`Loss_CategoricalCrossentropy`**: Calculates how bad our guesses were. It gives a high penalty for highly confident wrong answers.
6. **`Optimizer_SGD`**: Adjusts the layer weights based on feedback. Implements **momentum** to help step over local obstacles during training.
7. **`Activation_Softmax_Loss_CategoricalCrossentropy`**: An optimized combination class that simplifies and speeds up backpropagation calculations by merging the final activation and loss math.

---

## 📐 Mathematical Derivations & Backprop

During training, we calculate the gradients of the loss function $\mathcal{L}$ with respect to the weights $\mathbf{W}$ and biases $\mathbf{b}$ of each layer using the Chain Rule (our "Teacher's feedback").

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

## 🧪 Technical Notes

A few things that were tricky to get right and are worth calling out:

- **Numerical Stability**: Naive Softmax breaks down with large inputs because `exp()` overflows to infinity. The fix is to subtract the max value from every input before exponentiating — mathematically equivalent, but numerically stable. Same idea for Sigmoid, where I clamp inputs to prevent underflow.
- **Fusing Softmax + Cross-Entropy Backward**: Deriving the combined backward pass manually was the most satisfying part of this project. When you work through the math, the gradient simplifies beautifully to just $\mathbf{p} - \mathbf{y}$, which is far cheaper to compute than doing them separately.
- **Momentum in SGD**: Plain gradient descent zigzags around the loss surface. Adding a velocity term smooths this out — the optimizer builds up speed in directions that consistently reduce loss, which made a noticeable difference in training stability.

