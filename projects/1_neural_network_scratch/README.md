# Project 1: Multi-Layer Perceptron (MLP) from Scratch in NumPy

A modular, object-oriented implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using only Python and NumPy. No PyTorch, no TensorFlow, no scikit-learn.

---

## 🎯 What Are We Actually Building?

**The problem:** Given a fixed, small window of recent characters, what character comes next? This is the simplest possible version of "predict the next character" — a Bengio-2003 / Andrej Karpathy "makemore"-style model that looks at exactly the **last 10 characters** and predicts the 11th.

Trained on Shakespeare's plays (the real ~1.1M-character "tiny-shakespeare" corpus), it learns which character combinations tend to follow which — real words and speaker-label patterns emerge from pure statistics, with no grammar rules programmed in.

**What this project builds:**
- A network that takes the last 10 characters (one-hot encoded, 660 numbers total: 10 slots × 66 possible characters) as input
- Passes those values through one hidden layer of neurons that learn character-combination patterns
- Outputs 66 probability scores — one for each character in Shakespeare's actual vocabulary (letters, punctuation, spaces, newlines) plus a boundary token marking "start of text"
- Sampling repeatedly from these predictions, one character at a time, generates a text continuation

```
Context (last 10 chars)          Prediction
┌────────────────────────┐
│  "vernal, se"           │        [0.01, 0.00, ..., 0.58 ('r'), ...]
│  (one-hot, 660 dims)    │  ──►
└────────────────────────┘        Predicted next char: 'r'
```

**The catch — and the whole point of this project:** the network can *never* see further back than 10 characters. It has no memory of anything earlier in the text — no idea what character is speaking, what scene it's in, or what was said two sentences ago. That fixed ceiling is exactly what Project 2 (RNN/LSTM) removes by introducing a hidden state that carries information across the entire sequence.

**A real, honest limitation worth stating upfront:** a one-hot input of `block_size × vocab_size` numbers grows fast. At full corpus scale (1.1M characters) that would mean building a multi-gigabyte NumPy array — genuinely impractical for a "from scratch on a laptop" demo. This project trains on a capped slice of the corpus (200,000 characters for a full run, 50,000 for the fast `--demo` mode) and uses `float32` instead of NumPy's default `float64` to keep memory in check. That cap is itself an honest architectural cost of the one-hot/NumPy approach, not something Projects 2 or 3 need (their PyTorch embedding layers stay compact regardless of corpus size).

---

## 💡 The Intuitive "Teacher-Student" Analogy

If you are new to neural networks, the math of how they learn can seem complex. Think of the training process as a **Teacher and a Student** in a classroom:

1. **The Student (The Network)**: The student is trying to guess the correct answer (e.g., given the last 10 characters of Shakespeare's text, is the next character an 'r' or a 'z'?). The student's brain has millions of tiny dial knobs (**weights** and **biases**) that control how they think. At first, all these knobs are set randomly.
2. **The Forward Pass (The Guess)**: The student looks at the input question and turns their knobs to make a guess. Because the knobs are random, the first guess is probably terrible.
3. **The Loss Function (The Grade)**: The teacher grades the student's answer. The **Loss** is a score representing how far off the student's guess was from the truth. A high loss means a bad grade.
4. **The Backward Pass (Backpropagation - The Feedback)**: Instead of just saying "you're wrong," the teacher walks backward through the student's reasoning path. The teacher calculates exactly how changing each dial knob would have helped get a better score. 
5. **The Optimizer (Learning and Adjusting)**: The student adjusts each knob slightly in the direction the teacher suggested. Then, they try a new question. After practicing on thousands of questions, the student turns the knobs to the perfect positions and gets an A+!

---

## 🛠️ Architecture Overview

Here is how the text-generation pipeline maps to code. Data flows left to right:

```
10-char context (660)  →  Layer_Dense(660→128)  →  ReLU  →  Layer_Dense(128→66)  →  Softmax  →  Predicted next char
```

Each component and why it exists:

1. **`Layer_Dense`** — The core learning unit. 660 one-hot inputs (10 character slots × 66 possible characters) connect to 128 hidden neurons, each with its own **weight** ("how much does this specific letter-in-this-position matter?") and **bias** ("how easily does this neuron fire?"). After training, these weights encode things like "a speaker-label line ending in ':' is often followed by a newline".

2. **`Activation_ReLU`** — Placed after the first Dense layer. Without this, stacking two Dense layers is mathematically identical to one — you can only draw straight lines through the data. ReLU introduces the ability to learn non-linear character-combination patterns.

3. **`Activation_Sigmoid`** — Not used in the main pipeline, but available for binary problems (e.g., "is this character punctuation or not?"). Squashes output to a `0–1` probability.

4. **`Activation_Softmax`** — The final step. Takes the 66 raw output scores and converts them to a valid probability distribution: all 66 values sum to exactly `1.0`. So the network might output `[0.01, 0.00, ..., 0.58, ...]` — a 58% confidence the next character is `'r'`.

5. **`Loss_CategoricalCrossentropy`** — The grader. If the network said "58% chance the next character is 'r'" and it really was 'r', the loss is low. If it was actually 'z', the loss is high. It specifically punishes confident wrong answers.

6. **`Optimizer_SGD`** — The learning engine. After the grader scores each guess, the optimizer nudges every single weight and bias slightly in the direction that would have lowered the score. With **momentum**, it builds up speed on long slopes instead of zigzagging.

7. **`Activation_Softmax_Loss_CategoricalCrossentropy`** — A fused version of Softmax + Loss combined into one class. The math simplifies the backward pass from a complex Jacobian down to just `predicted - truth`, making training faster.

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

### Integration Test
Runs a fast training + generation sanity check on a tiny embedded text excerpt:
```bash
python neural_network_scratch.py --test-run
```

### Full Training
Trains on a 200,000-character slice of the Shakespeare corpus ([data/shakespeare.txt](../../data/shakespeare.txt) at the repo root) and generates sample text:
```bash
python neural_network_scratch.py --epochs 20
```

You can also open [neural_network_scratch.ipynb](neural_network_scratch.ipynb) in Jupyter, Colab, or VS Code and run all cells.

**What you'll actually see during training** (real observed output from an 18-epoch demo run on a 50,000-character slice):
```
Epoch 1/18  | Train Loss: 3.2410 | Train Acc: 0.1800 | Test Loss: 2.8299 | Test Acc: 0.2508 | LR: 0.0851
Epoch 9/18  | Train Loss: 2.0034 | Train Acc: 0.4263 | Test Loss: 2.0999 | Test Acc: 0.4008 | LR: 0.0387
Epoch 18/18 | Train Loss: 1.7976 | Train Acc: 0.4751 | Test Loss: 1.9896 | Test Acc: 0.4226 | LR: 0.0240
```
And real generated text continuations afterward: `"it that wav."`, `"ill cithtius."`, `"en, matties, ward it the breshere prom itis."` — some real words show up (`"it"`, `"that"`, `"ward"`), embedded in mostly non-words. That's an honest, direct consequence of the model only ever seeing 10 characters of context: enough to learn plausible letter clusters, not enough to reliably spell real words or track sentence structure. Compare this to Projects 2 and 3's samples on the exact same corpus to see the difference more context makes.

If `data/shakespeare.txt` isn't found (e.g. running this file in isolation), it falls back to a small embedded text excerpt so the script still runs end-to-end offline.

---

## 🧪 Technical Notes

A few things that were tricky to get right and are worth calling out:

- **Numerical Stability**: Naive Softmax breaks down with large inputs because `exp()` overflows to infinity. The fix is to subtract the max value from every input before exponentiating — mathematically equivalent, but numerically stable. Same idea for Sigmoid, where I clamp inputs to prevent underflow.
- **Fusing Softmax + Cross-Entropy Backward**: Deriving the combined backward pass manually was the most satisfying part of this project. When you work through the math, the gradient simplifies beautifully to just $\mathbf{p} - \mathbf{y}$, which is far cheaper to compute than doing them separately.
- **Momentum in SGD**: Plain gradient descent zigzags around the loss surface. Adding a velocity term smooths this out — the optimizer builds up speed in directions that consistently reduce loss, which made a noticeable difference in training stability.

