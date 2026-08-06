# =====================================================================
# Project 1: Multi-Layer Perceptron (MLP) from Scratch in NumPy
# Author: Shabee Ibn Ehtesham
#
# A first-principles implementation of a modular feedforward neural network.
# Built to understand the underlying matrix calculus, forward/backward passes,
# activation functions, and optimizer mechanics without relying on modern deep
# learning frameworks.
# =====================================================================

import numpy as np
import argparse
import sys
import os
import json
import time

# =====================================================================
# 1. Core Neural Network Components (OOP from Scratch)
# =====================================================================

class Layer_Dense:
    """
    A Fully Connected (Dense) Layer.
    Think of this like a group of neurons where every single neuron connects to all the inputs.
    Each connection has a 'Weight' (strength of connection) and each neuron has a 'Bias' (how easy it is to fire).
    """
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights randomly (scale by 0.01 to keep them small at first)
        # Weights represent the strength of each input connection.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Biases are initialized to 0. They control how easy it is to trigger each neuron.
        self.biases = np.zeros((1, n_neurons))

        # Gradients for adjusting the weights and biases during learning (feedback step)
        self.dweights = None
        self.dbiases = None

        # Cache inputs/outputs to use during backpropagation (backward step)
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        # Remember inputs for the backward pass
        self.inputs = inputs
        # Standard matrix math: Output = (Inputs * Weights) + Biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient of loss w.r.t weights (how changes in weights affect the final grade)
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Gradient of loss w.r.t biases (sum of output gradients across the batch)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient of loss w.r.t inputs (to pass backward to the previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    """
    ReLU (Rectified Linear Unit) Activation.
    Acts like a simple on/off gate:
    - If the input is positive, pass it through unchanged: f(x) = x
    - If the input is 0 or negative, turn it off completely: f(x) = 0
    This helps the network learn non-linear (curved) boundaries.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # If the input was negative/zero during the forward pass, block the gradient (no learning allowed here)
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
    """
    Sigmoid Activation.
    Squashes any input number into a neat range between 0 and 1 (like a percentage).
    Ideal for binary classification (Yes/No, 0 or 1).
    """
    def forward(self, inputs):
        self.inputs = inputs
        # np.clip prevents extremely large values from crashing the math (overflow)
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))

    def backward(self, dvalues):
        # Derivative of sigmoid is: sigmoid * (1 - sigmoid)
        # We multiply the incoming gradient by this derivative
        self.dinputs = dvalues * self.output * (1.0 - self.output)


class Activation_Softmax:
    """
    Softmax Activation.
    Takes a set of raw score outputs (logits) and turns them into clean probabilities
    that sum up to 1 (or 100%). Perfect for multi-class classification.
    """
    def forward(self, inputs):
        self.inputs = inputs
        # We subtract the max value from inputs for numerical stability (prevents np.exp from blowing up)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Isolated Softmax derivative is mathematically complex (Jacobian matrix).
        # Softmax + Cross-Entropy are usually combined together to speed up learning.
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss_CategoricalCrossentropy:
    """
    Categorical Cross-Entropy Loss.
    Measures how wrong our predicted probability distribution was compared to the actual answer.
    It penalizes heavily if the model is confident but wrong!
    """
    def forward(self, y_pred, y_true):
        # Clip values slightly to prevent log(0) which is undefined
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Retrieve predictions matching correct target classes
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Loss formula: -log(probability_of_correct_answer)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # Convert simple index labels (like 3) into one-hot lists (like [0,0,0,1,0,...])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Standard derivative of Cross-Entropy loss: -y_true / y_pred, divided by number of samples to average out
        self.dinputs = -y_true / dvalues / samples


class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) Optimizer with Momentum.
    Adjusts weights based on gradient feedback (slight steps down the hill).
    - Decay: Gradually shrinks learning rate over time to avoid overshooting.
    - Momentum: Like rolling a ball down a hill. It gains speed over smooth slopes
      and helps skip over tiny bumps/valleys (local minima).
    """
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        # Decay the learning rate if decay is set
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            # Create momentum arrays if they don't exist yet
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Apply momentum updates: velocity = momentum * old_velocity - learning_rate * gradient
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Standard step: update = -learning_rate * gradient
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Apply the update to parameters
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


# =====================================================================
# 2. Optimized Softmax + Cross Entropy Combined Layer
# =====================================================================

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combines Softmax and Categorical Cross-Entropy.
    During backpropagation, calculating Softmax and Cross-Entropy gradients separately
    is slow and mathematically complex. When combined, the math simplifies to:
    Gradient = (Predicted Probabilities) - (True Labels).
    This is extremely fast and numerically stable!
    """
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot encoded, convert to discrete indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Gradient calculation: p - y
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# =====================================================================
# 3. Model Architecture and Training Loop
# =====================================================================

def run_training(X_train, y_train, X_test, y_test, hidden_size=128, num_classes=66,
                  epochs=10, batch_size=256, lr=0.1, decay=1e-3, momentum=0.9):
    # Model definition: Input -> Dense(hidden_size) -> ReLU -> Dense(num_classes) -> Softmax
    layer1 = Layer_Dense(X_train.shape[1], hidden_size)
    activation1 = Activation_ReLU()
    layer2 = Layer_Dense(hidden_size, num_classes)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_SGD(learning_rate=lr, decay=decay, momentum=momentum)

    print(f"Starting training: {len(X_train)} samples, batch size {batch_size}, {epochs} epochs...")

    num_samples = X_train.shape[0]
    epoch_loss, epoch_acc, test_loss, test_acc = 0.0, 0.0, 0.0, 0.0

    for epoch in range(1, epochs + 1):
        # Shuffle training data at each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0
        epoch_acc = 0
        batches = int(np.ceil(num_samples / batch_size))

        for b in range(batches):
            start = b * batch_size
            end = min(start + batch_size, num_samples)

            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # 1. Forward Pass
            layer1.forward(X_batch)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)

            # 2. Compute Loss
            loss = loss_activation.forward(layer2.output, y_batch)
            predictions = np.argmax(loss_activation.output, axis=1)
            accuracy = np.mean(predictions == y_batch)

            epoch_loss += np.mean(loss) * (end - start)
            epoch_acc += accuracy * (end - start)

            # 3. Backward Pass
            loss_activation.backward(loss_activation.output, y_batch)
            layer2.backward(loss_activation.dinputs)
            activation1.backward(layer2.dinputs)
            layer1.backward(activation1.dinputs)

            # 4. Optimize parameters
            optimizer.pre_update_params()
            optimizer.update_params(layer1)
            optimizer.update_params(layer2)
            optimizer.post_update_params()

        epoch_loss /= num_samples
        epoch_acc /= num_samples

        # Test performance (held-out text the network never trained on)
        layer1.forward(X_test)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        test_loss = np.mean(loss_activation.forward(layer2.output, y_test))
        test_predictions = np.argmax(loss_activation.output, axis=1)
        test_acc = np.mean(test_predictions == y_test)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | LR: {optimizer.current_learning_rate:.4f}")

    return layer1, activation1, layer2, epoch_loss, epoch_acc, test_loss, test_acc


# =====================================================================
# 4. Fixed-Window Text Generation Data Pipeline & CLI Interface
# =====================================================================

# A dedicated boundary/padding symbol that can never collide with a real
# character in the corpus (unlike the names project, where '.' could safely
# double as both padding and boundary - here '.' is a real, frequent
# character in Shakespeare's text, so it needs its own token).
PAD = '\x00'

# Sentence-ending punctuation: generation stops when one of these is sampled.
STOP_CHARS = {'.', '!', '?'}

# The MLP one-hot input grows as block_size * vocab_size, and at the full
# corpus's ~1.1M characters that would mean building a multi-gigabyte NumPy
# array. These caps keep training fast and memory-safe; see the README for
# the honest trade-off this implies.
FULL_RUN_CORPUS_CHARS = 200_000
DEMO_CORPUS_CHARS = 50_000

# Small embedded fallback so --test-run (and any run without data/shakespeare.txt
# present) stays instant and fully offline. This is a REAL verbatim excerpt of
# the fetched corpus (the opening Citizens' scene), not synthetic text.
FALLBACK_TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"
    "All:\nWe know't, we know't.\n\n"
    "First Citizen:\nLet us kill him, and we'll have corn at our own price.\n"
    "Is't a verdict?\n\n"
    "All:\nNo more talking on't; let it be done: away, away!\n\n"
    "Second Citizen:\nOne word, good citizens.\n\n"
    "First Citizen:\nWe are accounted poor citizens, the patricians good.\n"
    "What authority surfeits on would relieve us: if they\n"
    "would yield us but the superfluity, while it were\n"
    "wholesome, we might guess they relieved us humanely;\n"
    "but they think we are too dear: the leanness that\n"
    "afflicts us, the object of our misery, is as an\n"
    "inventory to particularise their abundance; our\n"
    "sufferance is a gain to them Let us revenge this with\n"
    "our pikes, ere we become rakes: for the gods know I\n"
    "speak this in hunger for bread, not in thirst for revenge.\n\n"
    "Second Citizen:\nWould you proceed especially against Caius Marcius?\n\n"
    "All:\nAgainst him first: he's a very dog to the commonalty.\n\n"
    "Second Citizen:\nConsider you what services he has done for his country?\n\n"
    "First Citizen:\nVery well; and could be content to give him good\n"
    "report fort, but that he pays himself with being proud.\n\n"
    "Second Citizen:\nNay, but speak not maliciously.\n\n"
    "First Citizen:\nI say unto you, what he hath done famously, he did\n"
    "it to that end: though soft-conscienced men can be\n"
    "content to say it was for his country he did it to\n"
    "please his mother and to be partly proud; which he\n"
    "is, even till the altitude of his virtue.\n\n"
    "Second Citizen:\nWhat he cannot help in his nature, you account a\n"
    "vice in him. You must in no way say he is covetous."
)


def load_corpus(script_file):
    """
    Loads the shared Shakespeare corpus (see data/shakespeare.txt at the repo
    root). Falls back to a small embedded excerpt if the file isn't available.
    """
    candidates = []
    env_path = os.environ.get("CORPUS_DATASET_PATH")
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(os.path.abspath(script_file))
    candidates.append(os.path.join(here, "..", "..", "data", "shakespeare.txt"))
    candidates.append(os.path.join(os.getcwd(), "data", "shakespeare.txt"))

    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text:
                print(f"[OK] Loaded {len(text)} characters from {path}")
                return text

    print("[INFO] data/shakespeare.txt not found. Falling back to a small embedded excerpt.")
    return FALLBACK_TEXT


def build_vocab(text):
    """
    Derives the vocabulary directly from whatever text is loaded (unlike the
    fixed 27-symbol alphabet used for names, real prose can contain dozens of
    distinct characters - letters, punctuation, whitespace).
    """
    vocab = [PAD] + sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    return vocab, stoi, itos


def encode_context(context_idx, vocab_size):
    """
    Turns a list of character indices (the fixed-size context window) into a
    single flattened one-hot vector: length = block_size * vocab_size.
    """
    onehot = np.zeros(len(context_idx) * vocab_size, dtype=np.float32)
    for i, idx in enumerate(context_idx):
        onehot[i * vocab_size + idx] = 1.0
    return onehot


def build_dataset(text, stoi, block_size=10):
    """
    Builds (X, y) training pairs using a FIXED context window slid
    continuously across the whole corpus: given the previous `block_size`
    characters, predict the next one. The very start of the text is padded
    with the dedicated PAD token. Arrays are built as float32 to keep the
    (block_size * vocab_size)-wide one-hot matrix memory-safe at corpus scale.
    """
    vocab_size = len(stoi)
    pad_idx = stoi[PAD]
    idxs = [stoi.get(ch, pad_idx) for ch in text]
    padded = [pad_idx] * block_size + idxs

    X = np.zeros((len(idxs), block_size * vocab_size), dtype=np.float32)
    Y = np.zeros(len(idxs), dtype='int64')
    for i in range(len(idxs)):
        X[i] = encode_context(padded[i:i + block_size], vocab_size)
        Y[i] = idxs[i]
    return X, Y


def generate_sentence(layer1, activation1, layer2, stoi, itos, block_size=10, temperature=0.8, max_len=200):
    """
    Samples one text continuation character-by-character. The network only
    ever sees the last `block_size` characters - it has no memory beyond that
    fixed window, which is the core limitation this project demonstrates.
    Stops at sentence-ending punctuation, or the safety length cap.
    """
    vocab_size = len(stoi)
    pad_idx = stoi[PAD]
    context = [pad_idx] * block_size
    chars = []

    for _ in range(max_len):
        x = encode_context(context, vocab_size).reshape(1, -1)
        layer1.forward(x)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)

        logits = layer2.output[0] / max(temperature, 1e-6)
        exp_values = np.exp(logits - np.max(logits))
        probs = exp_values / np.sum(exp_values)

        ix = np.random.choice(vocab_size, p=probs)
        ch = itos[ix]
        if ch == PAD:
            break
        chars.append(ch)
        context = context[1:] + [ix]
        if ch in STOP_CHARS:
            break

    return ''.join(chars)


def generate_sentences(layer1, activation1, layer2, stoi, itos, num_sentences=5, block_size=10, temperature=0.8):
    samples = []
    attempts = 0
    while len(samples) < num_sentences and attempts < num_sentences * 5:
        attempts += 1
        sample = generate_sentence(layer1, activation1, layer2, stoi, itos, block_size=block_size, temperature=temperature)
        if sample:
            samples.append(sample)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-window character-level Shakespeare text generator, MLP trained from scratch using NumPy.")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast training + generation sanity check on a tiny embedded text excerpt.")
    parser.add_argument("--demo", action="store_true", help="Fast-but-real training pass on a subset of the full corpus; prints a machine-readable result line for scripts/compare_text_generators.py.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of text samples to generate at the end.")
    parser.add_argument("--block-size", type=int, default=10, help="Fixed context window size (characters of history the model can see).")
    args = parser.parse_args()

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        vocab, stoi, itos = build_vocab(FALLBACK_TEXT)
        X, Y = build_dataset(FALLBACK_TEXT, stoi, block_size=args.block_size)
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = Y[indices[:split]], Y[indices[split:]]

        layer1, activation1, layer2, *_ = run_training(
            X_train, y_train, X_test, y_test,
            hidden_size=32, num_classes=len(vocab), epochs=5, batch_size=32, lr=0.1
        )
        samples = generate_sentences(layer1, activation1, layer2, stoi, itos, num_sentences=2, block_size=args.block_size)
        print(f"  Sample generations: {samples}")
        print("[SUCCESS] NumPy MLP Integration Test Successful!")
        sys.exit(0)

    if args.demo:
        start_time = time.time()
        text = load_corpus(__file__)[:DEMO_CORPUS_CHARS]
        vocab, stoi, itos = build_vocab(text)
        X, Y = build_dataset(text, stoi, block_size=args.block_size)
        indices = np.random.permutation(len(X))
        split = int(0.9 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = Y[indices[:split]], Y[indices[split:]]

        layer1, activation1, layer2, _, _, test_loss, test_acc = run_training(
            X_train, y_train, X_test, y_test,
            hidden_size=128, num_classes=len(vocab), epochs=18, batch_size=256, lr=0.1, decay=1e-3, momentum=0.9
        )
        samples = generate_sentences(layer1, activation1, layer2, stoi, itos, num_sentences=args.num_samples, block_size=args.block_size)
        elapsed = time.time() - start_time

        result = {
            "project": 1,
            "arch": "MLP (fixed 10-char window)",
            "samples": samples,
            "val_loss": round(float(test_loss), 4),
            "val_acc": round(float(test_acc), 4),
            "train_seconds": round(elapsed, 2),
        }
        print("###RESULT_JSON### " + json.dumps(result))
        sys.exit(0)

    # Regular Execution
    print("=== Training NumPy MLP: Fixed-Window Character-Level Shakespeare Text Generator ===")
    text = load_corpus(__file__)[:FULL_RUN_CORPUS_CHARS]
    vocab, stoi, itos = build_vocab(text)
    print(f"Training on {len(text)} characters using a fixed {args.block_size}-character context window (vocab size: {len(vocab)}).")

    X, Y = build_dataset(text, stoi, block_size=args.block_size)
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = Y[indices[:split]], Y[indices[split:]]

    layer1, activation1, layer2, *_ = run_training(
        X_train, y_train, X_test, y_test,
        hidden_size=128, num_classes=len(vocab), epochs=args.epochs, batch_size=256, lr=0.1, decay=1e-3, momentum=0.9
    )

    print(f"\n=== GENERATING {args.num_samples} SAMPLE TEXT CONTINUATIONS ===")
    for sample in generate_sentences(layer1, activation1, layer2, stoi, itos, num_sentences=args.num_samples, block_size=args.block_size):
        print(f"  {sample!r}")

    print("=== Training Completed successfully! ===")
