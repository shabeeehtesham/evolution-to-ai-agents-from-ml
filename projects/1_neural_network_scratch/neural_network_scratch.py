# =====================================================================
# Project 1: Multi-Layer Perceptron (MLP) from Scratch in NumPy
# Author: Shabih Ehtesham
#
# A first-principles implementation of a modular feedforward neural network.
# Built to understand the underlying matrix calculus, forward/backward passes,
# activation functions, and optimizer mechanics without relying on modern deep
# learning frameworks.
# =====================================================================

import numpy as np
import argparse
import sys

# =====================================================================
# 1. Core Neural Network Components (OOP from Scratch)
# =====================================================================

class Layer_Dense:
    """
    Standard fully connected layer.
    """
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with Xavier/He-like scaling and biases as zeros
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        # Gradients for optimization
        self.dweights = None
        self.dbiases = None
        
        # Inputs cache for backpropagation
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on inputs to pass to the previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    """
    Rectified Linear Unit Activation.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero out gradients where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
    """
    Sigmoid Activation.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500))) # clip to prevent overflow

    def backward(self, dvalues):
        # Derivative of sigmoid: s * (1 - s)
        self.dinputs = dvalues * self.output * (1.0 - self.output)


class Activation_Softmax:
    """
    Softmax Activation for multi-class classification.
    """
    def forward(self, inputs):
        self.inputs = inputs
        # Subtract max for numerical stability (preventing overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Note: Backpropagation for Softmax + CrossEntropy is optimized together,
        # but here we implement the isolated mathematical backward pass for completeness.
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss_CategoricalCrossentropy:
    """
    Categorical Cross Entropy Loss.
    """
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Categorical labels (1D index array) or One-hot encoded vectors
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # Convert to one-hot if labels are 1D
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient: -y_true / y_pred / samples
        self.dinputs = -y_true / dvalues / samples


class Optimizer_SGD:
    """
    Stochastic Gradient Descent Optimizer with Momentum and Learning Rate Decay.
    """
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        # Decaying learning rate
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If momentum is used
        if self.momentum:
            # If layer does not have momentum arrays, create them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Weight updates with momentum
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            # Bias updates with momentum
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Standard SGD
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Apply updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


# =====================================================================
# 2. Optimized Softmax + Cross Entropy Combined Layer
# =====================================================================

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combines Softmax activation and Categorical Cross-Entropy Loss
    to simplify and speed up backpropagation gradients.
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

def run_training(X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=0.1, decay=1e-3, momentum=0.9):
    # Model definition: Input -> Dense(128) -> ReLU -> Dense(10) -> Softmax
    layer1 = Layer_Dense(X_train.shape[1], 128)
    activation1 = Activation_ReLU()
    layer2 = Layer_Dense(128, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_SGD(learning_rate=lr, decay=decay, momentum=momentum)

    print(f"Starting training: {len(X_train)} samples, batch size {batch_size}, {epochs} epochs...")
    
    num_samples = X_train.shape[0]

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
        
        # Test performance
        layer1.forward(X_test)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        test_loss = np.mean(loss_activation.forward(layer2.output, y_test))
        test_predictions = np.argmax(loss_activation.output, axis=1)
        test_acc = np.mean(test_predictions == y_test)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | LR: {optimizer.current_learning_rate:.4f}")

    return layer1, activation1, layer2


# =====================================================================
# 4. Data Generation & CLI Interface
# =====================================================================

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic spiral-like classification data.
    Perfect for demonstrating classification capability without hefty downloads.
    """
    classes = 10
    dimensionality = 64 # Let's make it 8x8 image size equivalent
    X = np.zeros((num_samples * classes, dimensionality))
    y = np.zeros(num_samples * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(num_samples*class_number, num_samples*(class_number+1))
        r = np.linspace(0.0, 1, num_samples) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, num_samples) + np.random.randn(num_samples)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t), np.random.randn(num_samples, dimensionality - 2)] # pad with random noise
        y[ix] = class_number
    return X, y


def load_mnist_torch():
    """
    Attempts to load real MNIST dataset using torchvision.
    If torchvision is not available, falls back to generating high-quality synthetic data.
    """
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        X_train = trainset.data.numpy().reshape(-1, 28*28) / 255.0
        y_train = trainset.targets.numpy()
        X_test = testset.data.numpy().reshape(-1, 28*28) / 255.0
        y_test = testset.targets.numpy()
        
        print("[OK] Loaded MNIST Dataset successfully via torchvision.")
        return X_train, y_train, X_test, y_test
    except ImportError:
        print("[INFO] torchvision not available or failed to load. Falling back to synthetic datasets...")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Layer Perceptron trained from scratch using NumPy.")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast training sequence with synthetic data for testing.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        # Generate small dataset
        X, y = generate_synthetic_data(num_samples=50)
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]
        
        run_training(X_train, y_train, X_test, y_test, epochs=3, batch_size=32, lr=0.1)
        print("[SUCCESS] NumPy MLP Integration Test Successful!")
        sys.exit(0)

    # Regular Execution
    print("=== Training NumPy MLP ===")
    mnist_data = load_mnist_torch()
    if mnist_data is not None:
        X_train, y_train, X_test, y_test = mnist_data
    else:
        print("Running full training loop on synthetic dataset (10,000 samples, 64 features, 10 classes)")
        X, y = generate_synthetic_data(num_samples=1000) # 1000 per class * 10 classes = 10,000 samples
        indices = np.random.permutation(len(X))
        split = int(0.85 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]

    run_training(X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=128, lr=0.1, decay=1e-3, momentum=0.9)
    print("=== Training Completed successfully! ===")
