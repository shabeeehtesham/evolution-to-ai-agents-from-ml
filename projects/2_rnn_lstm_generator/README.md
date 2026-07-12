# Project 2: Sequence Modeling with RNN & LSTM in PyTorch

A customizable PyTorch implementation of a character-level language model showcasing sequence-to-sequence processing and tokenization. 

This project implements two sequential architectures: a **standard Recurrent Neural Network (RNN)** and a **Long Short-Term Memory (LSTM)** network, enabling direct comparison of sequence propagation dynamics.

---

## 📐 Theory & Mathematical Differences

```
Standard RNN Step:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

LSTM Cell Step (Gated Memory):
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   (Forget Gate)
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   (Input Gate)
c~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)     (Candidate Memory Cell)
c_t = f_t * c_{t-1} + i_t * c~_t           (Updated Memory Cell)
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   (Output Gate)
h_t = o_t * tanh(c_t)                       (Updated Hidden State)
```

### 1. The Vanishing Gradient Problem (RNNs)
In a standard RNN, the hidden state at step $t$ depends directly on multiplying the previous hidden state by the transition matrix $\mathbf{W}_{hh}$. Backpropagating over $T$ steps involves multiplying by $\mathbf{W}_{hh}^T$. If the eigenvalues of $\mathbf{W}_{hh}$ are slightly less than 1, the gradient exponentially shrinks to zero, preventing the network from learning long-term dependencies.

### 2. Gated Solvers (LSTMs)
LSTMs introduce a **Cell State** ($c_t$) that runs straight down the sequence with only minor linear interactions. Gates (forget $f_t$, input $i_t$, and output $o_t$) regulate addition or removal of information. This additive nature of the cell state updates allows gradients to flow backwards through time without exponential decay.

---

## ⚡ Temperature-Scaled Generative Sampling

During text generation, the model outputs raw unnormalized prediction scores (logits). Rather than taking the absolute argmax (which results in repetitive text) or random sampling (which results in gibberish), we scale logits using a **Temperature parameter ($T$)** before applying Softmax:

$$P(x_i) = \frac{e^{\frac{z_i}{T}}}{\sum_{j} e^{\frac{z_j}{T}}}$$

- **Low Temperature (e.g., 0.3)**: Makes the probability distribution peakier (highly confident). The model is conservative, generating grammatically safe but repetitive sentences.
- **Medium Temperature (e.g., 0.7)**: Balances structure and creativity. Generates fluent, diverse text.
- **High Temperature (e.g., 1.2)**: Flattens the distribution. The model takes risks, generating highly creative but often nonsensical words.

---

## 🚀 How to Run

### Integration Test
Run a quick, 2-epoch integration test on a short sequence to verify the model compilation, sequence batching, and text generation logic:
```bash
# Test LSTM
python char_rnn_generator.py --model lstm --test-run

# Test RNN
python char_rnn_generator.py --model rnn --test-run
```

### Full Training
To run a full training session:
```bash
# Train LSTM (Default)
python char_rnn_generator.py --model lstm --epochs 15

# Train RNN
python char_rnn_generator.py --model rnn --epochs 15
```

---

## 💡 Key Takeaways for Recruiters
- **Hidden State Detachment**: In sequential training, hidden states must be detached (`hidden.detach()`) at each sequence step boundary to block backpropagation through infinite time (Backpropagation Through Time limit), saving GPU memory.
- **Gradient Clipping**: Outlines gradient norm clipping (`nn.utils.clip_grad_norm_`) which prevents exploding gradients when backpropagating through long sequences.
