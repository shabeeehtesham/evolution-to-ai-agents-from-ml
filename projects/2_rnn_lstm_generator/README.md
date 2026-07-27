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

---

## 💡 The Intuitive "Conveyor Belt & Volume Knobs" Analogy

When dealing with sequential data like text, models need memory. Standard RNNs have a short memory, but LSTMs solve this by using two types of memory and three gates:

1. **The Conveyor Belt (Cell State - $c_t$)**: Think of the **Cell State** as a conveyor belt running straight through the sequence of words. It's very easy for information to just ride along the conveyor belt without changing. This is why LSTMs don't suffer from the **Vanishing Gradient** (forgetting) problem—the conveyor belt acts as a superhighway for long-term memory!
2. **The Forget Gate ($f_t$ - Volume Knob 1)**: This is like a volume knob that controls how much of the old memory on the conveyor belt we should discard. If we finish a sentence or a subject, we turn this knob down to clear the memory.
3. **The Input Gate ($i_t$ - Volume Knob 2)**: This knob controls what new information from the current character or word we want to load onto the conveyor belt memory.
4. **The Output Gate ($o_t$ - Volume Knob 3)**: This knob decides which parts of the conveyor belt memory are useful *right now* to predict the next word, outputting it as our short-term **Hidden State** ($h_t$).

---

## ⚡ Temperature-Scaled Generative Sampling: The Creativity Slider

During text generation, the model predicts the probability of the next character. Instead of just picking the highest score, we use a **Temperature ($T$)** parameter to control how creative the model is:

* **Low Temperature (e.g., 0.3)**: Makes the probability distribution peakier. The model is conservative, generating grammatically safe but repetitive sentences.
* **Medium Temperature (e.g., 0.7)**: Balances structure and creativity. Generates fluent, diverse text (ideal).
* **High Temperature (e.g., 1.2)**: Flattens the distribution. The model takes major risks, generating highly creative but often nonsensical words and typos.

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

## 🧪 Technical Notes

A couple of implementation details that caught me off-guard when building this:

- **Detaching Hidden States Between Batches**: When I first wrote the training loop, gradients were accumulating across batch boundaries and blowing up the memory. The fix was detaching the hidden state at each step (`hidden.detach()`) — it keeps the short-term memory for generation while cutting off the computational graph so BPTT doesn't try to backprop infinitely far back.
- **Gradient Clipping**: RNNs are notoriously prone to exploding gradients on long sequences. I added `nn.utils.clip_grad_norm_` after computing gradients and before the optimizer step, which put a hard ceiling on the gradient norm and made training dramatically more stable.

