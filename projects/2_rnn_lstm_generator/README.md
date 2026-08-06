# Project 2: Sequence Modeling with RNN & LSTM in PyTorch

A customizable PyTorch implementation of a character-level language model showcasing sequence-to-sequence processing and tokenization. 

This project implements two sequential architectures: a **standard Recurrent Neural Network (RNN)** and a **Long Short-Term Memory (LSTM)** network, enabling direct comparison of sequence propagation dynamics.

---

## 🎯 What Are We Actually Building?

**The problem:** Project 1's MLP predicts the next character of Shakespeare's text from a **fixed 10-character window** — it's structurally incapable of using anything further back. That's fine for short local patterns, but the moment something depends on what character is speaking or what happened a full sentence earlier, the MLP simply cannot see it.

**What this project builds:** The same character-level text generation task, but the model now reads the text **one character at a time while keeping a running memory** (a *hidden state*) of everything it has seen so far — not just the last 10 characters:

```
Input (character by character):  C → O → R → I → O → L → A → N → U → S
Hidden state updates:              h1 → h2 → h3 → h4 → h5 → h6 → h7 → h8 → h9 → h10

Generated (after training):
  "CORIOLANUS:\nIn the grouther."
  "Morces:\nWhe canst we refore."
```

**Why does sequence order matter here?**
The MLP from Project 1 saw each 10-character window independently — every prediction started from scratch with no memory of anything earlier. Language doesn't work like that: whether a line is likely to end, or which character is speaking, often depends on much more context than the last 10 characters. To capture that, you need **memory** — the ability to carry context forward from arbitrarily far back.

This project shows two approaches to building that memory:
- **RNN** — simple recurrent memory, but it forgets things from too far back
- **LSTM** — gated memory that can selectively remember or forget, solving the long-range dependency problem

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
Run a quick, 2-epoch integration test on a short text excerpt to verify model compilation, sequence batching, and generation logic:
```bash
# Test LSTM
python char_rnn_generator.py --model lstm --test-run

# Test RNN
python char_rnn_generator.py --model rnn --test-run
```

### Full Training
Trains on the full ~1.1M-character Shakespeare corpus ([data/shakespeare.txt](../../data/shakespeare.txt) at the repo root):
```bash
# Train LSTM (Default)
python char_rnn_generator.py --model lstm --epochs 15

# Train RNN
python char_rnn_generator.py --model rnn --epochs 15
```

**What you'll actually see** (real observed output from a fast 10-epoch demo run on an 80,000-character slice — `--demo` mode, used by [scripts/compare_text_generators.py](../../scripts/compare_text_generators.py)):
```
Epoch 1/10  | Loss: 2.7853
Epoch 5/10  | Loss: 2.0713
Epoch 10/10 | Loss: 1.7435
```
Real generated text continuations: `"SICINIUS:\nNo, have no thent beless aill I thee."`, `"Prest they vouth to your fore lormer,\nThe carmins."` — recognizable character-name formatting and real words (`"have"`, `"no"`, `"your"`, `"thee"`) show up, though full sentences don't yet hold together. The loss (1.7435) is already meaningfully lower than Project 1's MLP (1.9896) trained on the same corpus, a direct, measured consequence of having access to the whole sequence instead of just 10 characters.

---

## 🧪 Technical Notes

A couple of implementation details that caught me off-guard when building this:

- **Detaching Hidden States Between Batches**: When I first wrote the training loop, gradients were accumulating across batch boundaries and blowing up the memory. The fix was detaching the hidden state at each step (`hidden.detach()`) — it keeps the short-term memory for generation while cutting off the computational graph so BPTT doesn't try to backprop infinitely far back.
- **Gradient Clipping**: RNNs are notoriously prone to exploding gradients on long sequences. I added `nn.utils.clip_grad_norm_` after computing gradients and before the optimizer step, which put a hard ceiling on the gradient norm and made training dramatically more stable.

