# Project 3: Autoregressive Decoder-only Transformer from Scratch in PyTorch

A clean, modular PyTorch implementation of a generative, decoder-only Transformer (nanoGPT-style) built block-by-block. 

---

## 🎯 What Are We Actually Building?

**The problem:** ChatGPT, Gemini, Llama, Claude — they all share the same core architecture: the **Transformer**. But what exactly is inside one?

**What this project builds:** A miniature GPT from scratch — the exact same architecture that powers modern LLMs, just much smaller. It trains on the same Shakespeare corpus as Projects 1 and 2, and generates text one character at a time — but this time, every character can attend directly to every other character generated so far, not just recent ones or a single hidden-state summary:

```
Prompt:    "\n"
Generated: "LARTIUS:\nThere's that that place hason."
           "He stame more than you are lend,\nAnd by the corn companed agains..."
```

**What's the big upgrade over the LSTM from Project 2?**

The LSTM processed text **one character at a time, left-to-right** — like reading a sentence with your finger covering everything to the right. By the time you get to the end, your memory of the beginning has faded.

The Transformer throws that limitation away entirely. It processes **all words in the sentence simultaneously** and lets every word look directly at every other word to decide what's relevant. This is called **Self-Attention** — and it's why modern LLMs can reference something said 10,000 words ago without forgetting it.

This is the architecture that changed everything:
- **2017**: Google publishes *"Attention Is All You Need"* — Transformers are born
- **2018**: OpenAI releases GPT-1 based on this architecture
- **2022**: ChatGPT (GPT-3.5) goes viral worldwide
- **This project**: We build the core decoder-only version from scratch, same family as GPT

---

## ⚡ Architecture Breakdown

```
        Input Token Indices (B, T)
                    │
            ┌───────┴───────┐
      Token Embedding  Position Embedding
            └───────┬───────┘
                    ▼
               Sum (B, T, C)
                    │
          ┌──────────┴──────────┐
          │  Transformer Block  │  x N Layers
          └──────────┬──────────┘
                    ▼
             Layer Normalization
                    │
            Linear Projection (LM Head)
                    │
                    ▼
           Output Logits (B, T, V)
```

---

## 💡 The Intuitive "Classroom Study Room & Exam Partitions" Analogy

Transformers completely removed the recurrent loop of RNNs/LSTMs, allowing them to process all words in a sentence at the same time. They do this using two key mechanisms:

### 1. Self-Attention (Classroom Study Room)
Imagine a group of students in a study room working on a group assignment. When a student needs to solve a specific problem, they don't look at all the pages of a textbook sequentially. Instead, they look at all other students and ask: **"Who has the most relevant notes for my current question?"**
* The student looking for notes broadcasts a **Query** ($\mathbf{Q}$).
* Each of the other students has a card describing their notes, which is a **Key** ($\mathbf{K}$).
* By multiplying Query $\times$ Key, we find which student has the best notes.
* The student then retrieves the actual notes, which is the **Value** ($\mathbf{V}$).

### 2. Causal Masking (Exam Partitions)
When generating text, we want the model to predict the next word without cheating by looking at future words.
* Imagine a student taking a test. To prevent cheating, the teacher places a **folder partition (causal mask)** on the desk.
* The student can only look to their left (the past answers they've written) but cannot see to their right (the future answers that haven't been written yet).
* We implement this by adding a value of negative infinity ($-\infty$) to the scores of future positions. When Softmax is applied, the probability of looking at future words becomes exactly $0$.

---

## 📐 Key Theoretical Concepts

### 1. Scaled Dot-Product Attention
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

* **Scaling Factor ($\frac{1}{\sqrt{d_k}}$)**: When the head size $d_k$ is large, dot products grow very large, pushing the softmax function into flat regions with tiny gradients. Dividing by $\sqrt{d_k}$ keeps the math stable.

### 2. Layer Normalization Placement (Pre-LN vs Post-LN)
We implement a **Pre-LN** design, where Layer Normalization is applied *before* the Multi-Head Attention and Feedforward blocks. This creates a clean gradient highway (residual stream) directly from the input to the final layers, enabling stable training of deeper architectures.

---

## 🚀 How to Run

### Integration Test
Verify the Causal Self-Attention matrix calculations, block size cropping, and gradient steps:
```bash
python mini_transformer_gpt.py --test-run
```

### Full Training
Trains the Mini-GPT on the full ~1.1M-character Shakespeare corpus ([data/shakespeare.txt](../../data/shakespeare.txt) at the repo root):
```bash
python mini_transformer_gpt.py --iters 1500
```

**What you'll actually see** (real observed output from a fast 2,000-iteration demo run on an 80,000-character slice — `--demo` mode, used by [scripts/compare_text_generators.py](../../scripts/compare_text_generators.py)):
```
Iteration 1/2000    | Loss: 4.1382
Iteration 1000/2000 | Loss: 1.5334
Iteration 2000/2000 | Loss: 1.2401
```
Real generated text continuations: `"MENENIUS:\nYou most the made of these to you are stal have and frightat."`, `"The offf, that you may have some of the people and it brought thee,\nThey in nabord valial it."` — real character names (`MENENIUS`), real words, and roughly plausible sentence shape, though not fully grammatical. The loss (1.2401) is measurably lower than both the MLP (1.9896) and the LSTM (1.7435) trained on the same corpus, the concrete payoff of full self-attention over a fixed window or a single recurrent hidden state.

---

## 🧪 Technical Notes

Things I found interesting or non-obvious while building this:

- **Batching Q, K, V Together**: Instead of using three separate `nn.Linear` layers for queries, keys, and values, I compute them all in one shot with a single `nn.Linear(C, 3*C)` and then split. It's cleaner and faster — one matrix multiply instead of three.
- **Top-k Sampling During Generation**: After the model outputs logits, I zero out everything outside the top-K scores before sampling. This stops the model from occasionally picking from the long tail of terrible, low-probability tokens that would otherwise produce gibberish.
