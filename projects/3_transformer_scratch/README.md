# Project 3: Autoregressive Decoder-only Transformer from Scratch in PyTorch

A clean, modular PyTorch implementation of a generative, decoder-only Transformer (nanoGPT-style) built block-by-block. 

This project demonstrates a thorough understanding of modern deep learning architectures, attention mechanisms, parallel processing, and positional representations, explained with simple and intuitive concepts.

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
To run a full training session of the Mini-GPT on Shakespeare quotes:
```bash
python mini_transformer_gpt.py --iters 1500
```

---

## 🧪 Technical Notes

Things I found interesting or non-obvious while building this:

- **Batching Q, K, V Together**: Instead of using three separate `nn.Linear` layers for queries, keys, and values, I compute them all in one shot with a single `nn.Linear(C, 3*C)` and then split. It's cleaner and faster — one matrix multiply instead of three.
- **Top-k Sampling During Generation**: After the model outputs logits, I zero out everything outside the top-K scores before sampling. This stops the model from occasionally picking from the long tail of terrible, low-probability tokens that would otherwise produce gibberish.
