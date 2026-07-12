# Project 3: Autoregressive Decoder-only Transformer from Scratch in PyTorch

A clean, modular PyTorch implementation of a generative, decoder-only Transformer (nanoGPT-style) built block-by-block. 

This project demonstrates a thorough understanding of modern deep learning architectures, attention mechanisms, parallel processing, and positional representations.

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

## 📐 Key Theoretical Concepts

### 1. Scaled Dot-Product Attention
Attention measures the alignment between query vectors ($\mathbf{Q}$) and key vectors ($\mathbf{K}$) to retrieve values ($\mathbf{V}$).

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

- **Scaling Factor ($\frac{1}{\sqrt{d_k}}$)**: For large values of head dimension $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ preserves numerical variance and prevents vanishing gradients.

### 2. Causal Masking (Look-Ahead Mask)
For autoregressive language generation, tokens must only attend to past positions. We implement this by applying a lower-triangular causal mask to the pre-softmax attention scores:

$$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

Adding $-\infty$ before the softmax forces future token weights to evaluate to exactly $0$, preventing the model from cheating by looking ahead.

### 3. Layer Normalization Placement (Pre-LN vs Post-LN)
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

## 💡 Key Takeaways for Recruiters
- **Multi-Head Dimension Routing**: The batch matrix multiplication is optimized by computing Query, Key, and Value matrices simultaneously using `nn.Linear(C, 3*C)` and splitting dimensions, avoiding expensive loop calculations.
- **Top-k Sampling**: Incorporates a top-k filter during generation (`torch.topk`), which zero-out logits for tokens outside the top $K$ probabilities to limit nonsense outputs (truncating the long tail of the probability distribution).
