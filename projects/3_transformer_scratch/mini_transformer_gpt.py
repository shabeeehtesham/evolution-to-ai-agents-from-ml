# =====================================================================
# Project 3: Decoder-Only Mini-GPT Transformer from Scratch
# Author: Shabih Ehtesham
#
# A block-by-block implementation of a generative Transformer language model
# using PyTorch. Implements causal multi-head self-attention, positional 
# encodings, feedforward expansions, and autoregressive text generation.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys

# =====================================================================
# 1. Transformer Core Components
# =====================================================================

class CausalSelfAttention(nn.Module):
    """
    Standard scaled dot-product attention module with causal masking.
    Allows tokens to attend only to past positions, enabling autoregressive generation.
    """
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        assert n_embed % n_head == 0, "Embedding size must be divisible by head count"
        
        # Key, Query, Value projections in a single batch matrix multiplication
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        # Output projection
        self.c_proj = nn.Linear(n_embed, n_embed)
        
        self.n_head = n_head
        self.n_embed = n_embed
        
        # Causal mask register (not a parameter, registered as buffer)
        # Lower triangular matrix of ones
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding channels (n_embed)
        
        # 1. Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        
        # 2. Reshape to split heads: (B, T, nh, hs) -> transpose to (B, nh, T, hs)
        # nh = number of heads, hs = head size (C / nh)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 3. Calculate Scaled Dot-Product Attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        
        # Apply causal masking (fill future tokens with -inf before softmax)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Apply softmax to calculate weight scores
        att = F.softmax(att, dim=-1)
        
        # 4. Weighted sum of values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        
        # Re-assemble all heads back to single vector space: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        return self.c_proj(y)


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity (GELU) and output projection.
    Applied position-wise across the sequence.
    """
    def __init__(self, n_embed):
        super().__init__()
        # Feed-forward expansion factor of 4 (standard GPT architecture)
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    A single Transformer Block (Layer).
    Combines Layer Normalization, Causal Self-Attention, FeedForward, and Residual Connections.
    """
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        # Pre-LN design (Layer Normalization applied before blocks)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        # x + residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


# =====================================================================
# 2. Complete Decoder GPT Architecture
# =====================================================================

class MiniTransformerGPT(nn.Module):
    def __init__(self, vocab_size, n_embed=128, n_head=4, n_layer=3, block_size=64):
        super().__init__()
        self.block_size = block_size
        
        # Token Embeddings table
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed),
            wpe = nn.Embedding(block_size, n_embed),
            h = nn.ModuleList([Block(n_embed, n_head, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed)
        ))
        
        # Language Model projection head
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.block_size, f"Sequence length {t} exceeds maximum block size {self.block_size}"
        
        # Generate position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        
        # Look up embedding representations
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embed)
        
        # Sum token and position representations
        x = tok_emb + pos_emb
        
        # Run through stacked Transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # Project hidden states to vocabulary space
        logits = self.lm_head(x) # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten predictions and targets to feed PyTorch CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates text autoregressively.
        """
        for _ in range(max_new_tokens):
            # Crop index context if it exceeds the maximum block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Predict logits
            logits, _ = self(idx_cond)
            
            # Focus on prediction at the last time step
            logits = logits[:, -1, :] / max(temperature, 1e-6) # shape (b, vocab_size)
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to calculate probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token index from probabilities distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append next token index to current sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# =====================================================================
# 3. Text Dataset Class
# =====================================================================

class CharDataset:
    def __init__(self, data_str, block_size):
        self.block_size = block_size
        self.chars = sorted(list(set(data_str)))
        self.vocab_size = len(self.chars)
        
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.data = torch.tensor([self.char2idx[c] for c in data_str], dtype=torch.long)

    def get_batch(self, batch_size):
        # Pick random starting offsets
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        # Target sequence is shifted by 1 index
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y


# =====================================================================
# 4. Training Pipeline
# =====================================================================

def train_transformer(model, dataset, device, max_iters=2000, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training Mini-GPT Transformer on device: {device}...")
    model.train()
    
    for i in range(1, max_iters + 1):
        # Get random training batch
        xb, yb = dataset.get_batch(batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass & loss evaluation
        logits, loss = model(xb, yb)
        
        # Optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Log evaluation metrics periodically
        if i == 1 or i % 250 == 0 or i == max_iters:
            print(f"Iteration {i}/{max_iters} | Loss: {loss.item():.4f}")
            
            # Print sample generation
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device) # starting with null token (usually index 0)
            generated_ids = model.generate(context, max_new_tokens=40, temperature=0.7, top_k=5)[0].tolist()
            sample_str = "".join([dataset.idx2char[idx] for idx in generated_ids])
            print(f"  Generated Sample: {repr(sample_str)}")
            model.train()

    return model


# =====================================================================
# 5. CLI Execution
# =====================================================================

TINY_CORPUS = """
JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

ROMEO:
Shall I hear more, or shall I speak at this?

JULIET:
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive Decoder-only Transformer built from scratch.")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast integration test on synthetic Shakespeare text.")
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        # Build small corpus
        test_corpus = "The quick brown fox jumps over the lazy dog. Attention is all you need for sequence models."
        dataset = CharDataset(test_corpus, block_size=8)
        
        # Instantiate tiny model
        model = MiniTransformerGPT(
            vocab_size=dataset.vocab_size, 
            n_embed=16, 
            n_head=2, 
            n_layer=1, 
            block_size=8
        ).to(device)
        
        train_transformer(model, dataset, device, max_iters=10, batch_size=4, lr=1e-3)
        print("[SUCCESS] Integration Test Successful!")
        sys.exit(0)

    # Full Run
    print("=== Training Decoder-only Transformer ===")
    corpus = TINY_CORPUS * 5 # Expand corpus slightly to ensure enough sample diversity
    dataset = CharDataset(corpus, block_size=64)
    
    print(f"Dataset Vocabulary Size: {dataset.vocab_size} unique characters")
    print(f"Corpus Length: {len(corpus)} characters")
    
    # Model parameters: 3 layers, 4 heads, 128 embedding size, block size 64
    model = MiniTransformerGPT(
        vocab_size=dataset.vocab_size, 
        n_embed=128, 
        n_head=4, 
        n_layer=3, 
        block_size=64
    ).to(device)
    
    # Train
    train_transformer(model, dataset, device, max_iters=args.iters, batch_size=32, lr=1e-3)
    
    # Final Generation Demo
    model.eval()
    print("\n=== GENERATING SHAKESPEARE (T=0.5, top_k=5) ===")
    context = torch.tensor([[dataset.char2idx['J']]], dtype=torch.long, device=device) # Priming with J
    gen_ids = model.generate(context, max_new_tokens=150, temperature=0.5, top_k=5)[0].tolist()
    print("".join([dataset.idx2char[idx] for idx in gen_ids]))
    
    print("\n=== GENERATING SHAKESPEARE (T=0.9, top_k=10) ===")
    context = torch.tensor([[dataset.char2idx['R']]], dtype=torch.long, device=device) # Priming with R
    gen_ids = model.generate(context, max_new_tokens=150, temperature=0.9, top_k=10)[0].tolist()
    print("".join([dataset.idx2char[idx] for idx in gen_ids]))

    print("\n=== Transformer training finished successfully! ===")
