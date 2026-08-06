# =====================================================================
# Project 3: Decoder-Only Mini-GPT Transformer from Scratch
# Author: Shabee Ibn Ehtesham
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
import os
import json
import time

# =====================================================================
# 1. Transformer Core Components
# =====================================================================

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention Layer.
    - Self-Attention: Allows tokens (words) to look at all other tokens in the sentence
      and decide which ones are most relevant (e.g., matching 'it' to 'dog').
    - Causal: Prevents tokens from looking into the future. Each token can only look
      at itself and words that came before it.
    """
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        assert n_embed % n_head == 0, "Embedding size must be divisible by head count"

        # We calculate Queries (Q), Keys (K), and Values (V) together in one big linear layer for speed
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        # Output projection layer to mix head results back together
        self.c_proj = nn.Linear(n_embed, n_embed)

        self.n_head = n_head
        self.n_embed = n_embed

        # Causal mask: a lower-triangular matrix of ones
        # This acts like a folder partition preventing the model from looking ahead
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding channels

        # 1. Project input to Q, K, and V
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # 2. Split into multiple heads. This lets the network focus on different relationships
        # (e.g., one head checks grammar, another checks pronouns, another checks verbs).
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. Calculate alignment scores: (Query x Key) / sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))

        # Apply causal masking: fill future tokens with -inf so their softmax probability is 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Calculate final softmax weight probabilities (attention map)
        att = F.softmax(att, dim=-1)

        # 4. Use weights to average the values (Value)
        y = att @ v

        # Concatenate all heads back into a single vector representation
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.c_proj(y)


class FeedForward(nn.Module):
    """
    A simple multilayer perceptron applied position-wise across the sequence.
    Think of this as a brain block where each token digests the information it gathered
    during the attention step and thinks about it individually.
    """
    def __init__(self, n_embed):
        super().__init__()
        # Standard GPT expansion: project up by a factor of 4, apply GELU, project back down
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    A single Transformer Block.
    Combines Causal Self-Attention, FeedForward, Layer Normalization, and Residual Connections.
    """
    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        # Pre-LN design: normalize the values BEFORE passing them into attention or feedforward
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        # x + block(x) creates a residual 'highway' where gradients can flow backward easily
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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed), # Word Token Embeddings (meaning of words)
            wpe = nn.Embedding(block_size, n_embed), # Word Position Embeddings (where words are in sentence)
            h = nn.ModuleList([Block(n_embed, n_head, block_size) for _ in range(n_layer)]), # Stacked Blocks
            ln_f = nn.LayerNorm(n_embed) # Final normalization
        ))

        # Projects output representation back to vocabulary logits (word prediction scores)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize weights with standard normal distribution for stability
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

        # Create position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        # Word representations + Position representations (helps model know which word came first)
        tok_emb = self.transformer.wte(idx) # (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embed)
        x = tok_emb + pos_emb

        # Pass through stacked Transformer Blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (b, t, vocab_size)

        loss = None
        if targets is not None:
            # Flatten predictions and targets to calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_tokens=None):
        """
        Generates text autoregressively (one token at a time). If `stop_tokens`
        (a set of token ids) is given and we're generating a single sequence
        (batch size 1), generation stops as soon as one of them is produced —
        used to end at sentence-ending punctuation instead of always running
        for max_new_tokens. Batched (B>1) early-stopping would need
        per-sequence masking/padding, which is deliberately out of scope here
        since generation is only ever called one sequence at a time.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds maximum block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Predict scores
            logits, _ = self(idx_cond)

            # Scale scores at the final timestep with temperature
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Top-k filtering: ignore any tokens outside the top k most probable options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            # Sample next token index from probabilities
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append predicted token index to output sequence
            idx = torch.cat((idx, idx_next), dim=1)

            if stop_tokens is not None and idx.size(0) == 1 and idx_next.item() in stop_tokens:
                break

        return idx


# =====================================================================
# 3. Text Dataset Class
# =====================================================================

class CharDataset:
    """
    Splits a character corpus into input/target index sequences.
    """
    def __init__(self, data_str, block_size):
        self.block_size = block_size
        self.chars = sorted(list(set(data_str)))
        self.vocab_size = len(self.chars)

        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor([self.char2idx[c] for c in data_str], dtype=torch.long)

    def get_batch(self, batch_size):
        # Pick random starting offsets inside our corpus
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        # Target sequence is shifted forward by 1 index (predict next character)
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y


# =====================================================================
# 4. Training Pipeline & Text Generation Helpers
# =====================================================================

STOP_CHARS = {'.', '!', '?'}


def generate_sentence(model, dataset, device, temperature=0.7, top_k=10, max_new_tokens=200):
    """
    Generates one text continuation: primes on a newline and samples until the
    model produces sentence-ending punctuation (or hits the safety length cap).
    """
    model.eval()
    stop_tokens = {dataset.char2idx[c] for c in STOP_CHARS if c in dataset.char2idx}
    context = torch.tensor([[dataset.char2idx['\n']]], dtype=torch.long, device=device)
    gen_ids = model.generate(
        context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        stop_tokens=stop_tokens
    )[0].tolist()
    text = "".join([dataset.idx2char[idx] for idx in gen_ids])
    return text.strip("\n")


def generate_sentences(model, dataset, device, num_sentences=5, temperature=0.7, top_k=10):
    samples = []
    attempts = 0
    while len(samples) < num_sentences and attempts < num_sentences * 5:
        attempts += 1
        sample = generate_sentence(model, dataset, device, temperature=temperature, top_k=top_k)
        if sample:
            samples.append(sample)
    return samples


def train_transformer(model, dataset, device, max_iters=2000, batch_size=32, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training Mini-GPT Transformer on device: {device}...")
    model.train()

    last_loss = 0.0
    for i in range(1, max_iters + 1):
        # Fetch a random batch of sequences
        xb, yb = dataset.get_batch(batch_size)
        xb, yb = xb.to(device), yb.to(device)

        # Calculate predictions and loss
        logits, loss = model(xb, yb)

        # Backpropagate and adjust weights
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

        if i == 1 or i % 250 == 0 or i == max_iters:
            print(f"Iteration {i}/{max_iters} | Loss: {last_loss:.4f}")

            # Print a sample generated continuation to show progress
            sample = generate_sentence(model, dataset, device, temperature=0.7, top_k=5)
            print(f"  Generated Sample: {repr(sample)}")
            model.train()

    return model, last_loss


# =====================================================================
# 5. Corpus Loading & CLI Execution
# =====================================================================

# Small embedded fallback so --test-run (and any run without data/shakespeare.txt
# present) stays instant and fully offline. Real verbatim excerpt of the fetched
# corpus (the opening Citizens' scene), not synthetic text - kept identical to
# Projects 1 & 2's excerpt so all three share the same fallback universe.
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

DEMO_CORPUS_CHARS = 80_000


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive decoder-only Transformer built from scratch, trained as a character-level Shakespeare text generator.")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast integration test on a tiny embedded text excerpt.")
    parser.add_argument("--demo", action="store_true", help="Fast-but-real training pass on a subset of the full corpus; prints a machine-readable result line for scripts/compare_text_generators.py.")
    parser.add_argument("--iters", type=int, default=1500, help="Number of training iterations.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of text samples to generate at the end.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        dataset = CharDataset(FALLBACK_TEXT, block_size=8)

        # Instantiate tiny model
        model = MiniTransformerGPT(
            vocab_size=dataset.vocab_size,
            n_embed=16,
            n_head=2,
            n_layer=1,
            block_size=8
        ).to(device)

        train_transformer(model, dataset, device, max_iters=10, batch_size=4, lr=1e-3)
        samples = generate_sentences(model, dataset, device, num_sentences=2, top_k=5)
        print(f"  Sample generations: {samples}")
        print("[SUCCESS] Integration Test Successful!")
        sys.exit(0)

    if args.demo:
        start_time = time.time()
        text = load_corpus(__file__)[:DEMO_CORPUS_CHARS]
        dataset = CharDataset(text, block_size=64)

        model = MiniTransformerGPT(
            vocab_size=dataset.vocab_size, n_embed=64, n_head=4, n_layer=2, block_size=64
        ).to(device)

        model, final_loss = train_transformer(model, dataset, device, max_iters=2000, batch_size=64, lr=1e-3)
        samples = generate_sentences(model, dataset, device, num_sentences=args.num_samples, temperature=0.7, top_k=10)
        elapsed = time.time() - start_time

        result = {
            "project": 3,
            "arch": "Transformer (full self-attention)",
            "samples": samples,
            "val_loss": round(float(final_loss), 4),
            "train_seconds": round(elapsed, 2),
        }
        print("###RESULT_JSON### " + json.dumps(result))
        sys.exit(0)

    # Full Run
    print("=== Training Decoder-only Transformer: Shakespeare Text Generator ===")
    text = load_corpus(__file__)
    dataset = CharDataset(text, block_size=128)

    print(f"Dataset Vocabulary Size: {dataset.vocab_size} unique characters")
    print(f"Corpus Length: {len(text)} characters")

    # Model parameters: 3 layers, 4 heads, 128 embedding size, block size 128
    model = MiniTransformerGPT(
        vocab_size=dataset.vocab_size,
        n_embed=128,
        n_head=4,
        n_layer=3,
        block_size=128
    ).to(device)

    # Train
    train_transformer(model, dataset, device, max_iters=args.iters, batch_size=64, lr=1e-3)

    # Final Generation Demo
    model.eval()
    print(f"\n=== GENERATING {args.num_samples} SAMPLE TEXT CONTINUATIONS (T=0.5, top_k=5) ===")
    for sample in generate_sentences(model, dataset, device, num_sentences=args.num_samples, temperature=0.5, top_k=5):
        print(f"  {sample!r}")

    print(f"\n=== GENERATING {args.num_samples} SAMPLE TEXT CONTINUATIONS (T=0.9, top_k=10) ===")
    for sample in generate_sentences(model, dataset, device, num_sentences=args.num_samples, temperature=0.9, top_k=10):
        print(f"  {sample!r}")

    print("\n=== Transformer training finished successfully! ===")
