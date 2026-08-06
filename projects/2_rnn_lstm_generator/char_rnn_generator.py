# =====================================================================
# Project 2: Character-Level RNN & LSTM Text Generator from Scratch
# Author: Shabee Ibn Ehtesham
#
# Comparing custom-configured Recurrent Neural Network (RNN) and Long
# Short-Term Memory (LSTM) layers in PyTorch to understand hidden states,
# sequence prediction loops, and how gating mitigates vanishing gradients.
# =====================================================================

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
import json
import time

# =====================================================================
# 1. Model Implementations in PyTorch
# =====================================================================

class CharRNN(nn.Module):
    """
    A Character-level Recurrent Neural Network (RNN).
    Shows how processing a sequence works by updating a 'Hidden State' (memory) at each step.
    Think of it like reading a sentence word-by-word while keeping a mental summary.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Word/Char Embedding layer (turns character IDs into vectors representing meanings)
        self.encoder = nn.Embedding(vocab_size, embed_size)

        # Standard RNN layer
        # batch_first=True makes input shape: (batch_size, sequence_length, features)
        self.rnn = nn.RNN(embed_size, hidden_size, n_layers, batch_first=True)

        # Decoder maps our final hidden state to scores (logits) for each character in our vocabulary
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # 1. Convert character index list to vectors
        embed = self.encoder(x)
        # 2. Pass vectors and old memory (hidden state) to get outputs and updated memory
        out, hidden = self.rnn(embed, hidden)

        # 3. Flatten outputs so we can project them to character prediction scores
        out_reshaped = out.contiguous().view(-1, self.hidden_size)
        decoded = self.decoder(out_reshaped)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state (memory) to all zeros
        # Shape: (n_layers, batch_size, hidden_size)
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)


class CharLSTM(nn.Module):
    """
    A Character-level Long Short-Term Memory (LSTM) network.
    Uses 'Gates' to manage long-term and short-term memory, solving the forgetting (vanishing gradient) issue.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        self.encoder = nn.Embedding(vocab_size, embed_size)

        # LSTM layer containing forget, input, and output gates
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)

        # Decoder layer to project hidden states to character predictions
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # 1. Convert characters to vectors
        embed = self.encoder(x)
        # 2. LSTM takes (hidden_state, cell_state) as memory and updates them both
        out, hidden = self.lstm(embed, hidden)

        # 3. Project output states to vocabulary logits
        out_reshaped = out.contiguous().view(-1, self.hidden_size)
        decoded = self.decoder(out_reshaped)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        # LSTM needs two memory states: short-term (h0) and the conveyor-belt long-term memory (c0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# =====================================================================
# 2. Text Preprocessing & Dataset Helper
# =====================================================================

class TextDataset:
    """
    Handles converting text to lists of numbers and creating training batches.
    """
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len

        # 1. Find all unique characters (our dictionary/vocabulary)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # 2. Map characters to numbers and vice-versa
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

        # 3. Convert the entire training text into a long list of numbers
        self.data = np.array([self.char2idx[c] for c in text], dtype=np.int64)

    def get_batches(self, batch_size):
        # Calculate how many characters fit into one batch
        num_characters_per_batch = batch_size * self.seq_len
        num_batches = len(self.data) // num_characters_per_batch

        if num_batches == 0:
            raise ValueError(f"Corpus too small for batch size {batch_size} and sequence length {self.seq_len}")

        # Chop off any extra characters at the end that don't fit perfectly
        truncated_data = self.data[:num_batches * num_characters_per_batch]
        # Reshape data into rows representing parallel batch streams
        reshaped_data = truncated_data.reshape(batch_size, -1)

        for n in range(0, reshaped_data.shape[1] - self.seq_len, self.seq_len):
            # x is the sequence of inputs
            x = reshaped_data[:, n:n+self.seq_len]
            # y is the target sequence (the exact same text but shifted forward by 1 character)
            y = reshaped_data[:, n+1:n+self.seq_len+1]
            yield torch.from_numpy(x), torch.from_numpy(y)


# =====================================================================
# 3. Training & Text Generation Functions
# =====================================================================

STOP_CHARS = {'.', '!', '?'}


def generate_text(model, dataset, device, start_str="\n", predict_len=200, temperature=0.8, stop_chars=STOP_CHARS):
    """
    Generates new text character-by-character using temperature scaling for creativity.
    Stops early once a character in `stop_chars` is produced, instead of always
    running for the full predict_len.
    """
    model.eval()
    chars = [c for c in start_str]
    hidden = model.init_hidden(1, device)

    # 1. Prime the model's memory (hidden state) with the seed string
    for char in start_str[:-1]:
        x = torch.tensor([[dataset.char2idx[char]]]).to(device)
        _, hidden = model(x, hidden)

    # 2. Generate the next characters one-by-one
    curr_char = start_str[-1]
    for _ in range(predict_len):
        x = torch.tensor([[dataset.char2idx[curr_char]]]).to(device)
        logits, hidden = model(x, hidden)

        # 3. Scale scores with temperature (creativity control slider)
        logits = logits[-1] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()

        # 4. Sample a character index based on the calculated probabilities
        char_idx = np.random.choice(len(probs), p=probs)
        curr_char = dataset.idx2char[char_idx]
        chars.append(curr_char)

        if curr_char in stop_chars:
            break

    return "".join(chars)


def generate_sentence(model, dataset, device, temperature=0.8, max_len=200):
    """
    Generates one text continuation: primes on a newline and samples until the
    model produces sentence-ending punctuation (or hits the safety length cap).
    Since the training corpus is a continuous stream of real dialogue (not one
    self-contained example per line, the way names were), this is a
    "continuation" of wherever the model's internal state happens to start,
    not a guaranteed fresh sentence beginning.
    """
    raw = generate_text(model, dataset, device, start_str="\n", predict_len=max_len, temperature=temperature)
    return raw.strip("\n")


def generate_sentences(model, dataset, device, num_sentences=5, temperature=0.8):
    samples = []
    attempts = 0
    while len(samples) < num_sentences and attempts < num_sentences * 5:
        attempts += 1
        sample = generate_sentence(model, dataset, device, temperature=temperature)
        if sample:
            samples.append(sample)
    return samples


def train_model(model, dataset, device, epochs=10, batch_size=32, lr=0.002, seq_len=50, model_name="LSTM"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {model_name} Model on device: {device}...")

    avg_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        hidden = model.init_hidden(batch_size, device)
        total_loss = 0
        batch_count = 0

        for x, y in dataset.get_batches(batch_size):
            batch_count += 1
            x, y = x.to(device), y.to(device)

            # Detach hidden states to prevent PyTorch from backpropagating back through previous batches.
            # This saves memory and keeps training computationally stable.
            if isinstance(hidden, tuple): # LSTM (hidden_state, cell_state)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else: # RNN (hidden_state)
                hidden = hidden.detach()

            model.zero_grad()

            # Forward pass: guess the next letters
            outputs, hidden = model(x, hidden)

            # Calculate loss (how far off the predictions were)
            loss = criterion(outputs, y.reshape(-1))

            # Backward pass (calculate feedback gradients)
            loss.backward()

            # Clip gradients to prevent exploding values from breaking model weights
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Adjust weights
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(batch_count, 1)

        # Print update and generate a short sample
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
        sample = generate_sentence(model, dataset, device, temperature=0.7)
        print(f"  Sample: {repr(sample)}")

    return model, avg_loss


# =====================================================================
# 4. Corpus Loading & CLI entry
# =====================================================================

# Small embedded fallback so --test-run (and any run without data/shakespeare.txt
# present) stays instant and fully offline. Real verbatim excerpt of the fetched
# corpus (the opening Citizens' scene), not synthetic text - kept identical to
# Project 1's excerpt so the two projects share the same fallback universe.
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
    parser = argparse.ArgumentParser(description="PyTorch Character-level Shakespeare Text Generator (RNN vs LSTM).")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast integration test on a tiny embedded text excerpt.")
    parser.add_argument("--demo", action="store_true", help="Fast-but-real training pass on a subset of the full corpus; prints a machine-readable result line for scripts/compare_text_generators.py.")
    parser.add_argument("--model", type=str, choices=["rnn", "lstm"], default="lstm", help="Choose architecture: rnn or lstm.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of text samples to generate at the end.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        dataset = TextDataset(FALLBACK_TEXT, seq_len=16)

        if args.model == "rnn":
            model = CharRNN(dataset.vocab_size, embed_size=16, hidden_size=32, n_layers=1).to(device)
        else:
            model = CharLSTM(dataset.vocab_size, embed_size=16, hidden_size=32, n_layers=1).to(device)

        train_model(model, dataset, device, epochs=2, batch_size=4, lr=0.01, seq_len=16, model_name=args.model.upper())
        samples = generate_sentences(model, dataset, device, num_sentences=2)
        print(f"  Sample generations: {samples}")
        print("[SUCCESS] Integration Test Successful!")
        sys.exit(0)

    if args.demo:
        start_time = time.time()
        text = load_corpus(__file__)[:DEMO_CORPUS_CHARS]
        dataset = TextDataset(text, seq_len=64)

        if args.model == "rnn":
            model = CharRNN(dataset.vocab_size, embed_size=32, hidden_size=128, n_layers=1).to(device)
        else:
            model = CharLSTM(dataset.vocab_size, embed_size=32, hidden_size=128, n_layers=1).to(device)

        model, final_loss = train_model(model, dataset, device, epochs=10, batch_size=64, lr=0.005, seq_len=64, model_name=args.model.upper())
        samples = generate_sentences(model, dataset, device, num_sentences=args.num_samples)
        elapsed = time.time() - start_time

        result = {
            "project": 2,
            "arch": f"{args.model.upper()} (sequential hidden state)",
            "samples": samples,
            "val_loss": round(float(final_loss), 4),
            "train_seconds": round(elapsed, 2),
        }
        print("###RESULT_JSON### " + json.dumps(result))
        sys.exit(0)

    # Full Run
    print("=== Training Recurrent Language Model: Shakespeare Text Generator ===")

    text = load_corpus(__file__)
    dataset = TextDataset(text, seq_len=128)

    print(f"Dataset Vocabulary Size: {dataset.vocab_size} unique characters")
    print(f"Corpus Length: {len(text)} characters")

    if args.model == "rnn":
        model = CharRNN(dataset.vocab_size, embed_size=64, hidden_size=256, n_layers=2).to(device)
    else:
        model = CharLSTM(dataset.vocab_size, embed_size=64, hidden_size=256, n_layers=2).to(device)

    train_model(model, dataset, device, epochs=args.epochs, batch_size=64, lr=0.003, seq_len=128, model_name=args.model.upper())

    # Generate final samples
    print(f"\n=== GENERATING {args.num_samples} SAMPLE TEXT CONTINUATIONS (Low Temperature: 0.5) ===")
    for sample in generate_sentences(model, dataset, device, num_sentences=args.num_samples, temperature=0.5):
        print(f"  {sample!r}")

    print(f"\n=== GENERATING {args.num_samples} SAMPLE TEXT CONTINUATIONS (High Temperature: 1.0) ===")
    for sample in generate_sentences(model, dataset, device, num_sentences=args.num_samples, temperature=1.0):
        print(f"  {sample!r}")

    print("\n=== Sequence Modeling training finished successfully! ===")
