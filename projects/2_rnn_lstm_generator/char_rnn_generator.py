# =====================================================================
# Project 2: Character-Level RNN & LSTM Sequence Generator from Scratch
# Author: Shabih Ehtesham
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

# =====================================================================
# 1. Model Implementations in PyTorch
# =====================================================================

class CharRNN(nn.Module):
    """
    A Character-level Recurrent Neural Network (RNN).
    Shows how sequence processing works by updating a hidden state at each timestep.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Word/Char Embedding layer
        self.encoder = nn.Embedding(vocab_size, embed_size)
        
        # RNN Layer
        # batch_first=True makes input/output shapes: (batch_size, seq_len, hidden_size)
        self.rnn = nn.RNN(embed_size, hidden_size, n_layers, batch_first=True)
        
        # Decoder layer mapping to vocabulary logits
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_len)
        embed = self.encoder(x) # (batch_size, seq_len, embed_size)
        out, hidden = self.rnn(embed, hidden) # out: (batch_size, seq_len, hidden_size)
        
        # Flatten for linear layer projection
        # Reshape to (batch_size * seq_len, hidden_size)
        out_reshaped = out.contiguous().view(-1, self.hidden_size)
        decoded = self.decoder(out_reshaped)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state with zeros
        # Shape: (n_layers, batch_size, hidden_size)
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)


class CharLSTM(nn.Module):
    """
    A Character-level Long Short-Term Memory (LSTM) network.
    Shows how gate mechanisms solve the vanishing gradient problem in RNNs.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(vocab_size, embed_size)
        
        # LSTM Layer
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # hidden is a tuple of (h_t, c_t)
        embed = self.encoder(x)
        out, hidden = self.lstm(embed, hidden)
        
        out_reshaped = out.contiguous().view(-1, self.hidden_size)
        decoded = self.decoder(out_reshaped)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        # LSTM hidden state is a tuple (hidden_state, cell_state)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# =====================================================================
# 2. Text Preprocessing & Dataset Helper
# =====================================================================

class TextDataset:
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        
        # Find unique characters (vocabulary)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Mapping dicts
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Encode whole corpus
        self.data = np.array([self.char2idx[c] for c in text], dtype=np.int64)

    def get_batches(self, batch_size):
        # Calculate maximum possible complete batches
        num_characters_per_batch = batch_size * self.seq_len
        num_batches = len(self.data) // num_characters_per_batch
        
        if num_batches == 0:
            raise ValueError(f"Corpus too small for batch size {batch_size} and sequence length {self.seq_len}")
            
        # Slice data to fit exactly into batches
        truncated_data = self.data[:num_batches * num_characters_per_batch]
        # Reshape to (batch_size, -1) so we can slice sequences sequentially
        reshaped_data = truncated_data.reshape(batch_size, -1)
        
        for n in range(0, reshaped_data.shape[1] - self.seq_len, self.seq_len):
            x = reshaped_data[:, n:n+self.seq_len]
            # y is target sequence, shifted by 1 character
            y = reshaped_data[:, n+1:n+self.seq_len+1]
            yield torch.from_numpy(x), torch.from_numpy(y)


# =====================================================================
# 3. Training & Text Generation Functions
# =====================================================================

def generate_text(model, dataset, device, start_str="The ", predict_len=100, temperature=0.8):
    """
    Generates text character-by-character using temperature scaling.
    """
    model.eval()
    chars = [c for c in start_str]
    hidden = model.init_hidden(1, device)
    
    # Priming the model with the seed string
    for char in start_str[:-1]:
        x = torch.tensor([[dataset.char2idx[char]]]).to(device)
        _, hidden = model(x, hidden)
        
    # Generate following characters
    curr_char = start_str[-1]
    for _ in range(predict_len):
        x = torch.tensor([[dataset.char2idx[curr_char]]]).to(device)
        logits, hidden = model(x, hidden)
        
        # Scale output logits with temperature
        logits = logits[-1] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        
        # Sample from the probability distribution
        char_idx = np.random.choice(len(probs), p=probs)
        curr_char = dataset.idx2char[char_idx]
        chars.append(curr_char)
        
    return "".join(chars)


def train_model(model, dataset, device, epochs=10, batch_size=32, lr=0.002, seq_len=50, model_name="LSTM"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training {model_name} Model on device: {device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        hidden = model.init_hidden(batch_size, device)
        total_loss = 0
        batch_count = 0
        
        for x, y in dataset.get_batches(batch_size):
            batch_count += 1
            x, y = x.to(device), y.to(device)
            
            # Detach hidden states to prevent backpropagating to the beginning of training
            if isinstance(hidden, tuple): # LSTM hidden state
                hidden = (hidden[0].detach(), hidden[1].detach())
            else: # RNN hidden state
                hidden = hidden.detach()
                
            model.zero_grad()
            
            # Forward pass
            outputs, hidden = model(x, hidden)
            
            # Compute loss (flatten targets to match outputs)
            loss = criterion(outputs, y.reshape(-1))
            
            # Backward and optimize
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(batch_count, 1)
        
        # Print update and generate a short sample
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
        sample = generate_text(model, dataset, device, start_str="The ", predict_len=40, temperature=0.7)
        print(f"  Sample: {repr(sample)}")

    return model


# =====================================================================
# 4. Main script entry
# =====================================================================

# A classic small text corpus to train on if no file is provided
DEFAULT_CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th' oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th' unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Character-level Language Model (RNN vs LSTM).")
    parser.add_argument("--test-run", action="store_true", help="Runs a very fast integration test on synthetic Shakespeare text.")
    parser.add_argument("--model", type=str, choices=["rnn", "lstm"], default="lstm", help="Choose architecture: rnn or lstm.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_run:
        print("=== RUNNING FAST INTEGRATION TEST ===")
        # Build small corpus
        test_corpus = "The quick brown fox jumps over the lazy dog. To be or not to be, that is the question of the model."
        dataset = TextDataset(test_corpus, seq_len=10)
        
        # Instantiate small model
        if args.model == "rnn":
            model = CharRNN(dataset.vocab_size, embed_size=16, hidden_size=32, n_layers=1).to(device)
        else:
            model = CharLSTM(dataset.vocab_size, embed_size=16, hidden_size=32, n_layers=1).to(device)
            
        train_model(model, dataset, device, epochs=2, batch_size=4, lr=0.01, seq_len=10, model_name=args.model.upper())
        print("[SUCCESS] Integration Test Successful!")
        sys.exit(0)

    # Full Run
    print("=== Training Recurrent Language Model ===")
    
    # Double the default Shakespeare corpus to make it slightly longer for training
    corpus = DEFAULT_CORPUS * 3
    dataset = TextDataset(corpus, seq_len=50)
    
    print(f"Dataset Vocabulary Size: {dataset.vocab_size} unique characters")
    print(f"Corpus Length: {len(corpus)} characters")
    
    if args.model == "rnn":
        model = CharRNN(dataset.vocab_size, embed_size=64, hidden_size=128, n_layers=2).to(device)
    else:
        model = CharLSTM(dataset.vocab_size, embed_size=64, hidden_size=128, n_layers=2).to(device)
        
    train_model(model, dataset, device, epochs=args.epochs, batch_size=16, lr=0.005, seq_len=50, model_name=args.model.upper())
    
    # Generate final response
    print("\n=== GENERATING TEXT (Low Temperature: 0.3) ===")
    print(generate_text(model, dataset, device, start_str="To be ", predict_len=200, temperature=0.3))
    
    print("\n=== GENERATING TEXT (Medium Temperature: 0.7) ===")
    print(generate_text(model, dataset, device, start_str="To be ", predict_len=200, temperature=0.7))

    print("\n=== GENERATING TEXT (High Temperature: 1.2) ===")
    print(generate_text(model, dataset, device, start_str="To be ", predict_len=200, temperature=1.2))
    
    print("\n=== Sequence Modeling training finished successfully! ===")
