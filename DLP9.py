# ---------------------------
# 1. Import Required Libraries
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 2. Prepare the Data
# ---------------------------

# Our training text (you can replace this with any other text)
text = "hello world, this is a simple text generation using LSTMs."

# Get a sorted list of unique characters (vocabulary)
chars = sorted(set(text))

# Create mappings from characters to indices and vice versa
char2idx = {char: idx for idx, char in enumerate(chars)}  # e.g., {' ': 0, 'L': 1, ...}
idx2char = {idx: char for idx, char in enumerate(chars)}  # e.g., {0: ' ', 1: 'L', ...}

# Set sequence length (number of previous characters used to predict the next)
seq_len = 10

# Create input sequences (X) and corresponding target characters (Y)
# X will be a list of sequences, each of length 10
# Y will be the character that follows each sequence
X = []
Y = []

for i in range(len(text) - seq_len):
    input_seq = text[i:i + seq_len]
    target_char = text[i + seq_len]

    X.append([char2idx[char] for char in input_seq])  # convert chars to indices
    Y.append(char2idx[target_char])                   # convert target char to index

# Convert lists to PyTorch tensors
X = torch.tensor(X, dtype=torch.long)  # Shape: (num_samples, seq_len)
Y = torch.tensor(Y, dtype=torch.long)  # Shape: (num_samples,)

# ---------------------------
# 3. Define the LSTM Model
# ---------------------------

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=128):
        super(TextLSTM, self).__init__()

        # Embedding layer: converts character indices to dense vectors
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # LSTM layer: processes sequences of embeddings
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        # Fully connected output layer: maps LSTM output to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embed(x)                      # shape -> (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(x)             # h_n shape -> (1, batch_size, hidden_dim)
        output = self.fc(h_n[-1])              # Take the last hidden state for prediction
        return output                          # shape -> (batch_size, vocab_size)

# ---------------------------
# 4. Training Setup
# ---------------------------

# Define vocabulary size
vocab_size = len(chars)

# Initialize model, loss function, and optimizer
model = TextLSTM(vocab_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------
# 5. Train the Model
# ---------------------------

# Number of training epochs
epochs = 50

for epoch in range(epochs):
    model.train()              # Set model to training mode
    optimizer.zero_grad()      # Clear previous gradients

    output = model(X)          # Forward pass
    loss = loss_function(output, Y)  # Compute loss
    loss.backward()            # Backpropagation
    optimizer.step()           # Update model weights

    # Optionally print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# ---------------------------
# 6. Text Generation Function
# ---------------------------

def generate_text(start_seq, length=50):
    """
    Generate text from a starting sequence.
    """
    model.eval()  # Set model to evaluation mode

    # Convert the seed sequence to indices
    input_indices = [char2idx[c] for c in start_seq]

    # Generate characters one by one
    for _ in range(length):
        # Take the last `seq_len` characters for prediction
        input_seq = input_indices[-seq_len:]

        # Pad input if it's shorter than seq_len
        if len(input_seq) < seq_len:
            input_seq = [0] * (seq_len - len(input_seq)) + input_seq

        # Convert to tensor with batch dimension
        x = torch.tensor([input_seq], dtype=torch.long)

        # Predict the next character
        with torch.no_grad():
            output = model(x)
            predicted_index = torch.argmax(output, dim=1).item()

        # Append the predicted character index to the input
        input_indices.append(predicted_index)

    # Convert indices back to characters
    generated_text = start_seq + ''.join(idx2char[idx] for idx in input_indices[len(start_seq):])
    return generated_text

# ---------------------------
# 7. Generate and Print Text
# ---------------------------

# Start with a seed string and generate new text
seed = "hello wor"
generated = generate_text(seed, length=50)
print("\nGenerated Text:\n", generated)
