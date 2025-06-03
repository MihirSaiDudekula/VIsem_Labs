import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data Preparation
text = "hello world, this is a simple text generation using LSTMs."
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

seq_len = 10
X = [ [char2idx[c] for c in text[i:i+seq_len]] for i in range(len(text)-seq_len) ]
Y = [ char2idx[text[i+seq_len]] for i in range(len(text)-seq_len) ]

X = torch.tensor(X)
Y = torch.tensor(Y)

# 2. LSTM Model
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# 3. Train Model
vocab_size = len(chars)
model = TextLSTM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for _ in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, Y)
    loss.backward()
    optimizer.step()

# 4. Generate Text
def generate(start, length=50):
    model.eval()
    seq = [char2idx[c] for c in start]
    for _ in range(length):
        x = torch.tensor([seq[-seq_len:]])
        with torch.no_grad():
            out = model(x)
        pred = torch.argmax(out, dim=1).item()
        seq.append(pred)
    return start + ''.join(idx2char[i] for i in seq[len(start):])

print(generate("hello wor"))
