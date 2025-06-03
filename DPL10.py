import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0
        self.heads = heads
        self.head_dim = embed_size // heads

        self.to_qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_len, _ = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = [t.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2) for t in qkv]

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, seq_len, -1)

        return self.fc_out(out)

# Usage
x = torch.rand(2, 10, 128)  # (batch, seq_len, embed_dim)
mha = MultiHeadAttention(embed_size=128, heads=8)
out = mha(x)
print(out.shape)  # torch.Size([2, 10, 128])
