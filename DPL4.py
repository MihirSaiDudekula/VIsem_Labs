import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load MNIST dataset
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, download=True, transform=transform), batch_size=64)

# 2. Define model with optional BN and Dropout
class SimpleNN(nn.Module):
    def __init__(self, use_bn=False, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn = nn.BatchNorm1d(256) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# 3. Training & Evaluation
def train_and_test(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(3):  # Fewer epochs for speed
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
    return 100 * correct / len(test_loader.dataset)

# 4. Run experiments
acc_plain = train_and_test(SimpleNN())
acc_bn = train_and_test(SimpleNN(use_bn=True))
acc_dropout = train_and_test(SimpleNN(use_dropout=True))

print(f"Plain: {acc_plain:.2f}%, With BN: {acc_bn:.2f}%, With Dropout: {acc_dropout:.2f}%")
