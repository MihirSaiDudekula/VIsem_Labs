import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = DataLoader(datasets.CIFAR10('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.CIFAR10('.', train=False, download=True, transform=transform), batch_size=64)

# 2. Define Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# 3. Training and evaluation function
def train_and_test(model, optimizer, epochs=3):
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    # Test phase
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
    return 100 * correct / len(test_loader.dataset)

# 4. Run experiments
model_sgd = SimpleCNN()
model_adam = SimpleCNN()

acc_sgd = train_and_test(model_sgd, optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9))
acc_adam = train_and_test(model_adam, optim.Adam(model_adam.parameters(), lr=0.001))

print(f"SGD Accuracy: {acc_sgd:.2f}%")
print(f"Adam Accuracy: {acc_adam:.2f}%")
