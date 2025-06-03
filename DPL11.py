import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_data = Subset(datasets.MNIST(root="./",train,download,transform),range(200))
train_loader = DataLoader(td,bs=10,s=t)
test_loader = DataLoader(test_data,bs)
# Data
transform = transforms.ToTensor()
train_data = Subset(datasets.MNIST(root="./data", train=True, download=True, transform=transform), range(200))
test_data = Subset(datasets.MNIST(root="./data", train=False, download=True, transform=transform), range(50))
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

class AutoEncoder(nn.Module):
    def __init__(self):
        super.
        self.encoder=nn.Sequential(nn.linear(784,64),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(64,784),nn.Sigmoid)
# Model
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(784, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 784), nn.Sigmoid())

    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.decoder(self.encoder(x))

    model = AutoEncoder()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters,lr=0.001)

    for epoch in range(10):
        model.train()
        for x, _  in train_loader:
            out = model(x)
            loss = loss_fn(out,)
            optimizer.zero_grad()
            loss.backward()
            
            

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.decoder(self.encoder(x))

model = AutoEncoder().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    model.train()
    for x, _ in train_loader:
        x = x.to(device)
        out = model(x)
        loss = loss_fn(out, x.view(x.size(0), -1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            out = model(x)
            test_loss += loss_fn(out, x.view(x.size(0), -1)).item()

    print(f"Epoch {epoch+1}, Test Loss: {test_loss / len(test_loader):.4f}")
