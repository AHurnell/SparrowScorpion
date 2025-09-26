# train.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tinycnn_model import TinyCNN

# Data transforms
transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
])

# Load datasets
train_ds = datasets.ImageFolder("dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("dataset/val",   transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# Setup model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        opt.step()
    print("Finished epoch", epoch+1)

torch.save(model.state_dict(), "tinycnn.pth")
print("✅ Training complete. Model saved as tinycnn.pth")
