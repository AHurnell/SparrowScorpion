import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*25*25, 64)   # 100x100 images
        self.fc2 = nn.Linear(64, 2)          # 2 classes: drone / non-drone

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32*25*25)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
])

train_ds = datasets.ImageFolder("dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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
print("Training complete. Model saved as tinycnn.pth")
