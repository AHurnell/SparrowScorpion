# test.py
import torch
from PIL import Image
from torchvision import transforms
from tinycnn_model import TinyCNN

# Load trained model
model = TinyCNN()
model.load_state_dict(torch.load("tinycnn.pth", map_location="cpu"))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
])

# Single image test
img = Image.open("some_test_picture.jpg").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)
    pred = out.argmax(dim=1).item()

print("Prediction:", "drone" if pred==0 else "non-drone")

# Optional speed test
import time
N = 200
start = time.time()
with torch.no_grad():
    for _ in range(N):
        _ = model(x)
end = time.time()
print(f"Speed: {N/(end-start):.1f} images/sec")
