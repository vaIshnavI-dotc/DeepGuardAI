import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ===============================
# 1. DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# 2. DATA TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===============================
# 3. LOAD DATASET
# ===============================
data_path = os.path.join(os.getcwd(), "dataset")

if not os.path.exists(data_path):
    raise FileNotFoundError("Dataset folder not found. Make sure 'dataset/real' and 'dataset/fake' exist.")

full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
print(f"Dataset Classes: {full_dataset.class_to_idx}")
print(f"Total Images: {len(full_dataset)}")

# ===============================
# 4. TRAIN / VALIDATION SPLIT
# ===============================
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# ===============================
# 5. LOAD PRETRAINED MODEL
# ===============================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ===============================
# 6. LOSS & OPTIMIZER
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ===============================
# 7. TRAINING LOOP
# ===============================
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ===========================
    # VALIDATION
    # ===========================
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / val_size
    avg_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Complete")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%\n")

# ===============================
# 8. SAVE MODEL
# ===============================
torch.save(model.state_dict(), "deepfake_model.pth")
print("Model saved as deepfake_model.pth")