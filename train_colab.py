"""
Plant Disease Detection - Colab Training Script
--------------------------------------------------
Built for a plant disease detection computer vision project focused on
precision/vertical farming applications.

WHY THIS SCRIPT:
- Runs on Colab's free GPU (15-30x faster than your CPU)
- Saves a checkpoint after EVERY epoch, so if Colab disconnects
  or you close the tab, you resume instead of starting over
- Automatically exports the final model to ONNX when training finishes

HOW TO USE ON COLAB:
1. Go to https://colab.research.google.com -> New Notebook
2. Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4)
3. Upload this file: click the folder icon on the left -> upload train_colab.py
4. Upload your dataset (see Step A below) OR mount Google Drive
5. In a Colab cell, run:  !python train_colab.py
6. If disconnected, just re-run the same command - it auto-resumes.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ============================================================
# CONFIG - adjust these if needed
# ============================================================
DATA_DIR = "data"                  # expects data/train, data/val, data/test
CHECKPOINT_PATH = "checkpoint.pth"  # auto-saved after every epoch
BEST_MODEL_PATH = "best_model.pth"
ONNX_EXPORT_PATH = "plant_disease_model.onnx"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: No GPU detected. In Colab, go to Runtime > Change runtime "
          "type > GPU before running this script.")

# ============================================================
# DATA PIPELINE
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")
print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# MODEL - ResNet50 transfer learning
# ============================================================
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False  # freeze backbone
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(device)

model = build_model(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============================================================
# CHECKPOINT / RESUME LOGIC
# ============================================================
start_epoch = 0
best_val_acc = 0.0
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

if os.path.exists(CHECKPOINT_PATH):
    print(f"Found existing checkpoint at {CHECKPOINT_PATH} - resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    history = checkpoint["history"]
    print(f"Resuming from epoch {start_epoch}")
else:
    print("No checkpoint found - starting fresh training run.")

# ============================================================
# TRAIN / VALIDATE
# ============================================================
def run_epoch(loader, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(training)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


print(f"\nStarting training from epoch {start_epoch + 1} to {NUM_EPOCHS}...\n")

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Acc: {train_acc:.2%} Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.2%} Loss: {val_loss:.4f} | "
          f"{elapsed:.1f}s")

    # Save checkpoint EVERY epoch so we can always resume
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "class_names": class_names,
    }, CHECKPOINT_PATH)

    # Save best model separately
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc,
        }, BEST_MODEL_PATH)
        print(f"  -> New best model saved (val acc {val_acc:.2%})")

# Save training history as JSON for your README / dashboard later
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")

# ============================================================
# ONNX EXPORT (for edge deployment - your next milestone)
# ============================================================
print("\nExporting best model to ONNX...")
best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
export_model = build_model(num_classes)
export_model.load_state_dict(best_checkpoint["model_state_dict"])
export_model.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.onnx.export(
    export_model,
    dummy_input,
    ONNX_EXPORT_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12,
)
print(f"ONNX model saved to {ONNX_EXPORT_PATH}")
print("\nDone! Download best_model.pth, plant_disease_model.onnx, and "
      "training_history.json from the Colab file browser (left sidebar).")
