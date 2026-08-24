"""
Plant Disease Detection - Fine-Tuning Script (Round 2)
--------------------------------------------------------
Starts from your already-trained best_model.pth and unfreezes
the last ResNet50 block (layer4) + classifier head, training
them together with a small learning rate. This lets the model
adjust deeper visual features specifically for leaf disease
patterns, instead of only the final decision layer.

WHY THIS SHOULD IMPROVE ACCURACY:
Round 1 only trained the final layer on top of a frozen,
general-purpose ResNet50. That plateaued around 73% train /
77% val accuracy because the deeper layers were still tuned for
generic ImageNet objects (cats, cars, etc.), not leaf textures
and lesion patterns specifically.

HOW TO USE ON COLAB:
1. Make sure best_model.pth is still in /content (from Round 1).
   If your Colab session restarted, re-upload it with:
     from google.colab import files
     uploaded = files.upload()
2. Make sure your data/ folder (train/val/test) is still there too.
   If not, re-upload and unzip data.zip again as before.
3. Upload this file (finetune_colab.py) the same way.
4. Run:  !python finetune_colab.py
5. When done, download finetuned_best_model.pth and re-export to ONNX
   using export_onnx_standalone.py (update MODEL_PATH inside it first).
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
# CONFIG
# ============================================================
DATA_DIR = "data"
PREVIOUS_MODEL_PATH = "best_model.pth"          # Round 1 result (input)
CHECKPOINT_PATH = "finetune_checkpoint.pth"      # auto-saved every epoch
BEST_MODEL_PATH = "finetuned_best_model.pth"     # Round 2 result (output)
NUM_EPOCHS = 10                                   # fewer needed - we're refining, not starting fresh
BATCH_SIZE = 32
LEARNING_RATE = 1e-5                              # much smaller - we don't want to destroy existing knowledge
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATA PIPELINE (same as round 1)
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# MODEL - load Round 1 weights, then unfreeze layer4 + fc
# ============================================================
def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

model = build_model(num_classes).to(device)

print(f"Loading Round 1 weights from {PREVIOUS_MODEL_PATH}...")
prev_checkpoint = torch.load(PREVIOUS_MODEL_PATH, map_location=device)
model.load_state_dict(prev_checkpoint["model_state_dict"])
print(f"Loaded. Round 1 val accuracy was: {prev_checkpoint.get('val_acc', 'unknown')}")

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (ResNet50's last residual block) + the classifier head
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
total_trainable = sum(p.numel() for p in trainable_params)
print(f"Fine-tuning {total_trainable:,} parameters (layer4 + classifier)")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ============================================================
# CHECKPOINT / RESUME LOGIC
# ============================================================
start_epoch = 0
best_val_acc = prev_checkpoint.get("val_acc", 0.0)
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

if os.path.exists(CHECKPOINT_PATH):
    print(f"Found fine-tune checkpoint - resuming...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    history = checkpoint["history"]
    print(f"Resuming from epoch {start_epoch}")

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


print(f"\nStarting fine-tuning from epoch {start_epoch + 1} to {NUM_EPOCHS}...")
print(f"Starting point: Round 1 val accuracy = {best_val_acc:.2%}\n")

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

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "class_names": class_names,
    }, CHECKPOINT_PATH)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc,
        }, BEST_MODEL_PATH)
        print(f"  -> New best model saved (val acc {val_acc:.2%})")

with open("finetune_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nFine-tuning complete. Best validation accuracy: {best_val_acc:.2%}")
print(f"Improvement over Round 1: {best_val_acc - prev_checkpoint.get('val_acc', 0):.2%}")
print(f"\nDownload {BEST_MODEL_PATH} and re-run the ONNX export script "
      f"(update MODEL_PATH to '{BEST_MODEL_PATH}' inside it) to get your improved edge model.")
