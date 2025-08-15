import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_model.baseline_cnn import BaselineCNN

# Configs
data_dir = "report_data"
test_data_dir = "test_data"
batch_size = 32  # Larger batch size for baseline model
epochs = 30
learning_rate = 0.0001
weight_decay = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms (simpler for baseline model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Note: No normalization here as the model handles it internally
])

# Load train, val, and test datasets
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

# Load test dataset if it exists
test_dataset = None
test_loader = None
if Path(test_data_dir).exists():
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
if test_dataset:
    print(f"Test dataset size: {len(test_dataset)}")

# Check if batch size is appropriate
if batch_size > len(train_dataset):
    print(f"Warning: batch_size ({batch_size}) > train dataset size ({len(train_dataset)})")
    print("Reducing batch_size to train dataset size")
    batch_size = len(train_dataset)
elif batch_size > len(val_dataset):
    print(f"Warning: batch_size ({batch_size}) > val dataset size ({len(val_dataset)})")
    print("Reducing batch_size to val dataset size")
    batch_size = len(val_dataset)

print(f"Using batch_size: {batch_size}")

# Model, loss, optimizer
model = BaselineCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning curves
train_losses, val_losses = [], []
train_accs, val_accs = [], []

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_model_state = checkpoint.get('best_model_state', None)
        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"   Starting from epoch: {start_epoch}")
        print(f"   Best validation accuracy so far: {best_val_acc:.4f}")
        return start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc, best_model_state
    return 0, [], [], [], [], 0.0, None

# Checkpoint saving configuration
save_every_n_epochs = 5
checkpoint_dir = Path("checkpoints_baseline")
checkpoint_dir.mkdir(exist_ok=True)

# Best model tracking
best_val_acc = 0.0
best_model_state = None

# Check for existing checkpoint to resume training
resume_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc, best_model_state = load_checkpoint(
    model, optimizer, resume_checkpoint
)

# Training loop
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        running_corrects += torch.sum(preds == labels.bool())

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())

    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            val_corrects += torch.sum(preds == labels.bool())

    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = val_corrects.double() / len(val_dataset)
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.item())

    # Save best model based on validation accuracy
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_model_state = model.state_dict().copy()
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | "
              f"Val Loss={val_epoch_loss:.4f}, Acc={val_epoch_acc:.4f} "
              f"*** NEW BEST MODEL ***")
    else:
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | "
              f"Val Loss={val_epoch_loss:.4f}, Acc={val_epoch_acc:.4f}")
    
    # Save intermediate checkpoint
    if (epoch + 1) % save_every_n_epochs == 0:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_model_state': best_model_state,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest checkpoint for resuming
        latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_model_state': best_model_state,
        }, latest_checkpoint_path)

# Plot loss curve
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Baseline Model - Loss Curve')
plt.legend()
plt.savefig("baseline_loss_curve.png")
plt.close()

# Save the best model (based on validation accuracy)
if best_model_state is not None:
    torch.save({
        'epoch': epochs,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
    }, 'baseline_best_model.pth')
    print(f"Best baseline model saved to baseline_best_model.pth (Val Acc: {best_val_acc:.4f})")
else:
    # Fallback: save the last model if no best model was found
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }, 'baseline_trained_model.pth')
    print("Last baseline model saved to baseline_trained_model.pth")

# Plot accuracy curve
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Baseline Model - Accuracy Curve')
plt.legend()
plt.savefig("baseline_accuracy_curve.png")
plt.close()

print("Baseline model training complete. Learning curves saved as 'baseline_loss_curve.png' and 'baseline_accuracy_curve.png'.") 