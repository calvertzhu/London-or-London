import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.primary_model.resnet_cbam_mlp import ResNet50_CBAM_MLP

# Configs
data_dir = "report_data"
batch_size = 16
epochs = 10
learning_rate = 0.001
weight_decay = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Load train and val datasets from split folders
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Model, loss, optimizer
model = ResNet50_CBAM_MLP().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning curves
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Best model tracking
best_val_acc = 0.0
best_model_state = None

# Training loop
for epoch in range(epochs):
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

# Plot loss curve
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig("loss_curve.png")
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
    }, 'best_model.pth')
    print(f"Best model saved to best_model.pth (Val Acc: {best_val_acc:.4f})")
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
    }, 'trained_model.pth')
    print("Last model saved to trained_model.pth")

# Plot accuracy curve
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

print("Training complete. Learning curves saved as 'loss_curve.png' and 'accuracy_curve.png'.")
