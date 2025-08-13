#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates trained model on test data and generates performance metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
import sys

# Add models to path
sys.path.append('..')
from models.primary_model.resnet_cbam_mlp import ResNet50_CBAM_MLP

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = ResNet50_CBAM_MLP().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}, using untrained model")
    
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and true labels."""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['London_ON', 'London_UK'],
                yticklabels=['London_ON', 'London_UK'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_prob, save_path="roc_curve.png"):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    # Load test data
    test_data_dir = "../test_data"
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        print("Please collect test data first using test_data_collector.py")
        return
    
    # Create test dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model
    model_path = "trained_model.pth"
    model = load_model(model_path, device)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['London_ON', 'London_UK']))
    
    # Plot results
    plot_confusion_matrix(true_labels, predictions)
    plot_roc_curve(true_labels, probabilities)
    
    # Save results
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }
    
    np.save('evaluation_results.npy', results)
    print("\nEvaluation complete! Results saved to evaluation_results.npy")

if __name__ == "__main__":
    main() 