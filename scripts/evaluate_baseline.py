#!/usr/bin/env python3
"""
Baseline Model Evaluation Script

Evaluates trained baseline model on test data and generates performance metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import os
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.baseline_model.baseline_cnn import BaselineCNN

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = BaselineCNN().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded baseline model from {model_path}")
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

def plot_confusion_matrix(y_true, y_pred, save_path="baseline_confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['London_ON', 'London_UK'],
                yticklabels=['London_ON', 'London_UK'])
    plt.title('Baseline Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_prob, save_path="baseline_roc_curve.png"):
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
    plt.title('Baseline Model - Receiver Operating Characteristic (ROC) Curve')
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
        # Note: No normalization here as the model handles it internally
    ])
    
    # Load test data
    test_data_dir = "test_data"
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        print("Please collect test data first using test_data_collector.py")
        return
    
    # Create test dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Load model (try best model first, then fallback to trained model)
    model_path = "baseline_best_model.pth"
    if not os.path.exists(model_path):
        model_path = "baseline_trained_model.pth"
        if not os.path.exists(model_path):
            print(f"No baseline model found at baseline_best_model.pth or baseline_trained_model.pth")
            print("Please train the baseline model first using train_baseline.py")
            return
        else:
            print(f"Using baseline_trained_model.pth (best model not found)")
    else:
        print(f"Using baseline_best_model.pth")
    
    model = load_model(model_path, device)
    
    # Evaluate model
    print("Evaluating baseline model...")
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    
    # Calculate F1 scores
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    # Calculate precision and recall
    precision_macro = precision_score(true_labels, predictions, average='macro')
    recall_macro = recall_score(true_labels, predictions, average='macro')
    
    print(f"\nBASELINE MODEL PERFORMANCE METRICS")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    print(f"  London_ON: {f1_per_class[0]:.4f}")
    print(f"  London_UK: {f1_per_class[1]:.4f}")
    
    # Classification report
    print("\nDETAILED CLASSIFICATION REPORT")
    print(classification_report(true_labels, predictions, 
                              target_names=['London_ON', 'London_UK']))
    
    # Plot results
    plot_confusion_matrix(true_labels, predictions)
    plot_roc_curve(true_labels, probabilities)
    
    # Create metrics summary plot
    metrics_names = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Precision', 'Macro Recall']
    metrics_values = [accuracy, f1_macro, f1_weighted, precision_macro, recall_macro]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B4513'])
    plt.title('Baseline Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("baseline_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }
    
    np.save('baseline_evaluation_results.npy', results)
    
    print("\nBaseline model evaluation complete!")
    print("Results saved to baseline_evaluation_results.npy")
    print("Performance metrics plot saved to baseline_performance_metrics.png")

if __name__ == "__main__":
    main() 