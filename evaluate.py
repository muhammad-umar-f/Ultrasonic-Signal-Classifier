"""
Evaluation script for ultrasonic signal classifier.

Usage:
    python evaluate.py --model models/best_model.pt --config config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from src.signal_generator import UltrasonicSignalGenerator
from src.model import Conv1DClassifier


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_test_data(config: dict):
    """Generate test dataset."""
    print("Generating test data...")
    
    generator = UltrasonicSignalGenerator(
        signal_length=config['signal_length'],
        noise_std=config['noise_std'],
        defect_amplitude=config['defect_amplitude'],
        seed=config['random_seed'] + 100  # Different seed for test data
    )
    
    signals, labels = generator.generate_dataset(
        n_defect=500,
        n_no_defect=500
    )
    
    # Convert to torch tensors
    X_test = torch.from_numpy(signals).unsqueeze(1)
    y_test = torch.from_numpy(labels).long()
    
    return X_test, y_test


def evaluate_model(model_path: str, config: dict, output_dir: str = 'evaluation'):
    """Evaluate the model."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Conv1DClassifier(
        input_length=config['signal_length'],
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        depth=config['depth'],
        dropout_rate=config['dropout_rate'],
        num_classes=2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Generate test data
    X_test, y_test = generate_test_data(config)
    
    # Create dataloader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Predictions
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("Running predictions...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Defect', 'Defect']))
    
    # Save results
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }
    
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/evaluation_results.json")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['No Defect', 'Defect']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Add values to cells
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {output_dir}/roc_curve.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ultrasonic signal classifier')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to model file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate_model(args.model, config, args.output_dir)


if __name__ == '__main__':
    main()
