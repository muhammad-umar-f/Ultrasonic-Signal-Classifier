"""
Simplified demo script for ultrasonic signal classifier.
This version works without PyTorch DLL issues and demonstrates the full pipeline.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

print("=" * 70)
print("ULTRASONIC SIGNAL CLASSIFIER - DEMO")
print("=" * 70)

# Load configuration
print("\n1. Loading configuration...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"   - Signal length: {config['signal_length']}")
print(f"   - Noise std: {config['noise_std']}")
print(f"   - Defect amplitude: {config['defect_amplitude']}")

# Import signal generator
print("\n2. Initializing signal generator...")
from src.signal_generator import UltrasonicSignalGenerator

generator = UltrasonicSignalGenerator(
    signal_length=config['signal_length'],
    noise_std=config['noise_std'],
    defect_amplitude=config['defect_amplitude'],
    seed=config['random_seed']
)
print("   ✓ Signal generator ready")

# Generate dataset
print("\n3. Generating synthetic ultrasonic signals...")
signals, labels = generator.generate_dataset(
    n_defect=config['n_defect_samples'],
    n_no_defect=config['n_no_defect_samples']
)
print(f"   ✓ Generated {signals.shape[0]} signals")
print(f"   - Shape: {signals.shape}")
print(f"   - No-defect samples: {np.sum(labels == 0)}")
print(f"   - Defect samples: {np.sum(labels == 1)}")

# Visualize sample signals
print("\n4. Creating signal visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Sample Ultrasonic A-Scan Signals', fontsize=16, fontweight='bold')

# No-defect samples
no_defect_indices = np.where(labels == 0)[0][:3]
for i, idx in enumerate(no_defect_indices):
    axes[0, i].plot(signals[idx], linewidth=1.5, color='blue')
    axes[0, i].set_title(f'No Defect Sample {i+1}', fontweight='bold')
    axes[0, i].set_ylabel('Amplitude')
    axes[0, i].grid(True, alpha=0.3)

# Defect samples
defect_indices = np.where(labels == 1)[0][:3]
for i, idx in enumerate(defect_indices):
    axes[1, i].plot(signals[idx], color='red', linewidth=1.5)
    axes[1, i].set_title(f'Defect Sample {i+1}', fontweight='bold')
    axes[1, i].set_xlabel('Time Sample')
    axes[1, i].set_ylabel('Amplitude')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation/01_sample_signals.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved to evaluation/01_sample_signals.png")
plt.close()

# Split data
print("\n5. Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    signals, labels,
    test_size=config['test_split'],
    random_state=config['random_seed'],
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=config['val_split'],
    random_state=config['random_seed'],
    stratify=y_temp
)

print(f"   - Train set: {X_train.shape[0]} samples")
print(f"   - Validation set: {X_val.shape[0]} samples")
print(f"   - Test set: {X_test.shape[0]} samples")

# Try to import and use PyTorch
print("\n6. Testing model architecture...")
try:
    import torch
    import torch.nn as nn
    from src.model import Conv1DClassifier
    
    # Create model
    device = torch.device('cpu')
    model = Conv1DClassifier(
        input_length=config['signal_length'],
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        depth=config['depth'],
        dropout_rate=config['dropout_rate'],
        num_classes=2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created successfully")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Model config: {model.get_config()}")
    
    # Test forward pass
    X_sample = torch.from_numpy(X_train[:8]).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(X_sample)
    print(f"   ✓ Forward pass successful: output shape {output.shape}")
    
    pytorch_available = True
    
except Exception as e:
    print(f"   ⚠ PyTorch not available: {str(e)}")
    print("   Continuing with signal analysis only...")
    pytorch_available = False

# Statistical analysis
print("\n7. Signal Statistics Analysis...")
no_defect_signals = signals[labels == 0]
defect_signals = signals[labels == 1]

print(f"\n   No-Defect Signals:")
print(f"   - Mean amplitude: {np.mean(np.abs(no_defect_signals)):.4f}")
print(f"   - Std amplitude: {np.std(np.abs(no_defect_signals)):.4f}")
print(f"   - Max amplitude: {np.max(np.abs(no_defect_signals)):.4f}")
print(f"   - Energy (RMS): {np.sqrt(np.mean(no_defect_signals**2)):.4f}")

print(f"\n   Defect Signals:")
print(f"   - Mean amplitude: {np.mean(np.abs(defect_signals)):.4f}")
print(f"   - Std amplitude: {np.std(np.abs(defect_signals)):.4f}")
print(f"   - Max amplitude: {np.max(np.abs(defect_signals)):.4f}")
print(f"   - Energy (RMS): {np.sqrt(np.mean(defect_signals**2)):.4f}")

# Feature extraction for simple classifier
print("\n8. Building Simple Classifier (Feature-based)...")

def extract_features(signals):
    """Extract simple features from signals."""
    features = []
    for signal in signals:
        f = [
            np.max(np.abs(signal)),           # Max amplitude
            np.std(signal),                   # Standard deviation
            np.sqrt(np.mean(signal**2)),      # RMS energy
            np.mean(np.abs(signal)),          # Mean absolute amplitude
            np.sum(np.abs(np.diff(signal))),  # Total variation
        ]
        features.append(f)
    return np.array(features)

X_train_feat = extract_features(X_train)
X_val_feat = extract_features(X_val)
X_test_feat = extract_features(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_val_scaled = scaler.transform(X_val_feat)
X_test_scaled = scaler.transform(X_test_feat)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=config['random_seed'])
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = clf.predict(X_train_scaled)
y_pred_val = clf.predict(X_val_scaled)
y_pred_test = clf.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"   ✓ Random Forest Classifier trained")
print(f"   - Train accuracy: {train_acc:.4f}")
print(f"   - Validation accuracy: {val_acc:.4f}")
print(f"   - Test accuracy: {test_acc:.4f}")

# Confusion matrix
print("\n9. Generating evaluation plots...")
cm = confusion_matrix(y_test, y_pred_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Defect', 'Defect'],
            yticklabels=['No Defect', 'Defect'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')
axes[0].set_title(f'Confusion Matrix (Accuracy: {test_acc:.2%})', fontweight='bold', fontsize=12)

# Classification metrics
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average=None)

metrics = {
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

x = np.arange(len(metrics))
width = 0.35

for i, (metric_name, values) in enumerate(metrics.items()):
    axes[1].bar(i * width, values[0], width, label='No Defect' if i == 0 else '', alpha=0.8)
    axes[1].bar(i * width + width, values[1], width, label='Defect' if i == 0 else '', alpha=0.8)

axes[1].set_ylabel('Score')
axes[1].set_title('Performance Metrics', fontweight='bold', fontsize=12)
axes[1].set_xticks([i * width + width/2 for i in range(len(metrics))])
axes[1].set_xticklabels(metrics.keys())
axes[1].set_ylim([0, 1.1])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

for i, (metric_name, values) in enumerate(metrics.items()):
    for j, val in enumerate(values):
        axes[1].text(i * width + j * width, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('evaluation/02_evaluation_metrics.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved to evaluation/02_evaluation_metrics.png")
plt.close()

# Classification report
print("\n10. Detailed Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No Defect', 'Defect']))

# Summary
print("\n" + "=" * 70)
print("PROJECT EXECUTION SUMMARY")
print("=" * 70)
print(f"✓ Signal generation: {signals.shape[0]} signals created")
print(f"✓ Data split: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
print(f"✓ Feature-based classifier: {test_acc:.2%} accuracy")
if pytorch_available:
    print(f"✓ PyTorch model: Ready ({total_params:,} parameters)")
print(f"✓ Visualizations saved to: evaluation/")
print("\n" + "=" * 70)

print("\n📝 NEXT STEPS:")
print("  1. Run full training: python train.py --config config.yaml")
print("  2. Evaluate model: python evaluate.py --model models/best_model.pt")
print("  3. Tune hyperparameters: python hyperparameter_tuning.py --n-trials 50")
print("  4. View notebook: jupyter notebook notebooks/demo.ipynb")
print("  5. MLflow UI: mlflow ui --backend-store-uri file:./logs/mlflow")
print("\n" + "=" * 70)
