"""
Training script for ultrasonic signal classifier.

Usage:
    python train.py --config config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

from src.signal_generator import UltrasonicSignalGenerator
from src.model import Conv1DClassifier


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict) -> None:
    """Setup MLflow experiment."""
    mlflow.set_experiment(config.get('mlflow_experiment_name', 'ultrasonic-classifier'))
    mlflow.set_tracking_uri(config.get('mlflow_tracking_uri', 'file:./logs/mlflow'))


def generate_or_load_data(config: dict):
    """Generate synthetic dataset."""
    print("Generating synthetic ultrasonic signals...")
    
    generator = UltrasonicSignalGenerator(
        signal_length=config['signal_length'],
        noise_std=config['noise_std'],
        defect_amplitude=config['defect_amplitude']
    )
    
    signals, labels = generator.generate_dataset(
        n_defect=config['n_defect_samples'],
        n_no_defect=config['n_no_defect_samples']
    )
    
    # Split into train, val, test
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
    
    # Convert to torch tensors and add channel dimension
    X_train = torch.from_numpy(X_train).unsqueeze(1)
    X_val = torch.from_numpy(X_val).unsqueeze(1)
    X_test = torch.from_numpy(X_test).unsqueeze(1)
    
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataloaders(train_data, val_data, test_data, batch_size: int):
    """Create PyTorch data loaders."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (signals, labels) in enumerate(train_loader):
        signals, labels = signals.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train ultrasonic signal classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    train_data, val_data, test_data = generate_or_load_data(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=config['batch_size']
    )
    
    # Create model
    model = Conv1DClassifier(
        input_length=config['signal_length'],
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        depth=config['depth'],
        dropout_rate=config['dropout_rate'],
        num_classes=2
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Start MLflow run
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config)
        mlflow.log_param('device', str(device))
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\nStarting training...")
        for epoch in range(config['epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_acc, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            
            print(f"Epoch {epoch + 1}/{config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                os.makedirs(args.model_dir, exist_ok=True)
                model_path = os.path.join(args.model_dir, 'best_model.pt')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Evaluate on test set
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_acc)
        
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nModel saved in {args.model_dir}/")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    main()
