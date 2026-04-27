"""
Hyperparameter tuning script using Optuna.

Usage:
    python hyperparameter_tuning.py --config config.yaml --n-trials 50
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
from optuna.trial import Trial
import mlflow
import mlflow.pytorch

from src.signal_generator import UltrasonicSignalGenerator
from src.model import Conv1DClassifier


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_data(config: dict):
    """Generate dataset."""
    print("Generating synthetic ultrasonic signals for tuning...")
    
    generator = UltrasonicSignalGenerator(
        signal_length=config['signal_length'],
        noise_std=config['noise_std'],
        defect_amplitude=config['defect_amplitude'],
        seed=config['random_seed']
    )
    
    signals, labels = generator.generate_dataset(
        n_defect=config['n_defect_samples'],
        n_no_defect=config['n_no_defect_samples']
    )
    
    # Split into train and val (use val as test during tuning for faster iteration)
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels,
        test_size=0.3,
        random_state=config['random_seed'],
        stratify=labels
    )
    
    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).unsqueeze(1)
    X_val = torch.from_numpy(X_val).unsqueeze(1)
    
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val)


class ObjectiveFunction:
    """Objective function for Optuna trials."""
    
    def __init__(self, train_data, val_data, config: dict, device, base_config: dict):
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.config = config
        self.device = device
        self.base_config = base_config
    
    def __call__(self, trial: Trial) -> float:
        """Execute a single trial."""
        
        # Define hyperparameters to tune
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        num_filters = trial.suggest_int('num_filters', 16, 128, step=16)
        kernel_size = trial.suggest_int('kernel_size', 3, 11, step=2)
        depth = trial.suggest_int('depth', 2, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
        batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
        
        # Log hyperparameters
        trial.set_user_attr('learning_rate', learning_rate)
        trial.set_user_attr('num_filters', num_filters)
        trial.set_user_attr('kernel_size', kernel_size)
        trial.set_user_attr('depth', depth)
        trial.set_user_attr('dropout_rate', dropout_rate)
        trial.set_user_attr('batch_size', batch_size)
        
        try:
            # Create model
            model = Conv1DClassifier(
                input_length=self.config['signal_length'],
                num_filters=num_filters,
                kernel_size=kernel_size,
                depth=depth,
                dropout_rate=dropout_rate,
                num_classes=2
            ).to(self.device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Create dataloaders
            train_dataset = TensorDataset(self.X_train, self.y_train)
            val_dataset = TensorDataset(self.X_val, self.y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.base_config['tuning_epochs']):
                # Training
                model.train()
                for signals, labels in train_loader:
                    signals, labels = signals.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for signals, labels in val_loader:
                        signals, labels = signals.to(self.device), labels.to(self.device)
                        
                        outputs = model(signals)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break
                
                # Report to Optuna for pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
        
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')


def run_tuning(config: dict, n_trials: int = 50, output_dir: str = 'logs/tuning'):
    """Run hyperparameter tuning."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    train_data, val_data = generate_data(config)
    
    # Create study
    study_name = 'ultrasonic-classifier-tuning'
    storage_path = os.path.join(output_dir, 'optuna_study.db')
    storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path}')
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create objective function
    objective = ObjectiveFunction(train_data, val_data, config, device, config)
    
    # Optimize
    print(f"\nStarting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("Optimization Results")
    print("=" * 50)
    
    best_trial = study.best_trial
    
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    import json
    best_params = {
        'trial_number': best_trial.number,
        'validation_loss': best_trial.value,
        'hyperparameters': best_trial.params,
        'user_attrs': best_trial.user_attrs
    }
    
    with open(os.path.join(output_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest hyperparameters saved to {output_dir}/best_hyperparameters.json")
    
    # Log to MLflow
    mlflow.set_experiment('ultrasonic-classifier-tuning')
    
    with mlflow.start_run():
        mlflow.log_params(best_trial.params)
        mlflow.log_metric('best_validation_loss', best_trial.value)
        mlflow.log_artifact(os.path.join(output_dir, 'best_hyperparameters.json'))
    
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    
    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, 'trials_history.csv'), index=False)
    print(f"Trials history saved to {output_dir}/trials_history.csv")
    
    return best_trial.params


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning using Optuna')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials')
    parser.add_argument('--output-dir', type=str, default='logs/tuning',
                        help='Directory to save tuning results')
    args = parser.parse_args()
    
    config = load_config(args.config)
    best_params = run_tuning(config, n_trials=args.n_trials, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
