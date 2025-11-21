import itertools
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import copy

import os
os.mkdirs("./results/", exist_ok=True)

class HyperparameterSearch:
    """
    A class to systematically search hyperparameters for CNN training.
    """

    def __init__(self, train_dataset, val_dataset, test_dataset, device):
        """
        Initialize the search with datasets and device.

        Args:
            train_dataset: PyTorch Dataset for training
            val_dataset: PyTorch Dataset for validation
            test_dataset: PyTorch Dataset for testing
            device: torch.device for training
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.results = []

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epochs: int,
        seed: 42,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train a model and return training metrics.

        Returns:
            Dictionary containing training history and best validation accuracy
        """
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if scheduler is not None:
                scheduler.step()

            train_loss = running_loss / total
            train_acc = correct / total
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss /= total
            val_acc = correct / total
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            if verbose and epoch % 5 == 0:
                print(f"  Epoch [{epoch}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return {
            'train_loss_history': train_loss_history,
            'train_acc_history': train_acc_history,
            'val_loss_history': val_loss_history,
            'val_acc_history': val_acc_history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_acc': train_acc_history[-1],
            'final_val_acc': val_acc_history[-1]
        }

    def search_training_hyperparameters(
        self,
        param_grid: Dict[str, List],
        model_config: Dict[str, Any],
        max_combinations: int = None,
        verbose: bool = True,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Search over training hyperparameters (batch_size, epochs, learning_rate, etc.)
        while keeping model architecture fixed.

        Args:
            param_grid: Dictionary with lists of values for each hyperparameter
                       e.g., {'batch_size': [16, 32], 'epochs': [30, 50],
                              'learning_rate': [0.001, 0.0001],
                              'step_size': [10, 20], 'gamma': [0.5, 0.7]}
            model_config: Fixed model architecture configuration
            max_combinations: Maximum number of combinations to try (None = all)
            verbose: Print progress
            save_results: Save results to CSV

        Returns:
            DataFrame with results for each combination
        """
        print("=" * 80)
        print("SEARCHING TRAINING HYPERPARAMETERS")
        print("=" * 80)
        print(f"Model config (fixed): {model_config}")
        print(f"Training param grid: {param_grid}")
        print()

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        if max_combinations and len(combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations out of {len(combinations)}")
            combinations = combinations[:max_combinations]

        print(f"Total combinations to try: {len(combinations)}\n")

        results = []

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            if verbose:
                print(f"\n[{i}/{len(combinations)}] Testing: {params}")

            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=params.get('batch_size', 32),
                shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=params.get('batch_size', 32),
                shuffle=False
            )

            # Create model
            from CNN_module import CNN, ACTIVATIONS  # Import here to avoid circular dependency
            model = CNN(**model_config)
            model.to(self.device)

            # print(model)

            # Create optimizer and scheduler
            optimizer = optim.Adam(
                model.parameters(),
                lr=params.get('learning_rate', 0.001)
            )
            scheduler = StepLR(
                optimizer,
                step_size=params.get('step_size', 10),
                gamma=params.get('gamma', 0.5)
            )

            criterion = nn.CrossEntropyLoss()

            # Train model
            try:
                metrics = self.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=params.get('epochs', 50),
                    seed=params.get('seed', 42),
                    verbose=verbose
                )

                # Store results
                result = {**params, **metrics}
                result['model_config'] = str(model_config)
                result['success'] = True
                results.append(result)

                if verbose:
                    print(f"  Best Val Acc: {metrics['best_val_acc']:.4f} "
                          f"(epoch {metrics['best_epoch']})")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                result = {**params, 'success': False, 'error': str(e)}
                results.append(result)

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Sort by best validation accuracy
        if 'best_val_acc' in df_results.columns:
            df_results = df_results.sort_values('best_val_acc', ascending=False)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./results/training_hyperparam_search_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")

        return df_results

    def search_architecture_hyperparameters(
        self,
        arch_param_grid: Dict[str, List],
        training_config: Dict[str, Any],
        max_combinations: int = None,
        verbose: bool = True,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Search over architecture hyperparameters while keeping training config fixed.

        Args:
            arch_param_grid: Dictionary with lists of architecture parameters
                            e.g., {'conv_channels': [[32, 64], [64, 128, 256]],
                                   'kernel_sizes': [3, 5],
                                   'activation': ['relu', 'gelu'],
                                   'fc_layer_sizes': [[128, 15], [256, 15]]}
            training_config: Fixed training configuration
                           e.g., {'batch_size': 32, 'epochs': 30,
                                  'learning_rate': 0.001, 'step_size': 10, 'gamma': 0.5}
            max_combinations: Maximum number of combinations to try
            verbose: Print progress
            save_results: Save results to CSV

        Returns:
            DataFrame with results for each combination
        """
        print("=" * 80)
        print("SEARCHING ARCHITECTURE HYPERPARAMETERS")
        print("=" * 80)
        print(f"Training config (fixed): {training_config}")
        print(f"Architecture param grid: {arch_param_grid}")
        print()

        # Generate all combinations
        param_names = list(arch_param_grid.keys())
        param_values = list(arch_param_grid.values())
        combinations = list(itertools.product(*param_values))

        if max_combinations and len(combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations out of {len(combinations)}")
            combinations = combinations[:max_combinations]

        print(f"Total combinations to try: {len(combinations)}\n")

        results = []

        for i, combo in enumerate(combinations, 1):
            arch_params = dict(zip(param_names, combo))

            if verbose:
                print(f"\n[{i}/{len(combinations)}] Testing architecture: {arch_params}")

            # Merge with base architecture config
            model_config = {
                'input_channels': 1,
                'input_size': (64, 64),
                'use_padding': True,
                'padding_size': 1,
                'use_maxpooling': True,
                'use_maxpooling_every': 1,
                **arch_params
            }

            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=training_config.get('batch_size', 32),
                shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=training_config.get('batch_size', 32),
                shuffle=False
            )

            # Create model
            try:
                from CNN_module import CNN, ACTIVATIONS
                model = CNN(**model_config)
                model.to(self.device)

                # Create optimizer and scheduler
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=training_config.get('learning_rate', 0.001)
                )
                scheduler = StepLR(
                    optimizer,
                    step_size=training_config.get('step_size', 10),
                    gamma=training_config.get('gamma', 0.5)
                )

                criterion = nn.CrossEntropyLoss()

                # Train model
                metrics = self.train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=training_config.get('epochs', 50),
                    seed=training_config.get('seed', 42),
                    verbose=verbose
                )

                # Store results
                result = {**arch_params, **metrics}
                result['training_config'] = str(training_config)
                result['success'] = True
                results.append(result)

                if verbose:
                    print(f"  Best Val Acc: {metrics['best_val_acc']:.4f} "
                          f"(epoch {metrics['best_epoch']})")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                result = {**arch_params, 'success': False, 'error': str(e)}
                results.append(result)

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Sort by best validation accuracy
        if 'best_val_acc' in df_results.columns:
            df_results = df_results.sort_values('best_val_acc', ascending=False)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./results/architecture_hyperparam_search_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")

        return df_results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_training_hyperparameter_search():
    """
    Example: Search training hyperparameters with fixed architecture.
    """
    from datasets import train_dataset, val_dataset, test_dataset, device

    # Initialize search
    search = HyperparameterSearch(train_dataset, val_dataset, test_dataset, device)

    # Define fixed model architecture
    model_config = {
        'input_channels': 1,
        'conv_channels': [64, 128, 256],
        'kernel_sizes': 5,
        'use_padding': True,
        'padding_size': 1,
        'use_maxpooling': True,
        'use_maxpooling_every': 1,
        'fc_layer_sizes': [256, 15],
        'input_size': (64, 64),
        'activation': 'relu',
        'use_batchnorm': False,
        'dropout': 0.2,
    }

    # Define training hyperparameter grid
    training_param_grid = {
        'batch_size': [32,],
        'epochs': [25, ], # [30, 50],
        'learning_rate': [0.001, 0.0005],
        'step_size': [10, ],
        'gamma': [0.5, ],
        'seed': [42, 133301, 1, 39]
    }

    # Run search
    results = search.search_training_hyperparameters(
        param_grid=training_param_grid,
        model_config=model_config,
        verbose=True,
        save_results=True
    )

    print("\n" + "=" * 80)
    print("TOP 5 RESULTS:")
    print("=" * 80)
    print(results.head()[['batch_size', 'learning_rate', 'step_size',
                          'gamma', 'best_val_acc', 'best_epoch']])

    return results


def example_architecture_hyperparameter_search():
    """
    Example: Search architecture hyperparameters with fixed training config.
    """
    from datasets import train_dataset, val_dataset, test_dataset, device

    # Initialize search
    search = HyperparameterSearch(train_dataset, val_dataset, test_dataset, device)

    # Define fixed training configuration
    training_config = {
        'batch_size': 32,
        'epochs': 30,  # Reduced for faster search
        'learning_rate': 0.001,
        'step_size': 10,
        'gamma': 0.5,
        'seed' : 42
    }

    # Define architecture hyperparameter grid
    arch_param_grid = {
        'conv_channels': [
            [16, 32],
            [32, 64],
            [8, 16, 32]
        ],
        'kernel_sizes': [3, 5],
        'activation': ['relu',],
        'fc_layer_sizes': [
            [64, 15],
        ],
        'dropout': [0.2, 0.4],
    }

    # Run search (limit combinations for demo)
    results = search.search_architecture_hyperparameters(
        arch_param_grid=arch_param_grid,
        training_config=training_config,
        max_combinations=10,  # Remove this to try all combinations
        verbose=True,
        save_results=True
    )

    print("\n" + "=" * 80)
    print("TOP 5 RESULTS:")
    print("=" * 80)
    print(results.head()[['conv_channels', 'kernel_sizes', 'activation',
                          'best_val_acc', 'best_epoch']])

    return results


if __name__ == "__main__":
    print("Hyperparameter Search")
    print("=" * 80)
    print("\nAvailable functions:")
    print("1. example_training_hyperparameter_search() - Search training hyperparameters")
    print("2. example_architecture_hyperparameter_search() - Search architecture hyperparameters")

    # Uncomment one of these to run:
    # results = example_training_hyperparameter_search()
    results = example_architecture_hyperparameter_search()
