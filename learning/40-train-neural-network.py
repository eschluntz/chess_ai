#!/usr/bin/env python

"""
Train a neural network model to evaluate chess positions.
"""

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import helper functions from the eval accuracy script
from eval_accuracy_helpers import (
    create_board_from_fen,
    should_skip_position,
    extract_centipawn_score
)
from feature_extraction import extract_features_piece_square


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions."""
    
    def __init__(self, features: np.ndarray, scores: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.scores = torch.FloatTensor(scores)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.scores[idx]


class ChessEvalNet(nn.Module):
    """Neural network for chess position evaluation."""
    
    def __init__(self, input_size: int = 779, hidden_sizes: list[int] = [512, 256, 128]):
        super(ChessEvalNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def load_neural_network_eval(model_path: str):
    """Load the trained neural network model and return an evaluation function."""
    import chess
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    input_size = data['input_size']
    hidden_sizes = data['hidden_sizes']
    
    # Recreate the model architecture
    net = ChessEvalNet(input_size=input_size, hidden_sizes=hidden_sizes)
    net.load_state_dict(model)
    net.eval()
    
    def neural_network_eval(board: chess.Board) -> tuple[int, bool]:
        """Evaluate a position using the trained neural network."""
        features = extract_features_piece_square(board)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            score = net(features_tensor).item()
        
        return int(score), False
    
    return neural_network_eval


def train_neural_network(num_train_samples: int = 50000, num_val_samples: int = 10000,
                        hidden_sizes: list[int] = [512, 256, 128],
                        batch_size: int = 256, epochs: int = 50,
                        learning_rate: float = 0.001, model_suffix: str = '',
                        plot_live: bool = True):
    """
    Train a neural network model on chess positions.
    
    Args:
        num_train_samples: Number of training samples to use
        num_val_samples: Number of validation samples to use
        hidden_sizes: List of hidden layer sizes
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_suffix: Additional suffix for model filename
        plot_live: Whether to show live training plots
    
    Returns:
        tuple: (model, metrics_dict) where metrics_dict contains train and validation errors
    """
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Skip first 10k (reserved for evaluation), then load training data
    print("\nData split:")
    print("  Positions 0-9,999: Reserved for evaluation (skipped)")
    print(f"  Positions 10,000-{10000+num_train_samples-1:,}: Training data")
    print(f"  Positions {10000+num_train_samples:,}-{10000+num_train_samples+num_val_samples-1:,}: Validation data")
    
    print(f"\nLoading {num_train_samples:,} training positions...")
    train_stream = ds["train"].skip(10000)
    train_data = list(train_stream.take(num_train_samples))
    
    # Process training data
    print("Processing training data...")
    train_df = pd.DataFrame(train_data)
    train_df = train_df.dropna(subset=['fen'])
    
    # Filter positions
    train_df['should_skip'] = train_df.apply(
        lambda row: should_skip_position(row),
        axis=1
    )
    train_df = train_df[~train_df['should_skip']].copy()
    
    # Extract scores and create boards
    train_df['true_score'] = train_df.apply(extract_centipawn_score, axis=1)
    train_df['board'] = train_df['fen'].apply(create_board_from_fen)
    
    print(f"Extracting features from {len(train_df)} training positions...")
    # Extract features using piece-square representation
    X_train = np.array([extract_features_piece_square(board) for board in train_df['board']])
    y_train = train_df['true_score'].values
    print(f"Feature shape: {X_train.shape}")
    
    # Load validation data
    print(f"\nLoading {num_val_samples} validation positions...")
    val_stream = ds["train"].skip(10000 + num_train_samples)
    val_data = list(val_stream.take(num_val_samples))
    
    # Process validation data
    print("Processing validation data...")
    val_df = pd.DataFrame(val_data)
    val_df = val_df.dropna(subset=['fen'])
    
    val_df['should_skip'] = val_df.apply(
        lambda row: should_skip_position(row),
        axis=1
    )
    val_df = val_df[~val_df['should_skip']].copy()
    
    val_df['true_score'] = val_df.apply(extract_centipawn_score, axis=1)
    val_df['board'] = val_df['fen'].apply(create_board_from_fen)
    
    print(f"Extracting features from {len(val_df)} validation positions...")
    X_val = np.array([extract_features_piece_square(board) for board in val_df['board']])
    y_val = val_df['true_score'].values
    
    # Create PyTorch datasets and dataloaders
    train_dataset = ChessDataset(X_train, y_train)
    val_dataset = ChessDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = ChessEvalNet(input_size=X_train.shape[1], hidden_sizes=hidden_sizes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"\nTraining Neural Network with:")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Initialize lists to store metrics for plotting
    train_losses = []
    train_rmses = []
    val_rmses = []
    val_maes = []
    
    # Set up the plot
    if plot_live:
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, scores in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            features, scores = features.to(device), scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_rmse = np.sqrt(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for features, scores in val_loader:
                features, scores = features.to(device), scores.to(device)
                outputs = model(features)
                loss = criterion(outputs, scores)
                
                val_loss += loss.item() * len(features)
                val_predictions.extend(outputs.cpu().numpy())
                val_true.extend(scores.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataset)
        val_rmse = np.sqrt(avg_val_loss)
        val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_true)))
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)
        
        print(f"Epoch {epoch+1}: Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}, Val MAE: {val_mae:.2f}")
        
        # Update plots
        if plot_live:
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Plot RMSE
            ax1.plot(range(1, epoch+2), train_rmses, 'b-', label='Train RMSE', linewidth=2)
            ax1.plot(range(1, epoch+2), val_rmses, 'r-', label='Val RMSE', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('RMSE (centipawns)')
            ax1.set_title('Training and Validation RMSE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MAE
            ax2.plot(range(1, epoch+2), val_maes, 'g-', label='Val MAE', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (centipawns)')
            ax2.set_title('Validation MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add current learning rate to the plot
            current_lr = optimizer.param_groups[0]['lr']
            fig.suptitle(f'Neural Network Training Progress - {model_suffix} (LR: {current_lr:.2e})', fontsize=14)
            
            plt.tight_layout()
            plt.pause(0.1)  # Brief pause to update the plot
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"  New best model (Val RMSE: {val_rmse:.2f})")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save final plot
    if plot_live:
        plt.ioff()  # Turn off interactive mode
        plot_filename = f"nn_training_progress_{num_train_samples}_{model_suffix}.png"
        plot_path = os.path.join(os.path.dirname(__file__), plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining plot saved to: {plot_path}")
        plt.show()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Training set
        train_predictions = []
        for features, scores in train_loader:
            features = features.to(device)
            outputs = model(features)
            train_predictions.extend(outputs.cpu().numpy())
        
        train_mae = np.mean(np.abs(np.array(train_predictions) - y_train))
        train_rmse = np.sqrt(np.mean((np.array(train_predictions) - y_train)**2))
        
        # Validation set
        val_predictions = []
        for features, scores in val_loader:
            features = features.to(device)
            outputs = model(features)
            val_predictions.extend(outputs.cpu().numpy())
        
        val_mae = np.mean(np.abs(np.array(val_predictions) - y_val))
        val_rmse = np.sqrt(np.mean((np.array(val_predictions) - y_val)**2))
    
    print("\nFinal evaluation:")
    print(f"  Training MAE: {train_mae:.2f} centipawns")
    print(f"  Training RMSE: {train_rmse:.2f} centipawns")
    print(f"  Validation MAE: {val_mae:.2f} centipawns")
    print(f"  Validation RMSE: {val_rmse:.2f} centipawns")
    
    # Save the model
    if model_suffix:
        model_filename = f"neural_network_chess_model_{num_train_samples}_{model_suffix}.pkl"
    else:
        model_filename = f"neural_network_chess_model_{num_train_samples}.pkl"
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    print(f"\nSaving model to {model_path}...")
    
    model_data = {
        'model': model.cpu().state_dict(),
        'input_size': X_train.shape[1],
        'hidden_sizes': hidden_sizes
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    
    # Return metrics
    metrics = {
        'num_train_samples': num_train_samples,
        'hidden_sizes': hidden_sizes,
        'train_mae': train_mae,
        'val_mae': val_mae
    }
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train neural network for chess evaluation')
    parser.add_argument('--no-plot', action='store_true', help='Disable live plotting')
    args = parser.parse_args()
    
    # Train with different architectures
    experiments = [
        # Small network
        # {'hidden_sizes': [256, 128], 'epochs': 30, 'model_suffix': 'small'},
        # Medium network
        # {'hidden_sizes': [512, 256, 128], 'epochs': 50, 'model_suffix': 'medium'},
        # # Large network
        {'hidden_sizes': [1024, 512, 256, 128], 'epochs': 50, 'model_suffix': 'large'},
    ]
    
    for exp in experiments:
        print("\n" + "="*80)
        print(f"Training {exp['model_suffix']} network")
        print("="*80)
        
        train_neural_network(
            num_train_samples=100_000,
            plot_live=not args.no_plot,
            **exp
        )