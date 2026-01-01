#!/usr/bin/env python
"""
Policy network training with time-based loop and wandb logging.

Run from learning_policy directory:
    python 20_train_policy.py
"""

import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from data import build_move_vocabulary, get_eval_data, get_index_to_move, get_train_data
from features import extract_features_piece_square
from mlp_model import SimplePolicyMLP


def prepare_data(
    df, vocab: dict[str, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert DataFrame to feature and label tensors on device."""
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        board = chess.Board(row["fen"])
        feat = extract_features_piece_square(board)
        features.append(feat)
        labels.append(vocab[row["target_move"]])

    X = torch.tensor(np.array(features), dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return X, y


def main():
    # Hyperparameters
    num_train = 100_000
    hidden_size = 256
    batch_size = 256
    lr = 0.001
    max_seconds = 300  # 5 minutes
    eval_interval_seconds = 30
    eval_iters = 50  # batches for train loss estimate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Policy Network Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading {num_train:,} training samples...")
    train_df = get_train_data(num_train)
    print("Loading eval samples...")
    eval_df = get_eval_data()

    vocab = build_move_vocabulary()
    num_moves = len(vocab)
    print(f"Move vocabulary size: {num_moves}")

    print("\nPreparing training data...")
    X_train, y_train = prepare_data(train_df, vocab, device)
    print(f"Training features shape: {X_train.shape}")

    print("Preparing eval data...")
    X_eval, y_eval = prepare_data(eval_df, vocab, device)

    # Batch sampler
    def get_batch(split: str):
        X, y = (X_train, y_train) if split == "train" else (X_eval, y_eval)
        ix = torch.randint(len(X), (batch_size,))
        return X[ix], y[ix]

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "eval"):
            losses = torch.zeros(eval_iters)
            correct = 0
            total = 0
            for k in range(eval_iters):
                X_batch, y_batch = get_batch(split)
                logits = model(X_batch)
                losses[k] = criterion(logits, y_batch).item()
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)
            out[f"{split}_loss"] = losses.mean().item()
            out[f"{split}_acc"] = correct / total
        model.train()
        return out

    # Create model
    print(f"\nDevice: {device}")

    model = SimplePolicyMLP(
        input_size=X_train.shape[1],
        num_moves=num_moves,
        hidden_size=hidden_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel: {num_params:,} parameters")
    print(f"Hidden size: {hidden_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Max time: {max_seconds}s")

    # Wandb setup
    wandb.init(
        project="chess-policy",
        config={
            "num_train": num_train,
            "num_eval": len(X_eval),
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seconds": max_seconds,
            "num_params": num_params,
            "num_moves": num_moves,
        },
    )

    # Checkpoint directory
    checkpoint_dir = Path("checkpoints") / wandb.run.id
    checkpoint_dir.mkdir(parents=True)

    # Training loop
    print("\n" + "-" * 70)
    print(
        f"{'Time':>6} | {'Step':>7} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Eval Loss':>10} | {'Eval Acc':>9}"
    )
    print("-" * 70)

    start_time = time.time()
    last_eval_time = start_time
    step = 0

    model.train()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            break

        # Training step
        X_batch, y_batch = get_batch("train")
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        step += 1

        # Periodic eval
        if time.time() - last_eval_time >= eval_interval_seconds:
            last_eval_time = time.time()
            metrics = estimate_loss()
            elapsed = time.time() - start_time

            print(
                f"{elapsed:>5.0f}s | {step:>7,} | {metrics['train_loss']:>10.4f} | {metrics['train_acc']:>8.2%} | "
                f"{metrics['eval_loss']:>10.4f} | {metrics['eval_acc']:>8.2%}"
            )

            wandb.log({
                **metrics,
                "step": step,
                "elapsed_seconds": elapsed,
            })

            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_dir / "latest.pt")

    # Final eval
    metrics = estimate_loss()
    elapsed = time.time() - start_time

    print("-" * 70)
    print(
        f"{elapsed:>5.0f}s | {step:>7,} | {metrics['train_loss']:>10.4f} | {metrics['train_acc']:>8.2%} | "
        f"{metrics['eval_loss']:>10.4f} | {metrics['eval_acc']:>8.2%}"
    )

    wandb.log({
        **metrics,
        "step": step,
        "elapsed_seconds": elapsed,
    })

    # Save final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_dir / "latest.pt")
    print(f"Checkpoint saved to {checkpoint_dir / 'latest.pt'}")

    # Show some predictions
    print("\nSample predictions:")
    idx_to_move = get_index_to_move()
    model.eval()

    with torch.no_grad():
        logits = model(X_eval[:5])
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        for i in range(5):
            true_move = idx_to_move[y_eval[i].item()]
            pred_move = idx_to_move[preds[i].item()]
            confidence = probs[i, preds[i]].item()
            match = "✓" if true_move == pred_move else "✗"
            print(f"  {match} True: {true_move:6s} | Pred: {pred_move:6s} ({confidence:.1%})")

    wandb.finish()
    print(f"\nTraining complete. {step:,} steps in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
