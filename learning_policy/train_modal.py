"""
Chess policy training - runs locally or on Modal.

Usage:
    modal run train_modal.py
    python train_modal.py --max-seconds 300
"""
import modal

app = modal.App("chess-policy")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy", "python-chess")
    .add_local_file("data.py", "/root/data.py")
    .add_local_file("features.py", "/root/features.py")
    .add_local_file("mlp_model.py", "/root/mlp_model.py")
)

data_volume = modal.Volume.from_name("chess-policy-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("chess-policy-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/root/cache": data_volume, "/root/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train(
    hidden_size: int = 256,
    batch_size: int = 256,
    lr: float = 0.001,
    max_seconds: int = 1200,
    eval_interval_seconds: int = 60,
    eval_iters: int = 50,
    run_name: str = "precomputed2",
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 0,
):
    import sys
    sys.path.insert(0, "/root")

    import os
    import time
    from itertools import cycle

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import wandb

    from data import get_dataloaders
    from mlp_model import SimplePolicyMLP
    from features import TOTAL_FEATURES

    is_modal = bool(os.environ.get("MODAL_TASK_ID"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"[{run_name}] Using device: {device}")

    if run_name is None:
        run_name = f"h{hidden_size}"

    print("=" * 60)
    print(f"[{run_name}] Policy Network Training")
    print("=" * 60)

    # Load data from precomputed features
    cache_dir = "/root/cache" if is_modal else None
    train_loader, eval_loader, vocab = get_dataloaders(batch_size, cache_dir=cache_dir, num_workers=num_workers)
    num_moves = len(vocab)
    total_samples = train_loader.num_samples
    print(f"[{run_name}] Training on {total_samples:,} samples")

    # Create model
    model = SimplePolicyMLP(input_size=TOTAL_FEATURES, num_moves=num_moves, hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    eval_iter = cycle(eval_loader)

    @torch.no_grad()
    def estimate_eval_loss():
        model.eval()
        losses = []
        correct = 0
        total = 0
        for _ in range(eval_iters):
            X_batch, y_batch = next(eval_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            losses.append(criterion(logits, y_batch).item())
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
        model.train()
        return {"eval_loss": sum(losses) / len(losses), "eval_acc": correct / total}

    num_params = sum(p.numel() for p in model.parameters())

    print(f"[{run_name}] Model: {num_params:,} parameters")
    print(f"[{run_name}] Hidden size: {hidden_size}")
    print(f"[{run_name}] Batch size: {batch_size}")
    print(f"[{run_name}] Learning rate: {lr}")
    print(f"[{run_name}] Max time: {max_seconds}s")

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}.pt")

    step = 0
    elapsed_before = 0.0
    wandb_run_id = None

    if os.path.exists(checkpoint_path):
        print(f"[{run_name}] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        elapsed_before = ckpt["elapsed_seconds"]
        wandb_run_id = ckpt.get("wandb_run_id")
        print(f"[{run_name}] Resumed from step {step}, {elapsed_before:.0f}s already elapsed")

    wandb.init(
        project="chess-policy",
        name=run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seconds": max_seconds,
            "num_params": num_params,
            "num_moves": num_moves,
            "total_samples": total_samples,
        },
    )
    wandb_run_id = wandb.run.id

    print(f"[{run_name}] " + "-" * 70)
    print(f"[{run_name}] {'Time':>6} | {'Samples':>12} | {'Step':>7} | {'Train Loss':>10} | {'Train Acc':>9} | {'Eval Acc':>9}")
    print(f"[{run_name}] " + "-" * 70)

    start_time = time.time()
    last_eval_time = start_time
    samples_seen = step * batch_size
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    total_elapsed = elapsed_before

    # Timing accumulators
    time_data = 0.0
    time_transfer = 0.0
    time_train = 0.0
    time_eval = 0.0

    def save_checkpoint():
        tmp_path = checkpoint_path + ".tmp"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "elapsed_seconds": total_elapsed,
            "wandb_run_id": wandb_run_id,
        }, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        if is_modal:
            checkpoint_volume.commit()

    model.train()
    train_iter = cycle(train_loader)

    while True:
        elapsed_this_run = time.time() - start_time
        total_elapsed = elapsed_before + elapsed_this_run
        if total_elapsed >= max_seconds:
            break

        t0 = time.time()
        X_batch, y_batch = next(train_iter)
        time_data += time.time() - t0

        t0 = time.time()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        time_transfer += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        time_train += time.time() - t0

        step += 1
        samples_seen += len(X_batch)
        train_loss_sum += loss.item()
        with torch.no_grad():
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        train_total += len(y_batch)

        if time.time() - last_eval_time >= eval_interval_seconds:
            last_eval_time = time.time()
            t0 = time.time()
            eval_metrics = estimate_eval_loss()
            time_eval += time.time() - t0
            total_elapsed = elapsed_before + (time.time() - start_time)

            train_loss = train_loss_sum / (train_total / batch_size)
            train_acc = train_correct / train_total

            time_total = time_data + time_transfer + time_train + time_eval
            pct_data = 100 * time_data / time_total
            pct_transfer = 100 * time_transfer / time_total
            pct_train = 100 * time_train / time_total
            pct_eval = 100 * time_eval / time_total

            print(
                f"[{run_name}] {total_elapsed:>5.0f}s | {samples_seen:>12,} | {step:>7,} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
                f"{eval_metrics['eval_acc']:>8.2%} | data {pct_data:.0f}% xfer {pct_transfer:.0f}% train {pct_train:.0f}% eval {pct_eval:.0f}%"
            )

            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                **eval_metrics,
                "samples_seen": samples_seen,
                "pct_data": pct_data,
                "pct_transfer": pct_transfer,
                "pct_train": pct_train,
                "pct_eval": pct_eval,
            }, step=step)

            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            save_checkpoint()

    eval_metrics = estimate_eval_loss()
    total_elapsed = elapsed_before + (time.time() - start_time)

    print("-" * 80)
    print(f"[{run_name}] Final eval_acc: {eval_metrics['eval_acc']:.2%}")

    save_checkpoint()
    wandb.finish()
    print(f"[{run_name}] Training complete. {step:,} steps in {total_elapsed:.0f}s")

    return {
        "run_name": run_name,
        "final_eval_acc": eval_metrics["eval_acc"],
        "num_params": num_params,
        "total_steps": step,
    }


@app.function(image=image, timeout=1800)
def sweep():
    """Run parallel training with different configurations."""
    configs = [
        {"hidden_size": 2048, "run_name": "precomputed"}
    ]

    def make_run_name(cfg: dict) -> str:
        parts = [f"{k[0]}{v}" for k, v in sorted(cfg.items()) if k != "run_name"]
        return "_".join(parts) or "default"

    for cfg in configs:
        if "run_name" not in cfg:
            cfg["run_name"] = make_run_name(cfg)

    print(f"Launching {len(configs)} parallel runs:")
    for cfg in configs:
        print(f"  {cfg['run_name']}: {cfg}")

    handles = [(cfg["run_name"], train.spawn(**cfg)) for cfg in configs]

    results = []
    failed = []
    for name, handle in handles:
        try:
            results.append(handle.get())
        except Exception as e:
            failed.append((name, e))

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)

    if failed:
        print(f"\nFAILED ({len(failed)}/{len(configs)}):")
        for name, exc in failed:
            print(f"  {name:>20}: {type(exc).__name__}: {exc}")

    if results:
        print(f"\nSUCCEEDED ({len(results)}/{len(configs)}):")
        for r in sorted(results, key=lambda x: x["final_eval_acc"], reverse=True):
            print(f"  {r['run_name']:>20}: {r['final_eval_acc']:.2%} ({r['num_params']:,} params)")

    return results


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"Result: {result}")


if __name__ == "__main__":
    import fire
    fire.Fire(train.local)
