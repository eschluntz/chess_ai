"""
Chess policy training - runs locally or on Modal.

Usage:
    modal run --detach train_modal.py::train --num-samples 50M --max-seconds 3600
    modal run --detach train_modal.py::sweep
    python train_modal.py --num-samples 1M --max-seconds 300
"""

import modal

app = modal.App("chess-policy")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy", "einops")
    .add_local_file("data.py", "/root/data.py")
    .add_local_file("board_repr.py", "/root/board_repr.py")
    .add_local_file("mlp_model.py", "/root/mlp_model.py")
    .add_local_file("cnn_model.py", "/root/cnn_model.py")
)

data_volume = modal.Volume.from_name("chess-policy-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "chess-policy-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/root/cache": data_volume, "/root/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train(
    hidden_channels: int = 128,
    num_layers: int = 14,
    kernel_size: int = 3,
    batch_size: int = 1024,
    lr: float = 0.001,
    max_seconds: int = 14400,
    eval_interval_seconds: int = 30,
    run_name: str = None,
    checkpoint_dir: str = "checkpoints",
    num_samples: str = "50M",
):
    import sys

    sys.path.insert(0, "/root")

    import os
    import time

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import wandb

    # from mlp_model import PolicyMLP
    from cnn_model import PolicyCNN
    from data import get_dataloaders

    is_modal = bool(os.environ.get("MODAL_TASK_ID"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"[{run_name}] Using device: {device}")

    if run_name is None:
        run_name = f"L{num_layers}_H{hidden_channels}_K{kernel_size}_B{batch_size}_LR{lr}_S{num_samples}"

    print("=" * 60)
    print(f"[{run_name}] Policy Network Training")
    print("=" * 60)

    # Load data from precomputed features
    train_loader, eval_loader, num_classes = get_dataloaders(
        batch_size, num_samples=num_samples
    )
    total_samples = train_loader.num_samples
    num_moves = num_classes
    print(f"[{run_name}] Training on {total_samples:,} samples")

    # Create model
    model = PolicyCNN(
        num_moves=num_moves,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
    ).to(device)
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    @torch.no_grad()
    def estimate_eval_loss():
        model.eval()
        losses = []
        correct = 0
        total = 0
        for planes, meta, target in eval_loader:
            planes, meta, target = planes.to(device), meta.to(device), target.to(device)
            logits = model(planes, meta)
            losses.append(criterion(logits, target).item())
            correct += (logits.argmax(dim=1) == target.argmax(dim=1)).sum().item()
            total += len(target)
        model.train()
        if total == 0:
            return {"eval_loss": float("nan"), "eval_acc": float("nan")}
        return {"eval_loss": sum(losses) / len(losses), "eval_acc": correct / total}

    num_params = sum(p.numel() for p in model.parameters())

    print(f"[{run_name}] Model: {num_params:,} parameters")
    print(f"[{run_name}] Layers: {num_layers}, Hidden: {hidden_channels}")
    print(f"[{run_name}] Output classes: {num_moves}")
    print(f"[{run_name}] Batch size: {batch_size}, LR: {lr}")
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
        print(
            f"[{run_name}] Resumed from step {step}, {elapsed_before:.0f}s already elapsed"
        )

    wandb.init(
        project="chess-policy",
        name=run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        config={
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seconds": max_seconds,
            "num_params": num_params,
            "num_moves": num_moves,
            "total_samples": total_samples,
        },
    )
    wandb_run_id = wandb.run.id

    print(f"[{run_name}] " + "-" * 80)
    print(
        f"[{run_name}] {'Time':>6} | {'Epoch':>5} | {'Samples':>12} | {'Step':>7} | {'Loss':>7} | {'Train':>6} | {'Eval':>6}"
    )
    print(f"[{run_name}] " + "-" * 80)

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
    time_forward = 0.0
    time_backward = 0.0
    time_optim = 0.0
    time_eval = 0.0

    def save_checkpoint():
        tmp_path = checkpoint_path + ".tmp"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "elapsed_seconds": total_elapsed,
                "wandb_run_id": wandb_run_id,
            },
            tmp_path,
        )
        os.replace(tmp_path, checkpoint_path)
        if is_modal:
            checkpoint_volume.commit()

    def infinite_batches(loader):
        """Yield batches forever, restarting when exhausted. Tracks epoch."""
        epoch = 0
        while True:
            for batch in loader:
                yield epoch, batch
            epoch += 1

    model.train()
    train_iter = infinite_batches(train_loader)

    while True:
        elapsed_this_run = time.time() - start_time
        total_elapsed = elapsed_before + elapsed_this_run
        if total_elapsed >= max_seconds:
            break

        t0 = time.time()
        epoch, (planes, meta, target) = next(train_iter)
        time_data += time.time() - t0

        t0 = time.time()
        planes, meta, target = planes.to(device), meta.to(device), target.to(device)
        time_transfer += time.time() - t0

        optimizer.zero_grad()

        t0 = time.time()
        logits = model(planes, meta)
        loss = criterion(logits, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_forward += time.time() - t0

        t0 = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_backward += time.time() - t0

        t0 = time.time()
        optimizer.step()
        time_optim += time.time() - t0

        step += 1
        samples_seen += len(target)
        train_loss_sum += loss.item()
        with torch.no_grad():
            train_correct += (logits.argmax(dim=1) == target.argmax(dim=1)).sum().item()
        train_total += len(target)

        if time.time() - last_eval_time >= eval_interval_seconds:
            last_eval_time = time.time()
            t0 = time.time()
            eval_metrics = estimate_eval_loss()
            time_eval += time.time() - t0
            total_elapsed = elapsed_before + (time.time() - start_time)

            train_loss = train_loss_sum / (train_total / batch_size)
            train_acc = train_correct / train_total

            time_total = (
                time_data
                + time_transfer
                + time_forward
                + time_backward
                + time_optim
                + time_eval
            )
            pct_data = 100 * time_data / time_total
            pct_transfer = 100 * time_transfer / time_total
            pct_forward = 100 * time_forward / time_total
            pct_backward = 100 * time_backward / time_total
            pct_optim = 100 * time_optim / time_total

            # Dataloader breakdown
            dl_total = (
                train_loader.time_mmap
                + train_loader.time_tensor
                + train_loader.time_labels
            )
            if dl_total > 0:
                pct_mmap = 100 * train_loader.time_mmap / dl_total
                pct_tensor = 100 * train_loader.time_tensor / dl_total
                pct_labels = 100 * train_loader.time_labels / dl_total
                dl_breakdown = f"[mmap {pct_mmap:.0f}% tensor {pct_tensor:.0f}% labels {pct_labels:.0f}%]"
            else:
                dl_breakdown = ""

            print(
                f"[{run_name}] {total_elapsed:>5.0f}s | {epoch:>5} | {samples_seen:>12,} | {step:>7,} | {train_loss:>7.3f} | {train_acc:>5.1%} | {eval_metrics['eval_acc']:>5.1%} | "
                f"data {pct_data:.0f}% xfer {pct_transfer:.0f}% fwd {pct_forward:.0f}% bwd {pct_backward:.0f}% opt {pct_optim:.0f}%"
            )
            print(f"[{run_name}]   dataloader: {dl_breakdown}")

            wandb.log(
                {
                    "elapsed_t": total_elapsed,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    **eval_metrics,
                    "samples_seen": samples_seen,
                    "pct_data": pct_data,
                    "pct_transfer": pct_transfer,
                    "pct_forward": pct_forward,
                    "pct_backward": pct_backward,
                    "pct_optim": pct_optim,
                },
                step=step,
            )

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
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
        "total_steps": step,
    }


@app.function(image=image, timeout=86000)  # 23.5 hours for sweep coordination
def sweep():
    """24hr scaling sweep: 10M to 100M params on A10 GPU."""
    # K=3, L=14 fixed (wide architecture from previous sweeps)
    # Scale via hidden_channels to hit param targets
    # Params ≈ 126,000*H + 252*H²
    configs = [
        {"hidden_channels": 72, "run_name": "scale_10M"},  # ~10M
        {"hidden_channels": 128, "run_name": "scale_20M"},  # ~20M
        {"hidden_channels": 176, "run_name": "scale_30M"},  # ~30M
        {"hidden_channels": 264, "run_name": "scale_50M"},  # ~51M
        {"hidden_channels": 432, "run_name": "scale_100M"},  # ~101M
    ]

    # Add common settings
    for cfg in configs:
        cfg["kernel_size"] = 3
        cfg["num_layers"] = 14
        cfg["batch_size"] = 1024
        cfg["max_seconds"] = 84600  # 23.5 hours (buffer for Modal's 24hr limit)

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
        print(f"  {'Name':>10} | {'Acc':>7} | {'Params':>12}")
        print(f"  {'-' * 10}-+-{'-' * 7}-+-{'-' * 12}")
        for r in sorted(results, key=lambda x: x["final_eval_acc"], reverse=True):
            print(
                f"  {r['run_name']:>10} | {r['final_eval_acc']:>6.2%} | {r['num_params']:>12,}"
            )

    return results


if __name__ == "__main__":
    import fire

    fire.Fire(train.local)
