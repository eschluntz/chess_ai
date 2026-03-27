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
    .add_local_file("transformer_model.py", "/root/transformer_model.py")
)

data_volume = modal.Volume.from_name("chess-policy-output", create_if_missing=True)
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
    n_embed: int = 256,
    num_heads: int = 8,
    ff_ratio: int = 4,
    num_layers: int = 8,
    head_type: str = "spatial_unshared",
    pos_encoding: str = "shaw2d",
    batch_size: int = 2048,
    lr: float = 4.24e-4,
    max_seconds: int = 0,
    max_steps: int = 0,
    eval_interval_seconds: int = 120,
    run_name: str = None,
    checkpoint_dir: str = "checkpoints",
    num_samples: str = "full",
    min_depth: int = 0,
    weight_decay: float = 0.01,
    cosine_decay: bool = False,
    grad_clip: float = 1.0,
    grad_skip_threshold: float = 20.0,
    warmup_steps: int = 250,
):
    import sys

    sys.path.insert(0, "/root")

    import os
    import time

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import wandb

    from transformer_model import Transformer, TransformerConfig
    from data import get_dataloaders

    is_modal = bool(os.environ.get("MODAL_TASK_ID"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    print(f"[{run_name}] Using device: {device}")

    if run_name is None:
        run_name = f"tf_L{num_layers}_D{n_embed}_H{num_heads}_B{batch_size}_LR{lr}_S{num_samples}"

    print("=" * 60)
    print(f"[{run_name}] Policy Network Training")
    print("=" * 60)

    # Load data from precomputed features (flip always on — Transformer bakes it in)
    train_loader, eval_loader, eval_loader_deep, num_classes, vocab_moves = get_dataloaders(
        batch_size, num_samples=num_samples, min_depth=min_depth, flip_board=True
    )
    total_samples = train_loader.num_samples
    num_moves = num_classes
    vocab_t = torch.as_tensor(vocab_moves, dtype=torch.long)
    print(f"[{run_name}] Training on {total_samples:,} samples")

    # Create model
    cfg = TransformerConfig(
        n_embed=n_embed,
        num_heads=num_heads,
        ff_ratio=ff_ratio,
        num_layers=num_layers,
        num_moves=num_moves,
        head_type=head_type,
        pos_encoding=pos_encoding,
    )
    model = Transformer(cfg, vocab=vocab_t).to(device)
    flops_per_sample = cfg.flops_per_sample()
    model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR schedule: cosine decay based on time (wall-clock) or steps
    if cosine_decay and max_steps:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    else:
        scheduler = None  # time-based cosine handled manually in training loop

    # Mixed precision (bf16)
    _amp_dtype = torch.bfloat16

    @torch.no_grad()
    def run_eval(loader):
        model.eval()
        losses = []
        correct = 0
        total = 0
        for planes, meta, target in loader:
            planes, meta, target = planes.to(device), meta.to(device), target.to(device)
            with torch.amp.autocast(
                device_type="cuda", dtype=_amp_dtype, enabled=_amp_dtype is not None
            ):
                logits = model(planes, meta)
                losses.append(criterion(logits, target).item())
            correct += (logits.argmax(dim=1) == target.argmax(dim=1)).sum().item()
            total += len(target)
        model.train()
        if total == 0:
            return float("nan"), float("nan")
        return sum(losses) / len(losses), correct / total

    def estimate_eval_loss():
        eval_loss, eval_acc = run_eval(eval_loader)
        _, deep_acc = run_eval(eval_loader_deep)
        return {"eval_loss": eval_loss, "eval_acc": eval_acc, "eval_deep_acc": deep_acc}

    num_params = sum(p.numel() for p in model.parameters())

    print(f"[{run_name}] Model: {num_params:,} parameters")
    print(f"[{run_name}] Layers: {num_layers}, d_model: {n_embed}, heads: {num_heads}, ff_ratio: {ff_ratio}")
    print(f"[{run_name}] Output classes: {num_moves}")
    print(f"[{run_name}] Batch size: {batch_size}, LR: {lr}")
    if max_steps:
        print(f"[{run_name}] Max steps: {max_steps}")
    if max_seconds:
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
            "n_embed": n_embed,
            "num_heads": num_heads,
            "ff_ratio": ff_ratio,
            "num_layers": num_layers,
            "head_type": head_type,
            "pos_encoding": pos_encoding,
            "flops_per_sample": flops_per_sample,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seconds": max_seconds,
            "num_params": num_params,
            "num_moves": num_moves,
            "total_samples": total_samples,
            "grad_clip": grad_clip,
            "grad_skip_threshold": grad_skip_threshold,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
        },
    )
    wandb_run_id = wandb.run.id

    print(f"[{run_name}] " + "-" * 80)
    print(
        f"[{run_name}] {'Time':>6} | {'Epoch':>5} | {'Samples':>12} | {'Step':>7} | {'Loss':>7} | {'Train':>6} | {'Eval':>6} | {'Deep':>6}"
    )
    print(f"[{run_name}] " + "-" * 80)

    start_time = time.time()
    last_eval_time = start_time
    samples_seen = step * batch_size
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
    grad_skips = 0
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

    def prefetch(iterator, n=2):
        """Pre-compute batches in a background thread to overlap CPU/GPU work."""
        from queue import Queue
        from threading import Thread

        q = Queue(maxsize=n)

        def fill():
            for item in iterator:
                q.put(item)
            q.put(None)

        Thread(target=fill, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    model.train()
    train_iter = prefetch(infinite_batches(train_loader))

    while True:
        elapsed_this_run = time.time() - start_time
        total_elapsed = elapsed_before + elapsed_this_run
        if max_steps and step >= max_steps:
            break
        if max_seconds and total_elapsed >= max_seconds:
            break

        t0 = time.time()
        epoch, (planes, meta, target) = next(train_iter)
        time_data += time.time() - t0

        t0 = time.time()
        planes, meta, target = planes.to(device), meta.to(device), target.to(device)
        time_transfer += time.time() - t0

        optimizer.zero_grad()

        t0 = time.time()
        with torch.amp.autocast(
            device_type="cuda", dtype=_amp_dtype, enabled=_amp_dtype is not None
        ):
            logits = model(planes, meta)
            loss = criterion(logits, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_forward += time.time() - t0

        t0 = time.time()
        loss.backward()
        # clip_grad_norm_ returns the pre-clip total norm — track it to catch
        # gradient spikes before they show up as loss explosions.
        if grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        else:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        gn = gn.item()
        grad_norm_sum += gn
        grad_norm_max = max(grad_norm_max, gn)
        grad_norm_count += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_backward += time.time() - t0

        t0 = time.time()
        # Skip pathological batches entirely — a clipped step in a garbage
        # direction can still destabilize the model (seen: gn=2038 at lr=1e-3).
        if grad_skip_threshold > 0 and gn > grad_skip_threshold:
            grad_skips += 1
            optimizer.zero_grad()
        else:
            optimizer.step()
        # LR schedule: warmup overrides everything for the first warmup_steps,
        # then fall through to cosine or constant.
        if warmup_steps > 0 and step < warmup_steps:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        elif scheduler:
            scheduler.step()
        elif cosine_decay and max_seconds:
            import math
            progress = min(total_elapsed / max_seconds, 1.0)
            new_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        else:
            # Constant LR: reset to base after warmup completes
            for pg in optimizer.param_groups:
                pg["lr"] = lr
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

            pct_eval = 100 * time_eval / time_total

            print(
                f"[{run_name}] {total_elapsed:>5.0f}s | {epoch:>5} | {samples_seen:>12,} | {step:>7,} | {train_loss:>7.3f} | {train_acc:>5.1%} | {eval_metrics['eval_acc']:>5.1%} | {eval_metrics['eval_deep_acc']:>5.1%}"
            )
            print(
                f"[{run_name}]   timing: data {pct_data:.0f}% fwd {pct_forward:.0f}% bwd {pct_backward:.0f}% eval {pct_eval:.0f}% | "
                f"dataloader: {dl_breakdown} | eval: {time_eval:.1f}s"
            )

            weight_norm = torch.norm(
                torch.stack([torch.norm(p) for p in model.parameters()])
            ).item()

            wandb.log(
                {
                    "elapsed_t": total_elapsed,
                    "cumulative_gflops": samples_seen * flops_per_sample / 1e9,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    **eval_metrics,
                    "samples_seen": samples_seen,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm_mean": grad_norm_sum / max(grad_norm_count, 1),
                    "grad_norm_max": grad_norm_max,
                    "grad_skips": grad_skips,
                    "weight_norm": weight_norm,
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
            grad_norm_sum = 0.0
            grad_norm_max = 0.0
            grad_norm_count = 0
            grad_skips = 0

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
        "n_embed": n_embed,
        "num_heads": num_heads,
        "total_steps": step,
    }


@app.function(image=image, timeout=86000)  # 23.5 hours for sweep coordination
def sweep():
    """Position encoding comparison: absolute vs bias64 vs shaw2d, at 3 sizes.

    Absolute baselines already exist as tf-shape-w-d{192,256,384} (renamed
    from pareto). Filter with ^tf-(shape-w|pos)-d(192|256|384) for all 9.

    All spatial_unshared head, L=8, bs=2048. Compare on cumulative_gflops.

    Lc0 claims bias64 is "+50% effective size"; ChessFormer claims shaw2d is
    +1.83pp @ 6M and beats bias64. Test both at 3 sizes to check scale-dependence.
    """
    configs = [
        {"run_name": "tf-pos-bias64-d192", "n_embed": 192, "pos_encoding": "bias64"},
        {"run_name": "tf-pos-bias64-d256", "n_embed": 256, "pos_encoding": "bias64"},
        {"run_name": "tf-pos-bias64-d384", "n_embed": 384, "pos_encoding": "bias64"},
        {"run_name": "tf-pos-shaw2d-d192", "n_embed": 192, "pos_encoding": "shaw2d"},
        {"run_name": "tf-pos-shaw2d-d256", "n_embed": 256, "pos_encoding": "shaw2d"},
        {"run_name": "tf-pos-shaw2d-d384", "n_embed": 384, "pos_encoding": "shaw2d"},
    ]

    shared = {
        "head_type": "spatial_unshared",
        "num_layers": 8,
        "num_heads": 8,
        "ff_ratio": 4,
        "batch_size": 2048,
        "lr": 4.24e-4,
        "warmup_steps": 250,
        "weight_decay": 0.01,
        "cosine_decay": False,
        "grad_clip": 1.0,
        "grad_skip_threshold": 20.0,
        "num_samples": "full",
        "min_depth": 0,
        "max_seconds": 7200,
        "max_steps": 0,
    }
    configs = [{**shared, **cfg} for cfg in configs]

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


@app.function(image=image, gpu="A10G", timeout=300)
def bench_vram():
    """Check VRAM usage for different model sizes."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    import torch.nn as nn
    from cnn_model import PolicyCNN

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    batch_size = 1024

    for h, name in [(584, "160M")]:
        torch.cuda.reset_peak_memory_stats()
        model = PolicyCNN(num_moves=1968, hidden_channels=h, num_layers=14, kernel_size=3, flip_board=True).to(device)
        model = torch.compile(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        planes = torch.randint(0, 2, (batch_size, 13, 8, 8), dtype=torch.uint8, device=device)
        meta = torch.randint(-1, 2, (batch_size, 5), dtype=torch.int8, device=device)
        target = torch.zeros(batch_size, 1968, device=device)
        target[:, 0] = 1.0

        try:
            for _ in range(3):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(planes, meta)
                    loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"{name} (H={h}): {peak_gb:.1f} GB / {total_gb:.1f} GB ({100*peak_gb/total_gb:.0f}%)")
        except torch.cuda.OutOfMemoryError:
            print(f"{name} (H={h}): OOM!")
        finally:
            del model, optimizer, criterion
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import fire

    fire.Fire(train.local)
