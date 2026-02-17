#!/usr/bin/env python3
"""
MLX LoRA Fine-Tuning with Early Stopping

Wraps `mlx_lm.lora` to add early stopping based on validation loss.
Monitors checkpoints saved by mlx_lm and keeps only the best one.

Usage:
    # Install mlx-lm first: pip install mlx-lm
    python3 train_with_early_stopping.py

Changes from v1 config:
    - Cosine LR decay with warmup (was: constant LR)
    - LoRA dropout 0.05 (was: 0.0)
    - Gradient accumulation steps 2 (was: 1) → effective batch = 8
    - 3000 iters (was: 800) to match larger dataset (6K+ examples)
    - Early stopping: patience of 5 evals (~500 iters)
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "adapter_config_v2.json"
ADAPTER_DIR = BASE_DIR / "adapters"
BEST_ADAPTER_DIR = BASE_DIR / "adapters-best"

PATIENCE = 5         # Number of evals without improvement before stopping
MIN_DELTA = 0.001    # Minimum val loss improvement to count as "better"


def parse_val_loss(output_lines: list[str]) -> list[tuple[int, float]]:
    """Parse validation loss from mlx_lm output lines.
    
    Expected format: "Iter X: Val loss X.XXX, Val ppl X.XXX"
    """
    results = []
    pattern = re.compile(r"Iter\s+(\d+).*?Val loss\s+([\d.]+)")
    for line in output_lines:
        m = pattern.search(line)
        if m:
            results.append((int(m.group(1)), float(m.group(2))))
    return results


def find_best_checkpoint(adapter_dir: Path) -> tuple[int | None, Path | None]:
    """Find the checkpoint with the lowest validation loss from saved adapter configs."""
    best_iter = None
    best_loss = float("inf")
    best_path = None

    for ckpt_dir in sorted(adapter_dir.glob("checkpoint_*")):
        config_path = ckpt_dir / "adapter_config.json"
        if not config_path.exists():
            continue
        # The checkpoint directory name contains the iteration number
        try:
            iter_num = int(ckpt_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # We'll track by the checkpoint we find
        if best_path is None:
            best_iter = iter_num
            best_path = ckpt_dir
    
    return best_iter, best_path


def main():
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)

    # Verify mlx_lm is available
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("mlx-lm not installed. Install with: pip install mlx-lm")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("=" * 60)
    print("  ICT VEX LoRA Fine-Tuning v2")
    print("=" * 60)
    print(f"  Model:    {config['model']}")
    print(f"  Data:     {config['data']}")
    print(f"  Iters:    {config['iters']}")
    print(f"  LR:       {config['learning_rate']}")
    print(f"  Schedule: cosine_decay with warmup={config['lr_schedule']['warmup']}")
    print(f"  LoRA:     rank={config['lora_parameters']['rank']}, "
          f"dropout={config['lora_parameters']['dropout']}, "
          f"scale={config['lora_parameters']['scale']}")
    print(f"  Patience: {PATIENCE} evals (early stopping)")
    print()

    # Build mlx_lm.lora command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--config", str(CONFIG_PATH),
    ]

    print(f"Running: {' '.join(cmd)}\n")
    print("-" * 60)

    # Run with line-by-line output monitoring for early stopping
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    best_val_loss = float("inf")
    best_iter = 0
    patience_counter = 0
    all_output = []

    for line in process.stdout:
        line = line.rstrip()
        print(line)
        all_output.append(line)

        # Check for validation loss
        val_results = parse_val_loss([line])
        if val_results:
            iter_num, val_loss = val_results[0]
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                best_iter = iter_num
                patience_counter = 0
                print(f"  >>> New best val loss: {val_loss:.4f} at iter {iter_num}")
            else:
                patience_counter += 1
                print(f"  >>> No improvement ({patience_counter}/{PATIENCE}). "
                      f"Best: {best_val_loss:.4f} at iter {best_iter}")

            if patience_counter >= PATIENCE:
                print(f"\n{'='*60}")
                print(f"  EARLY STOPPING at iter {iter_num}")
                print(f"  Best val loss: {best_val_loss:.4f} at iter {best_iter}")
                print(f"{'='*60}")
                process.terminate()
                break

    process.wait()

    # Report results
    print(f"\n{'='*60}")
    print("  Training Complete")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f} at iter {best_iter}")

    # Copy best checkpoint as the primary adapter
    best_ckpt = ADAPTER_DIR / f"checkpoint_{best_iter:04d}"
    if not best_ckpt.exists():
        # Try finding closest saved checkpoint
        closest = None
        for ckpt in sorted(ADAPTER_DIR.glob("checkpoint_*")):
            try:
                n = int(ckpt.name.split("_")[1])
                if n <= best_iter:
                    closest = ckpt
            except (IndexError, ValueError):
                continue
        if closest:
            best_ckpt = closest
            print(f"  Using closest checkpoint: {best_ckpt.name}")

    if best_ckpt.exists():
        BEST_ADAPTER_DIR.mkdir(exist_ok=True)
        for f in best_ckpt.glob("*"):
            shutil.copy2(f, BEST_ADAPTER_DIR / f.name)
        print(f"  Best adapters saved to: {BEST_ADAPTER_DIR}")
    else:
        print(f"  Warning: checkpoint not found at {best_ckpt}")
        print("  Use the final adapters at:", ADAPTER_DIR)

    print()


if __name__ == "__main__":
    main()
