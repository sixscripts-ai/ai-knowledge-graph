#!/usr/bin/env python3
"""Run the enhanced training generator and export new training data."""
import sys, json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph_rag.graph_store import ICTGraphStore
from graph_rag.training_generator import TrainingDataGenerator

print("=" * 60)
print("Loading graph store...")
store = ICTGraphStore()
store.load_all()
print(f"Graph: {store.G.number_of_nodes()} nodes, {store.G.number_of_edges()} edges")

print("\n" + "=" * 60)
print("Running enhanced training generator...")
gen = TrainingDataGenerator(store)
total = gen.generate_all()

print("\n" + "=" * 60)
print(f"Total examples generated: {total}")
print(f"Examples in buffer: {len(gen.examples)}")

# Show a few examples from each category
cats = {}
for ex in gen.examples:
    cat = ex.get("category", "unknown")
    cats.setdefault(cat, []).append(ex)

print(f"\nExamples by category:")
for cat, examples in sorted(cats.items()):
    print(f"  {cat}: {len(examples)}")

# Show 2 samples from each category
print("\n" + "=" * 60)
print("Sample examples (2 per category):\n")
for cat, examples in sorted(cats.items()):
    print(f"--- {cat} ---")
    for ex in examples[:2]:
        q = ex.get("instruction", "")[:120]
        a = ex.get("output", "")[:120]
        print(f"  Q: {q}")
        print(f"  A: {a}")
    print()

# Export train/test/valid split
print("=" * 60)
print("Exporting train/test/valid split...")
output_dir = Path(__file__).parent / "training_output"
gen.export_train_test_split(output_dir=output_dir)

# Also export to finetune_data for MLX (needs train.jsonl, valid.jsonl, test.jsonl)
print("\nExporting to finetune_data for MLX...")
finetune_dir = Path(__file__).parent / "finetune_data"
gen.export_train_test_split(output_dir=finetune_dir)

# Create MLX-compatible copies (train.jsonl, valid.jsonl, test.jsonl)
mlx_dir = finetune_dir
import shutil
for src_name, dst_name in [("ict_train.jsonl", "train.jsonl"), ("ict_test.jsonl", "test.jsonl")]:
    src_path = mlx_dir / src_name
    dst_path = mlx_dir / dst_name
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"  Copied {src_name} -> {dst_name}")

# Create valid.jsonl as a small subset of test
test_path = mlx_dir / "test.jsonl"
valid_path = mlx_dir / "valid.jsonl"
if test_path.exists():
    with open(test_path) as f:
        test_lines = f.readlines()
    # Use first half as valid, keep full as test
    valid_lines = test_lines[:len(test_lines) // 2]
    with open(valid_path, "w") as f:
        f.writelines(valid_lines)
    print(f"  Created valid.jsonl with {len(valid_lines)} examples")

print("\nDone!")
