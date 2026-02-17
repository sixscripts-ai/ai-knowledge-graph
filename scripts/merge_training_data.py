#!/usr/bin/env python3
"""
Merge synthetic high-quality data with best graph-based examples.

Combines:
- 44 handcrafted synthetic examples (quality signal, paragraph format)
- Best examples from finetune_data_v2 with 100+ char answers (breadth)

This gives format quality + content diversity.
"""
import json
import random
from pathlib import Path

SYNTHETIC_DIR = Path("finetune_data_v3")
GRAPH_DIR = Path("finetune_data_v2")
OUTPUT_DIR = Path("finetune_data_merged")


def extract_assistant_text(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main():
    # Load synthetic data (all of it)
    synthetic = []
    for split in ["train", "valid", "test"]:
        p = SYNTHETIC_DIR / f"{split}.jsonl"
        if p.exists():
            synthetic.extend(load_jsonl(p))

    # Load graph-based data, keep only 100+ char answers
    graph_good = []
    graph_total = 0
    for split in ["train", "valid", "test"]:
        p = GRAPH_DIR / f"{split}.jsonl"
        if p.exists():
            for ex in load_jsonl(p):
                graph_total += 1
                answer = extract_assistant_text(ex.get("messages", []))
                if len(answer) >= 100:
                    graph_good.append(ex)

    print(f"Synthetic examples: {len(synthetic)}")
    print(f"Graph examples (100+ chars): {len(graph_good)} / {graph_total}")

    # Combine
    all_examples = synthetic + graph_good
    print(f"Combined: {len(all_examples)}")

    # Shuffle and split 80/10/10
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train": all_examples[:train_end],
        "valid": all_examples[train_end:valid_end],
        "test": all_examples[valid_end:],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, exs in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(exs)} -> {path}")

    # Quality stats
    all_answers = [extract_assistant_text(ex.get("messages", [])) for ex in all_examples]
    lengths = [len(a) for a in all_answers]
    print(f"\nAnswer length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Avg: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} chars")


if __name__ == "__main__":
    main()
