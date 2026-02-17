#!/usr/bin/env python3
"""
Merge ALL training data sources into finetune_data_final/.

Combines:
- finetune_data_v3:     44 handcrafted synthetic examples (v6 training core)
- finetune_data_v2:     Graph-based examples, filtered to 100+ char answers
- finetune_data_v4:     94 gap-filler examples covering 90 new concepts

Deduplicates on normalized question text.
Output: finetune_data_final/ with 80/10/10 train/valid/test split.
"""
import json
import random
from pathlib import Path

SOURCES = [
    ("finetune_data_v3", "Synthetic v3 (core quality)", None),
    ("finetune_data_v2", "Graph-based (filtered)", 100),   # min answer length
    ("finetune_data_v4", "Gap-filler v4 (90 concepts)", None),
]
OUTPUT_DIR = Path("finetune_data_final")


def extract_assistant_text(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def extract_user_text(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "").strip().lower()
    return ""


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    all_examples = []
    seen_questions = set()
    dupes = 0

    for src_name, label, min_answer_len in SOURCES:
        src_dir = Path(src_name)
        source_count = 0
        source_kept = 0
        for split in ["train", "valid", "test"]:
            p = src_dir / f"{split}.jsonl"
            if not p.exists():
                continue
            for ex in load_jsonl(p):
                source_count += 1
                # Filter by answer length if threshold set
                if min_answer_len:
                    answer = extract_assistant_text(ex.get("messages", []))
                    if len(answer) < min_answer_len:
                        continue
                # Deduplicate on question
                q = extract_user_text(ex.get("messages", []))
                if q in seen_questions:
                    dupes += 1
                    continue
                seen_questions.add(q)
                all_examples.append(ex)
                source_kept += 1
        print(f"  {label}: {source_kept} kept / {source_count} total")

    print(f"\nTotal unique examples: {len(all_examples)} (skipped {dupes} dupes)")

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

    print(f"\nDone — {len(all_examples)} total examples in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
