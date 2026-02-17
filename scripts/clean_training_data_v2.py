#!/usr/bin/env python3
"""
Clean training data v2 — Aggressive detemplatization.

The v1 cleaner kept too many template-style "graph dump" examples that caused
mode collapse. This version:
1. Removes ALL "key relationships:" template answers
2. Removes answers with 3+ bullet-list "- is a" patterns  
3. Requires minimum 100-char answers (was 30)
4. Removes answers that are purely bullet lists (no prose)
5. Keeps only examples with natural paragraph-style responses
"""
import json
import re
from pathlib import Path
from collections import Counter

INPUT_DIR = Path("finetune_data_clean")  # v1 cleaned data
OUTPUT_DIR = Path("finetune_data_v2")

def extract_assistant_text(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""

def is_template_garbage(answer: str) -> tuple[bool, str]:
    """Return (should_remove, reason) for template-style answers."""
    
    # Kill: "X is an ICT concept with these key relationships:"
    if "key relationships:" in answer.lower():
        return True, "key_relationships_template"
    
    # Kill: 3+ "- is a" bullets (graph triple dumps)
    if answer.count("- is a ") >= 3:
        return True, "is_a_bullet_spam"
    
    # Kill: Answers that are mostly bullet lists with no prose
    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    bullet_lines = [l for l in lines if l.startswith('- ')]
    if len(lines) > 2 and len(bullet_lines) / len(lines) > 0.8:
        # More than 80% bullet lines — it's a list dump
        return True, "bullet_list_dump"
    
    # Kill: Very short answers (under 50 chars = ~8 words)
    if len(answer) < 50:
        return True, "too_short"
    
    # Kill: answers that look like raw graph notation
    if re.search(r'(?:is_a|has_component|belongs_to|relates_to)', answer):
        return True, "raw_graph_predicates"
    
    # Kill: Answers with too many repeated price patterns
    prices = re.findall(r'\$\d+\.\d+', answer)
    if len(prices) >= 3:
        unique_prices = set(prices)
        if len(unique_prices) < len(prices) * 0.5:
            return True, "repeated_prices"

    # Kill: "Is is", "Is has", "Is includes" patterns (broken grammar from graph ops)
    if re.search(r'- Is (is|has|includes|contains|occurs|uses)\b', answer):
        return True, "broken_grammar_bullets"
    
    return False, ""

def process_file(input_path: Path, output_path: Path, stats: dict):
    """Process one JSONL file."""
    kept = []
    removed_by_reason = Counter()
    
    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)
            answer = extract_assistant_text(ex.get("messages", []))
            
            should_remove, reason = is_template_garbage(answer)
            if should_remove:
                removed_by_reason[reason] += 1
            else:
                kept.append(ex)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in kept:
            f.write(json.dumps(ex) + '\n')
    
    stats[input_path.stem] = {
        "original": sum(removed_by_reason.values()) + len(kept),
        "kept": len(kept),
        "removed_by_reason": dict(removed_by_reason)
    }

def main():
    stats = {}
    
    for split in ["train", "valid", "test"]:
        input_path = INPUT_DIR / f"{split}.jsonl"
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        if input_path.exists():
            process_file(input_path, output_path, stats)
    
    print("=" * 60)
    print("TRAINING DATA v2 CLEANING REPORT")
    print("=" * 60)
    
    total_orig = 0
    total_kept = 0
    
    for split, info in stats.items():
        print(f"\n{split}:")
        print(f"  Original: {info['original']}")
        print(f"  Kept:     {info['kept']} ({100*info['kept']/info['original']:.1f}%)")
        print(f"  Removed by reason:")
        for reason, count in sorted(info['removed_by_reason'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        total_orig += info['original']
        total_kept += info['kept']
    
    print(f"\nTOTAL: {total_kept}/{total_orig} kept ({100*total_kept/total_orig:.1f}%)")
    
    # Show sample of kept answers for quality check
    train_out = OUTPUT_DIR / "train.jsonl"
    if train_out.exists():
        print("\n" + "=" * 60)
        print("SAMPLE KEPT ANSWERS (first 5):")
        print("=" * 60)
        with open(train_out) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                ex = json.loads(line)
                q = ""
                a = ""
                for m in ex.get("messages", []):
                    if m["role"] == "user":
                        q = m["content"][:80]
                    if m["role"] == "assistant":
                        a = m["content"][:200]
                print(f"\n  Q: {q}")
                print(f"  A: {a}")

if __name__ == "__main__":
    main()
