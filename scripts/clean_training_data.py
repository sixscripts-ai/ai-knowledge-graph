#!/usr/bin/env python3
"""
Clean training data: remove noise, boilerplate, and low-quality examples.
Keeps only substantive ICT trading content.

Writes cleaned data to finetune_data_clean/
"""
import json
import os
import re
import random
from pathlib import Path

random.seed(42)

# ── Noise filters ─────────────────────────────────────────────────────────────

# Subjects/topics that are NOT ICT trading content
NOISE_SUBJECTS = {
    'script', 'folder structure', 'file', 'module', 'class', 'function',
    'import', 'code', 'python', 'repository', 'directory', 'archive',
    'vexcontroller', 'backtestengine', 'ict agent', 'controller',
    'config', 'configuration', 'package', 'dependency', 'pip',
}

# Patterns that indicate code/infra noise, not trading content
NOISE_PATTERNS = [
    r'\.py\b',                    # Python file references
    r'/archive/',                 # File paths
    r'/src/',                     # File paths
    r'__init__',                  # Python internals
    r'import\s+\w+',             # Import statements
    r'def\s+\w+',               # Function definitions
    r'class\s+\w+',             # Class definitions
    r'vexcontroller',            # VEX code, not ICT
    r'backtestengine',           # Code class names
    r'folder\s+structure',       # Repo structure
    r'\$_+',                     # Blank dollar amounts "$____"
]

# Patterns indicating journal noise (dates, personal entries)
JOURNAL_NOISE = [
    r'^\d{4}-\d{2}-\d{2}$',     # Just a date as subject
    r'january \d{4}',           # Month references
    r'february \d{4}',
    r'pending:\s*\d+',          # Journal stats
    r'i feel like',             # Personal diary
    r'i have something to prove',
    r'woke up in profit',       # Specific trade journal
    r'went to sleep',
    r'chris laurie',            # Names that aren't ICT concepts
]

COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]
COMPILED_JOURNAL = [re.compile(p, re.IGNORECASE) for p in JOURNAL_NOISE]


def is_noise(example):
    """Return True if this training example is noise, not ICT content."""
    question = example['messages'][1]['content'].lower()
    answer = example['messages'][2]['content']
    answer_lower = answer.lower()
    
    # 1. Too short to be useful
    if len(answer) < 30:
        return True
    
    # 2. Code/infra references in question
    for pat in COMPILED_NOISE:
        if pat.search(question) or pat.search(answer_lower):
            return True
    
    # 3. Journal noise
    for pat in COMPILED_JOURNAL:
        if pat.search(answer_lower):
            return True
    
    # 4. Noise subjects in question
    for ns in NOISE_SUBJECTS:
        if ns in question and 'market structure' not in question:
            return True
    
    # 5. Generic boilerplate with no real content
    if answer_lower.startswith('in ict methodology,') and len(answer) < 60:
        return True
    
    # 6. Answers that are just restating the predicate with no explanation
    # e.g., "In ICT methodology, X relates to Y."
    if re.match(r'^in ict methodology, .{5,40} \w+ .{5,40}\.$', answer_lower) and len(answer) < 80:
        return True
    
    return False


def clean_dataset(input_path, output_path):
    """Clean a JSONL dataset, return stats."""
    with open(input_path) as f:
        examples = [json.loads(line) for line in f]
    
    kept = []
    removed = []
    
    for ex in examples:
        if is_noise(ex):
            removed.append(ex)
        else:
            kept.append(ex)
    
    with open(output_path, 'w') as f:
        for ex in kept:
            f.write(json.dumps(ex) + '\n')
    
    return len(examples), len(kept), len(removed), removed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    input_dir = Path('finetune_data')
    output_dir = Path('finetune_data_clean')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  CLEANING TRAINING DATA")
    print("=" * 60)
    
    total_orig = 0
    total_kept = 0
    total_removed = 0
    
    for split in ['train', 'valid', 'test']:
        infile = input_dir / f'{split}.jsonl'
        outfile = output_dir / f'{split}.jsonl'
        
        if not infile.exists():
            print(f"  SKIP {split} — not found")
            continue
        
        orig, kept, removed, removed_examples = clean_dataset(infile, outfile)
        total_orig += orig
        total_kept += kept
        total_removed += removed
        
        print(f"\n  {split}:")
        print(f"    Original:  {orig}")
        print(f"    Kept:      {kept} ({100*kept/orig:.1f}%)")
        print(f"    Removed:   {removed} ({100*removed/orig:.1f}%)")
        
        if removed_examples and split == 'train':
            print(f"\n    5 removed examples:")
            for ex in random.sample(removed_examples, min(5, len(removed_examples))):
                q = ex['messages'][1]['content'][:80]
                a = ex['messages'][2]['content'][:80]
                print(f"      Q: {q}")
                print(f"      A: {a}")
                print()
    
    # Copy training_stats.json if exists
    stats_file = input_dir / 'training_stats.json'
    if stats_file.exists():
        import shutil
        shutil.copy2(stats_file, output_dir / 'training_stats.json')
    
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total original: {total_orig}")
    print(f"  Total kept:     {total_kept} ({100*total_kept/total_orig:.1f}%)")
    print(f"  Total removed:  {total_removed} ({100*total_removed/total_orig:.1f}%)")
    print(f"\n  Clean data written to: {output_dir}/")
    print(f"  Ready for training!")
