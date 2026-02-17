#!/usr/bin/env python3
"""Diagnose training data for repetition patterns that cause model collapse."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("finetune_data_clean")

def extract_assistant_text(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""

def extract_user_text(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def main():
    train_file = DATA_DIR / "train.jsonl"
    examples = []
    with open(train_file) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Total training examples: {len(examples)}")
    print()

    # 1. Check how many responses start with the template pattern
    template_starts = Counter()
    is_a_count = 0
    relationship_list_count = 0
    total_is_a_occurrences = 0

    for ex in examples:
        answer = extract_assistant_text(ex.get("messages", []))
        # First 80 chars
        start = answer[:80].strip()
        template_starts[start] += 1

        # Count "is a" patterns in answers
        is_a_in_answer = answer.count("- is a ")
        total_is_a_occurrences += is_a_in_answer
        if is_a_in_answer > 0:
            is_a_count += 1
        if "key relationships:" in answer:
            relationship_list_count += 1

    print("=" * 60)
    print("PATTERN 1: 'is a' bullet repetition in answers")
    print(f"  Examples containing '- is a ': {is_a_count} / {len(examples)} ({100*is_a_count/len(examples):.1f}%)")
    print(f"  Total '- is a ' occurrences: {total_is_a_occurrences}")
    print(f"  Examples with 'key relationships:': {relationship_list_count} / {len(examples)} ({100*relationship_list_count/len(examples):.1f}%)")
    print()

    # 2. Show the most common answer starts
    print("=" * 60)
    print("TOP 20 answer openings (first 80 chars):")
    for start, count in template_starts.most_common(20):
        print(f"  [{count:4d}x] {start!r}")
    print()

    # 3. Show some examples with heavy "is a" repetition
    print("=" * 60)
    print("WORST EXAMPLES (most '- is a ' repetitions):")
    scored = []
    for ex in examples:
        answer = extract_assistant_text(ex.get("messages", []))
        question = extract_user_text(ex.get("messages", []))
        count = answer.count("- is a ")
        scored.append((count, question[:60], answer[:200]))
    scored.sort(reverse=True)
    for count, q, a in scored[:10]:
        print(f"\n  [{count}x '- is a '] Q: {q}")
        print(f"  A: {a!r}")

    # 4. Check answer diversity
    print()
    print("=" * 60)
    print("ANSWER DIVERSITY:")
    answers = [extract_assistant_text(ex.get("messages", [])) for ex in examples]
    unique_answers = set(answers)
    print(f"  Total answers: {len(answers)}")
    print(f"  Unique answers: {len(unique_answers)}")
    print(f"  Duplication rate: {100*(1 - len(unique_answers)/len(answers)):.1f}%")

    # Check unique first-sentence
    first_sentences = [a.split('.')[0] for a in answers]
    unique_first = set(first_sentences)
    print(f"  Unique first sentences: {len(unique_first)}")
    print()

    # 5. Average answer length
    lengths = [len(a) for a in answers]
    print(f"  Avg answer length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Median: {sorted(lengths)[len(lengths)//2]}")

    # 6. Check for price patterns like "$1.50", "$1.00"
    print()
    print("=" * 60)
    print("PRICE PATTERNS in answers:")
    import re
    price_pattern = re.compile(r'\$\d+\.\d+')
    price_counts = Counter()
    examples_with_prices = 0
    for a in answers:
        prices = price_pattern.findall(a)
        if prices:
            examples_with_prices += 1
            for p in prices:
                price_counts[p] += 1
    print(f"  Examples with dollar amounts: {examples_with_prices} / {len(answers)} ({100*examples_with_prices/len(answers):.1f}%)")
    print(f"  Top prices: {price_counts.most_common(20)}")

    # 7. Check "- is a" pattern specifically
    print()
    print("=" * 60)
    print("BULLET POINT patterns in answers:")
    bullet_pattern = re.compile(r'- \w+ \w+')
    bullet_starts = Counter()
    for a in answers:
        bullets = re.findall(r'- (\w+ \w+)', a)
        for b in bullets:
            bullet_starts[b] += 1
    print(f"  Top bullet starts:")
    for pattern, count in bullet_starts.most_common(20):
        print(f"    [{count:5d}x] - {pattern}")

if __name__ == "__main__":
    main()
