#!/usr/bin/env python3
"""
Clean and deduplicate ict_graph_final.json.

Steps:
  1. Remove triples with null/empty subject, predicate, or object
  2. Flatten list-typed objects into separate triples
  3. Normalize whitespace in all fields
  4. Remove exact duplicates (case-insensitive)
  5. Remove triples with very short subjects or objects (< 3 chars)
  6. Merge generic predicates (is/are → is_a, has/have → has_component, etc.)
  7. Remove triples with singleton predicates (appear only once — usually LLM noise)
  8. Preserve chunk_id and source metadata
  9. Write cleaned file + stats report

Usage:
    python clean_graph.py                    # dry run (stats only)
    python clean_graph.py --apply            # write cleaned file
    python clean_graph.py --apply --min-pred 2  # custom min predicate count
"""

import json
import copy
import argparse
import re
from collections import Counter
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "ict_graph_final.json"
OUTPUT_FILE = Path(__file__).parent / "ict_graph_final.json"
BACKUP_FILE = Path(__file__).parent / "ict_graph_final.backup.json"

# Generic predicate merges: old → new
PREDICATE_MERGES = {
    "is": "is_a",
    "are": "is_a",
    "be": "is_a",
    "was": "is_a",
    "were": "is_a",
    "has": "has_component",
    "have": "has_component",
    "include": "includes",
    "includes": "includes",
    "value": "has_value",
    "set": "belongs_to_set",
    "exists": "exists_in",
    "use": "uses",
    "used": "uses",
    "do": "performs",
    "does": "performs",
    "make": "produces",
    "makes": "produces",
    "called": "is_named",
    "known": "is_known_as",
}

# Noise subjects/objects — these are sentence fragments, not entities
NOISE_PATTERNS = [
    r"^(i|we|they|it|he|she|you|this|that|these|those|there|here)$",
    r"^(the|a|an|of|in|on|at|to|for|with|and|or|but|not)$",
    r"^\d+$",          # bare numbers
    r"^[.\-_/\\]+$",   # punctuation only
    r"^(file|page|line|section|chapter|part|item|example|note)$",
]
_noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def is_noise(text: str) -> bool:
    """Check if a subject/object is noise."""
    return bool(_noise_re.match(text.strip()))


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip quotes."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_graph(
    min_pred_count: int = 2,
    apply: bool = False,
    verbose: bool = True,
):
    """Clean the graph data and optionally write the result."""

    with open(INPUT_FILE) as f:
        raw_triples = json.load(f)

    total_raw = len(raw_triples)
    stats = {
        "total_raw": total_raw,
        "removed_null": 0,
        "removed_short": 0,
        "removed_noise": 0,
        "removed_duplicate": 0,
        "removed_singleton_pred": 0,
        "merged_predicates": 0,
        "flattened_lists": 0,
    }

    # ---- Step 1: Flatten list objects ----
    expanded = []
    for t in raw_triples:
        obj = t.get("object")
        if isinstance(obj, list):
            for item in obj:
                new_t = copy.copy(t)
                new_t["object"] = str(item) if item is not None else ""
                expanded.append(new_t)
                stats["flattened_lists"] += 1
        else:
            expanded.append(t)

    # ---- Step 2: Remove nulls / empty fields ----
    clean = []
    for t in expanded:
        s = t.get("subject")
        p = t.get("predicate")
        o = t.get("object")
        if not s or not p or (o is None or (isinstance(o, str) and not o.strip())):
            stats["removed_null"] += 1
            continue
        # Normalize
        t["subject"] = normalize_text(s)
        t["predicate"] = normalize_text(p).lower().replace(" ", "_")
        t["object"] = normalize_text(o)
        clean.append(t)

    # ---- Step 3: Remove short subjects/objects ----
    filtered = []
    for t in clean:
        if len(t["subject"]) < 3:
            stats["removed_short"] += 1
            continue
        if len(t["object"]) < 3:
            stats["removed_short"] += 1
            continue
        filtered.append(t)
    clean = filtered

    # ---- Step 4: Remove noise entities ----
    filtered = []
    for t in clean:
        if is_noise(t["subject"]) or is_noise(t["object"]):
            stats["removed_noise"] += 1
            continue
        filtered.append(t)
    clean = filtered

    # ---- Step 5: Merge generic predicates ----
    for t in clean:
        old_p = t["predicate"]
        if old_p in PREDICATE_MERGES:
            t["predicate"] = PREDICATE_MERGES[old_p]
            stats["merged_predicates"] += 1

    # ---- Step 6: Deduplicate (case-insensitive on s/p/o) ----
    seen = set()
    deduped = []
    for t in clean:
        key = (t["subject"].lower(), t["predicate"].lower(), t["object"].lower())
        if key in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(key)
        deduped.append(t)
    clean = deduped

    # ---- Step 7: Remove singleton predicates ----
    pred_counts = Counter(t["predicate"] for t in clean)
    filtered = []
    for t in clean:
        if pred_counts[t["predicate"]] < min_pred_count:
            stats["removed_singleton_pred"] += 1
            continue
        filtered.append(t)
    clean = filtered

    # ---- Final stats ----
    final_preds = Counter(t["predicate"] for t in clean)
    stats["total_final"] = len(clean)
    stats["total_removed"] = total_raw - len(clean)
    stats["removal_pct"] = f"{(total_raw - len(clean)) / total_raw * 100:.1f}%"
    stats["unique_predicates_final"] = len(final_preds)
    stats["unique_subjects_final"] = len(set(t["subject"].lower() for t in clean))
    stats["unique_objects_final"] = len(set(t["object"].lower() for t in clean))

    if verbose:
        print("=" * 60)
        print("  ICT Graph Cleaning Report")
        print("=" * 60)
        print(f"  Raw triples:            {stats['total_raw']:>7,}")
        print(f"  Flattened list objects:  {stats['flattened_lists']:>7,}")
        print(f"  Removed null/empty:     {stats['removed_null']:>7,}")
        print(f"  Removed short (<3ch):   {stats['removed_short']:>7,}")
        print(f"  Removed noise words:    {stats['removed_noise']:>7,}")
        print(f"  Removed duplicates:     {stats['removed_duplicate']:>7,}")
        print(f"  Merged predicates:      {stats['merged_predicates']:>7,}")
        print(f"  Removed singleton pred: {stats['removed_singleton_pred']:>7,}")
        print(f"  ──────────────────────────────────────")
        print(f"  Final triples:          {stats['total_final']:>7,}")
        print(f"  Total removed:          {stats['total_removed']:>7,} ({stats['removal_pct']})")
        print(f"  Unique predicates:      {stats['unique_predicates_final']:>7,}")
        print(f"  Unique subjects:        {stats['unique_subjects_final']:>7,}")
        print(f"  Unique objects:         {stats['unique_objects_final']:>7,}")
        print("=" * 60)

        # Top 20 predicates
        print("\n  Top 20 predicates (after cleaning):")
        for pred, cnt in final_preds.most_common(20):
            print(f"    {pred:40s} {cnt:>5,}")

    if apply:
        # Backup original
        import shutil
        if not BACKUP_FILE.exists():
            shutil.copy2(INPUT_FILE, BACKUP_FILE)
            print(f"\n  Backup saved: {BACKUP_FILE.name}")
        else:
            print(f"\n  Backup already exists: {BACKUP_FILE.name}")

        # Write cleaned data
        with open(OUTPUT_FILE, "w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"  Cleaned data written: {OUTPUT_FILE.name}")
        print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")
    else:
        print("\n  DRY RUN — no files modified. Use --apply to write.")

    return clean, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean ICT graph data")
    parser.add_argument("--apply", action="store_true", help="Write cleaned file")
    parser.add_argument(
        "--min-pred",
        type=int,
        default=2,
        help="Minimum predicate occurrence count (default: 2)",
    )
    args = parser.parse_args()
    clean_graph(min_pred_count=args.min_pred, apply=args.apply)
