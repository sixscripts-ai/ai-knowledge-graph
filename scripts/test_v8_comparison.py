#!/usr/bin/env python3
"""v8 vs v7 comparison test.

Runs the same 16 questions from the v7 test suite against
the newly fused v8 model and prints a side-by-side comparison.

Usage:
    source venv/bin/activate
    python scripts/test_v8_comparison.py
"""

import sys, json, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ── Config ────────────────────────────────────────────────────────────────
V7_MODEL = str(_ROOT / "vex-3b-ict-v7")
V8_MODEL = str(_ROOT / "vex-3b-ict-v8")

QUESTIONS = [
    # Core concepts (5)
    ("What is a Fair Value Gap (FVG) and why does it matter to ICT traders?", "core"),
    ("Explain the Power of Three concept in ICT methodology.", "core"),
    ("What is a liquidity sweep and how does it relate to stop hunts?", "core"),
    ("What is the Optimal Trade Entry (OTE) zone?", "core"),
    ("Describe the relationship between order blocks and institutional order flow.", "core"),

    # Gap-filler concepts — v7 weaknesses (5)
    ("What is CISD and how does it confirm a change in market structure?", "gap"),
    ("Explain the Interbank Price Delivery Algorithm (IPDA) and its role in price delivery.", "gap"),
    ("What is the difference between a breaker block and a mitigation block?", "gap"),
    ("How does the Judas Swing work during the London session manipulation phase?", "gap"),
    ("What is the Silver Bullet model and when should it be traded?", "gap"),

    # Abbreviations — v7 weaknesses (3)
    ("What does MSS stand for in ICT terminology?", "abbr"),
    ("What does MMBM stand for?", "abbr"),
    ("What is BSL in ICT context?", "abbr"),

    # Cross-concept (3)
    ("How do FVGs and order blocks work together in a Model 11 setup?", "cross"),
    ("What is the relationship between displacement and fair value gaps?", "cross"),
    ("How does time (killzones) interact with liquidity levels in ICT?", "cross"),
]


def ask_model(model, tokenizer, question: str) -> str:
    """Single question → answer."""
    messages = [
        {"role": "system", "content": "You are VEX, an expert ICT trading assistant. Give clear, accurate, concise answers about ICT methodology."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.3)
    return generate(model, tokenizer, prompt=prompt, max_tokens=300, sampler=sampler)


def run_suite(model_path: str, label: str) -> list[dict]:
    """Run all questions through a model and collect results."""
    print(f"\n{'='*60}")
    print(f"Testing: {label} ({Path(model_path).name})")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    results = []

    for i, (q, cat) in enumerate(QUESTIONS, 1):
        print(f"[{i:2d}/{len(QUESTIONS)}] {cat}: {q[:60]}...")
        t0 = time.time()
        answer = ask_model(model, tokenizer, q)
        elapsed = time.time() - t0

        results.append({
            "question": q,
            "category": cat,
            "answer": answer.strip(),
            "time_s": round(elapsed, 1),
        })
        # Print first 120 chars of answer
        preview = answer.strip().replace("\n", " ")[:120]
        print(f"  → {preview}...")
        print(f"  ({elapsed:.1f}s)\n")

    return results


def main():
    # Check models exist
    for path, name in [(V7_MODEL, "v7"), (V8_MODEL, "v8")]:
        if not Path(path).exists():
            print(f"❌ {name} model not found at {path}")
            return 1

    v7_results = run_suite(V7_MODEL, "v7")
    v8_results = run_suite(V8_MODEL, "v8")

    # Save results
    output = {
        "v7": v7_results,
        "v8": v8_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = _ROOT / "reports" / "v8_vs_v7_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n📊 Full results saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    cats = {}
    for r7, r8 in zip(v7_results, v8_results):
        cat = r7["category"]
        cats.setdefault(cat, {"v7": [], "v8": []})
        cats[cat]["v7"].append(r7)
        cats[cat]["v8"].append(r8)

    for cat, data in cats.items():
        v7_avg = sum(r["time_s"] for r in data["v7"]) / len(data["v7"])
        v8_avg = sum(r["time_s"] for r in data["v8"]) / len(data["v8"])
        print(f"\n  {cat}: v7 avg {v7_avg:.1f}s, v8 avg {v8_avg:.1f}s")
        for r7, r8 in zip(data["v7"], data["v8"]):
            q_short = r7["question"][:50]
            print(f"    Q: {q_short}")
            print(f"    v7: {r7['answer'][:80]}...")
            print(f"    v8: {r8['answer'][:80]}...")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
