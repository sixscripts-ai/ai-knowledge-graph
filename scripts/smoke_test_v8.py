#!/usr/bin/env python3
"""Quick smoke test for v8 fused model — 5 key questions that v7 struggled with."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = str(_ROOT / "vex-3b-ict-v8")

SMOKE_QUESTIONS = [
    ("What is CISD and how does it confirm a change in market structure?", "CISD fix"),
    ("Explain the Interbank Price Delivery Algorithm (IPDA).", "IPDA fix"),
    ("What does MSS stand for in ICT?", "Abbreviation"),
    ("What does MMBM stand for?", "Abbreviation"),
    ("What is BSL in ICT context?", "Abbreviation"),
]

def main():
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)
    sampler = make_sampler(temp=0.3)

    print(f"\n{'='*60}")
    print("v8 Smoke Test — Focus Areas")
    print(f"{'='*60}\n")

    for q, label in SMOKE_QUESTIONS:
        messages = [
            {"role": "system", "content": "You are VEX, an expert ICT trading assistant. Give clear, accurate, concise answers."},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = generate(model, tokenizer, prompt=prompt, max_tokens=200, sampler=sampler)

        print(f"[{label}] Q: {q}")
        print(f"  A: {answer.strip()[:200]}")
        
        # Quick quality checks
        issues = []
        a_lower = answer.lower()
        if "1.343" in answer:
            issues.append("REGRESSION: '1.343' hallucination still present")
        if a_lower.count(a_lower[:30]) > 2 and len(a_lower) > 50:
            issues.append("WARNING: possible circular/repetitive text")
        
        if issues:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print(f"  ✅ No known regressions detected")
        print()

    print("Done.")

if __name__ == "__main__":
    main()
