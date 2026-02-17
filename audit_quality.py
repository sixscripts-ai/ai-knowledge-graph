#!/usr/bin/env python3
"""Audit the quality of graph data and training data — signal vs noise."""
import json
from collections import Counter

# ICT core terms
ICT_TERMS = {
    'fvg','fair value gap','fair_value_gap','order block','order_block',
    'displacement','liquidity','liquidity sweep','breaker','breaker block',
    'mitigation','silver bullet','judas swing','accumulation','manipulation',
    'distribution','market structure','bos','break of structure','choch',
    'change of character','premium','discount','optimal trade entry','ote',
    'fibonacci','killzone','kill zone','london','new york','asia','cbdr',
    'amd','ict','inner circle trader','price action','swing high','swing low',
    'imbalance','efficiency','inefficiency','stop hunt','equal highs',
    'equal lows','buyside','sellside','buy side','sell side','institutional',
    'smart money','power of 3','po3','session','time','candle','entry',
    'stop loss','target','risk','trade','setup','model','bias','htf','ltf',
    'higher timeframe','lower timeframe','consolidation','expansion',
    'retracement','reversal','trending','ranging','volume','gap','chart',
    'price','market','delivery','algorithm','fractal','unicorn','macro',
    'micro','midnight','open','close','high','low','nwog','ndog','weekly',
    'daily','monthly','quarterly','confluence','pd array','price delivery',
    'standard deviation','flout','propulsion','narrative','draw on liquidity',
    'dealing range','balanced price range','bpr','consequent encroachment',
    'stophunt','run on liquidity','turtle soup','breaker block',
}

def is_ict_related(text):
    t = text.lower().strip()
    return any(term in t for term in ICT_TERMS)

# ── Graph audit ──────────────────────────────────────────────────────────
print("=" * 60)
print("  GRAPH DATA AUDIT")
print("=" * 60)

with open('ict_graph_final.json') as f:
    triples = json.load(f)

ict_count = sum(1 for t in triples if is_ict_related(t['subject']) or is_ict_related(t['object']))
noise_count = len(triples) - ict_count

print(f"Total triples: {len(triples)}")
print(f"ICT-related (subject OR object has ICT term): {ict_count} ({100*ict_count/len(triples):.1f}%)")
print(f"Non-ICT (neither side has ICT term):          {noise_count} ({100*noise_count/len(triples):.1f}%)")

# Show noisiest non-ICT subjects
subs = Counter()
for t in triples:
    if not is_ict_related(t['subject']) and not is_ict_related(t['object']):
        subs[t['subject'].lower().strip()] += 1

print(f"\nTop 25 NOISE subjects (no ICT connection):")
for s, c in subs.most_common(25):
    print(f"  {c:>4}x  {s}")

# Show some noise triples
print(f"\n10 NOISE triple examples:")
noise_triples = [t for t in triples if not is_ict_related(t['subject']) and not is_ict_related(t['object'])]
import random
random.seed(42)
for t in random.sample(noise_triples, min(10, len(noise_triples))):
    print(f"  {t['subject']}  --[{t['predicate']}]-->  {t['object']}")

# ── Training data audit ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING DATA AUDIT")
print("=" * 60)

with open('finetune_data/train.jsonl') as f:
    train = [json.loads(l) for l in f]

# Check answer quality
short_answers = 0
generic_answers = 0
good_answers = 0
answer_lengths = []

for ex in train:
    answer = ex['messages'][2]['content']
    answer_lengths.append(len(answer))
    
    if len(answer) < 50:
        short_answers += 1
    elif 'is an ICT concept' in answer and len(answer) < 150:
        generic_answers += 1
    else:
        good_answers += 1

print(f"Total training examples: {len(train)}")
print(f"Short answers (<50 chars):  {short_answers} ({100*short_answers/len(train):.1f}%)")
print(f"Generic boilerplate:        {generic_answers} ({100*generic_answers/len(train):.1f}%)")
print(f"Substantive answers:        {good_answers} ({100*good_answers/len(train):.1f}%)")
print(f"Avg answer length:          {sum(answer_lengths)/len(answer_lengths):.0f} chars")
print(f"Median answer length:       {sorted(answer_lengths)[len(answer_lengths)//2]} chars")

# Show some bad examples
print(f"\n5 SHORT/BAD training examples:")
bad = [ex for ex in train if len(ex['messages'][2]['content']) < 50]
for ex in random.sample(bad, min(5, len(bad))):
    q = ex['messages'][1]['content'][:80]
    a = ex['messages'][2]['content']
    print(f"  Q: {q}")
    print(f"  A: {a}")
    print()

# Show some good examples
print(f"5 GOOD training examples:")
good = [ex for ex in train if len(ex['messages'][2]['content']) > 300]
for ex in random.sample(good, min(5, len(good))):
    q = ex['messages'][1]['content'][:80]
    a = ex['messages'][2]['content'][:200]
    print(f"  Q: {q}")
    print(f"  A: {a}...")
    print()
