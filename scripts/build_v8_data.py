#!/usr/bin/env python3
"""
v8 Training Data Pipeline
=========================
1. Scan finetune_data_final/ for degenerate/contaminated examples
2. Remove them
3. Generate replacement high-quality examples for CISD, IPDA, abbreviation resolution
4. Create finetune_data_v8/ with cleaned + new data
5. Generate adapter_config for v8 training
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Step 1: Identify bad examples ─────────────────────────────────────────────

def find_bad_examples(train_file):
    """Find template-contaminated or degenerate training examples."""
    bad_indices = set()
    all_examples = []

    for i, line in enumerate(open(train_file)):
        msg = json.loads(line)
        a = msg["messages"][2]["content"].strip()
        q = msg["messages"][1]["content"].lower()
        all_examples.append(msg)

        is_bad = False

        # Template contamination: starts with relational graph fragments
        lower_a = a.lower()
        if any(lower_a.startswith(p) for p in [
            "in ict methodology, liquidity distribution",
            "liquidity distribution (ipda) ",
            "liquidity distribution (ipda)",
        ]):
            is_bad = True

        # Graph-derived one-line relational answers
        words = a.split()
        if len(words) >= 2 and len(words) <= 15:
            first2 = " ".join(words[:2]).lower()
            if any(first2.endswith(x) for x in [" concept", " application", " structure", " component"]):
                is_bad = True

        # Repetitive number hallucinations
        if "1.343 1.343" in a or "1.343 " * 3 in a:
            is_bad = True

        # Very short answers (< 30 chars) that are just graph fragments
        if len(a) < 30 and any(x in lower_a for x in ["concept ", "application ", "component "]):
            is_bad = True

        if is_bad:
            bad_indices.add(i)
            print(f"  BAD [{i}] Q: {q[:70]}")
            print(f"       A: {a[:100]}")

    return bad_indices, all_examples


# ── Step 2: Generate high-quality replacement + new examples ──────────────────

def generate_v8_examples():
    """Create high-quality examples for problem areas."""
    examples = []

    # ── CISD fixes (was degenerate "1.343" repetition) ──
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What is Change in State of Delivery (CISD)?"},
                {"role": "assistant", "content": "Change in State of Delivery (CISD) is a critical ICT concept that identifies when price transitions from one delivery state to another. Specifically, CISD occurs when a candle body closes through a prior candle's body that was delivering price in the opposite direction. For example, if price has been delivering bearishly (closing lower) and a bullish candle body closes above the body of the last bearish candle, that marks a CISD to bullish delivery. CISD is significant because it signals that the algorithm has shifted its delivery program — it's no longer seeking liquidity in the previous direction. Traders use CISD as confirmation of a directional shift, often aligning it with displacement and FVGs for high-probability entries."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "How do you identify a CISD on a chart?"},
                {"role": "assistant", "content": "To identify a CISD on a chart, follow these steps:\n\n1. **Identify current delivery state**: Determine whether price is delivering bullishly (closing higher) or bearishly (closing lower) by looking at consecutive candle bodies.\n\n2. **Watch for the shift candle**: Look for a candle whose body closes through the body of the last candle delivering in the prior direction.\n\n3. **Confirm body-through-body**: The key is body-through-body — wicks don't count. The close of the new candle must penetrate the open-to-close range of the opposing delivery candle.\n\n4. **Mark the CISD level**: The point where the shift candle's body crosses through becomes a reference level for potential retests.\n\nCISD is most reliable on the 15-minute timeframe and above, and it's strongest when it aligns with a killzone, displacement, and a higher timeframe PD array."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What is the difference between CISD and market structure shift?"},
                {"role": "assistant", "content": "While both CISD and Market Structure Shift (MSS) signal directional changes, they measure different things:\n\n**Market Structure Shift (MSS):**\n- Defined by a break of a swing high or swing low\n- Focuses on price structure (higher highs/lows or lower highs/lows)\n- Occurs when price breaks a previous swing point with displacement\n- Can be seen across all timeframes\n\n**Change in State of Delivery (CISD):**\n- Defined by candle body closing through a prior opposing candle body\n- Focuses on the algorithm's delivery program (how price is being delivered)\n- Occurs at the candle-body level, not swing structure level\n- More granular and can signal shifts before MSS confirms\n\nIn practice, CISD often precedes MSS — the delivery shifts first, then structure follows. A CISD that aligns with an MSS provides stronger confirmation. Many traders use CISD for precision entries after MSS provides the directional bias."}
            ]
        },
    ])

    # ── IPDA fixes (was circular text) ──
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What is the IPDA?"},
                {"role": "assistant", "content": "The Interbank Price Delivery Algorithm (IPDA) is ICT's foundational concept — the idea that price movements are not random but are driven by an algorithm that delivers price to specific liquidity pools and PD arrays. The IPDA operates on a predictable framework:\n\n1. **Seek liquidity**: The algorithm targets resting orders (buy stops above equal highs, sell stops below equal lows)\n2. **Deliver to PD arrays**: After sweeping liquidity, price is delivered to PD arrays like FVGs, order blocks, and breakers\n3. **Rebalance inefficiencies**: The algorithm fills Fair Value Gaps and rebalances price between premium and discount levels\n\nThe IPDA works across all timeframes and is the reason ICT concepts like killzones, Power of Three, and Market Maker Models exist — they are all expressions of how the algorithm delivers price."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What are the IPDA data ranges?"},
                {"role": "assistant", "content": "The IPDA uses three lookback periods to determine which PD arrays and liquidity levels are most relevant for current price delivery:\n\n1. **20-Day Lookback**: The most immediate range — covers roughly one trading month. PD arrays within this window are highest priority for the algorithm.\n\n2. **40-Day Lookback**: Extends the view to approximately two trading months. Levels here are secondary targets but still actively referenced.\n\n3. **60-Day Lookback**: The broadest institutional window — roughly one quarter. Major untouched PD arrays and liquidity pools at this range represent significant draw-on-liquidity targets.\n\nTo use IPDA data ranges: mark all unfilled FVGs, untested order blocks, and unswept liquidity pools within each lookback period. The algorithm prioritizes the nearest untouched levels first, then works outward. When all nearby levels are exhausted, the algorithm reaches for the 40-day and 60-day targets."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "How does the IPDA relate to PD arrays?"},
                {"role": "assistant", "content": "The IPDA and PD arrays are intimately connected — PD arrays are the destinations the IPDA delivers price to.\n\n**The relationship works like this:**\n\n1. **IPDA is the engine**: It determines where price needs to go — which liquidity pools to sweep, which inefficiencies to fill.\n\n2. **PD arrays are the targets**: Fair Value Gaps, order blocks, breaker blocks, mitigation blocks, and void areas are all PD arrays that the IPDA uses as waypoints for price delivery.\n\n3. **Premium vs Discount**: The IPDA classifies PD arrays as premium (above equilibrium — sell targets) or discount (below equilibrium — buy targets). The algorithm delivers price to discount PD arrays during bullish programs and premium PD arrays during bearish programs.\n\n4. **IPDA Data Ranges**: The algorithm references PD arrays within 20, 40, and 60-day lookback windows, prioritizing the nearest unfilled arrays.\n\nWhen analyzing a chart, first identify the IPDA's likely program (bullish or bearish based on HTF bias), then map available PD arrays as potential delivery targets."}
            ]
        },
    ])

    # ── Abbreviation resolution training (15 missing) ──
    abbr_examples = [
        ("BOS", "Break of Structure", "Break of Structure (BOS) occurs when price breaks a previous swing high (bullish BOS) or swing low (bearish BOS), confirming continuation of the current trend. Unlike a Change of Character (CHoCH) which signals reversal, BOS confirms that the existing directional bias remains intact. For valid BOS: price must close beyond the swing point with displacement — a single wick through is not sufficient. BOS is used to confirm trend direction before seeking entries at PD arrays within the established trend."),
        ("BPR", "Balanced Price Range", "A Balanced Price Range (BPR) forms when two overlapping Fair Value Gaps exist in opposite directions at the same price level. This creates a zone where both bullish and bearish inefficiencies overlap, making it a high-probability support/resistance area. The BPR represents equilibrium — a price level that has been validated from both directions. When price returns to a BPR, it often acts as a strong reaction zone because both buyers and sellers have institutional interest at that level."),
        ("BSL", "Buy Side Liquidity", "Buy Side Liquidity (BSL) refers to the pool of resting buy stop orders above swing highs, equal highs, and resistance levels. These orders are placed by retail traders as stop losses on short positions or breakout entries on long positions. The IPDA algorithm targets BSL by driving price above these levels to fill institutional sell orders. When you see BSL being swept (a quick spike above and reversal), it indicates smart money has used the retail orders as counter-party liquidity. BSL sweeps during premium pricing are particularly significant for bearish setups."),
        ("CHoCH", "Change of Character", "Change of Character (CHoCH) signals a potential trend reversal by breaking a key swing point in the opposite direction of the prevailing trend. In a bullish trend, CHoCH occurs when price breaks below a higher low with displacement. In a bearish trend, CHoCH occurs when price breaks above a lower high. CHoCH differs from BOS — BOS confirms continuation while CHoCH signals reversal. A valid CHoCH requires displacement (strong, aggressive candle) through the swing point, not just a wick. CHoCH often precedes a Market Structure Shift and is used as the first signal to look for entries in the new direction."),
        ("EQH", "Equal Highs", "Equal Highs (EQH) are two or more swing highs at approximately the same price level, creating a visible pool of buy side liquidity above them. Equal highs are significant because they represent a concentration of buy stop orders — retail traders place stops above these obvious resistance levels, and breakout traders place buy limits above them. The IPDA algorithm frequently runs equal highs to access this liquidity before reversing. When price sweeps EQH and immediately reverses with displacement, it confirms that smart money used the liquidity to fill sell orders — a bearish signal."),
        ("EQL", "Equal Lows", "Equal Lows (EQL) are two or more swing lows at approximately the same price level, creating a visible pool of sell side liquidity below them. Similar to equal highs but on the opposite side — retail traders place sell stops below obvious support levels, and breakdown traders place sell limits. The algorithm sweeps EQL to access sell side liquidity before a potential bullish reversal. EQL sweeps during discount pricing that align with a bullish HTF bias are high-probability long entry setups, especially when confirmed by displacement and an FVG forming immediately after the sweep."),
        ("ERL", "External Range Liquidity", "External Range Liquidity (ERL) refers to the liquidity resting outside the current trading range's boundaries — above the range high (buy side) or below the range low (sell side). ERL is the ultimate target of most price movements. The IRL→ERL framework states that price moves from Internal Range Liquidity (PD arrays inside the range) toward External Range Liquidity (the range extremes and beyond). Every trade setup should consider: is price moving FROM IRL TO ERL? This directional flow — using internal PD arrays as springboards toward external liquidity targets — is the foundation of the IRL/ERL framework."),
        ("IRL", "Internal Range Liquidity", "Internal Range Liquidity (IRL) refers to PD arrays located within the current trading range — Fair Value Gaps, order blocks, breakers, and other inefficiencies between the range high and range low. IRL serves as the entry mechanism. In the IRL→ERL framework, traders enter at IRL (a PD array inside the range) with the expectation that price will be delivered to ERL (the range boundary). For example: price sweeps ERL on the sell side, then delivers to an IRL order block where you enter long, targeting the opposing ERL on the buy side. IRL provides the precision entry while ERL provides the target."),
        ("IFVG", "Inversion Fair Value Gap", "An Inversion Fair Value Gap (IFVG) is a previously established FVG that has been traded through and now acts as support/resistance in the opposite direction. When a bullish FVG gets broken to the downside and price returns to it, it becomes a bearish IFVG — now acting as resistance instead of support. Similarly, a broken bearish FVG becomes a bullish IFVG. IFVGs are significant because they mark levels where the original institutional interest has been invalidated and the level now serves a new purpose. They are particularly effective as confirmation confluences within larger setups."),
        ("LRLR", "Low Resistance Liquidity Run", "A Low Resistance Liquidity Run (LRLR) occurs when price moves aggressively in one direction with minimal resistance — few PD arrays blocking its path. LRLR happens when the algorithm has a clear delivery target (usually ERL) and there are no significant unfilled FVGs, unmitigated order blocks, or other PD arrays in the way. Identifying LRLR conditions helps traders understand when price is likely to move quickly and decisively toward a target. The absence of PD arrays between current price and the target creates a 'highway' for price to travel efficiently."),
        ("NDOG", "New Day Opening Gap", "The New Day Opening Gap (NDOG) is the price gap between the previous day's settlement price and the new day's opening price. This gap represents overnight institutional positioning and is a significant PD array. The IPDA algorithm often seeks to fill or reference the NDOG during the trading day. NDOGs that align with other PD arrays (FVGs, order blocks) become high-probability reaction levels. Similar to NWOGs but on a daily timeframe, NDOGs are particularly relevant during the first killzone of the day."),
        ("NWOG", "New Week Opening Gap", "The New Week Opening Gap (NWOG) is the price gap between Friday's settlement and Monday's opening price. NWOGs represent significant institutional gaps and act as weekly PD arrays. The IPDA algorithm frequently delivers price back to NWOGs during the trading week, making them excellent reference levels. A NWOG in premium territory acts as resistance; in discount territory, as support. When price respects a NWOG level and shows displacement, it confirms the algorithm is using it as a reference point for the week's delivery program."),
        ("PDA", "PD Array", "PD Array (Premium/Discount Array) is the umbrella term for institutional price levels that the IPDA algorithm uses as delivery targets. PD Arrays include: Fair Value Gaps (FVGs), Order Blocks, Breaker Blocks, Mitigation Blocks, Rejection Blocks, Void areas, Volume Imbalances, and Liquidity Voids. Each PD array is classified as either premium (above 50% of the range — sell zone) or discount (below 50% — buy zone). The key principle: sell at premium PD arrays, buy at discount PD arrays. PD arrays provide the 'where' for entries, while killzones provide the 'when,' and displacement provides the 'confirmation.'"),
        ("SFP", "Swing Failure Pattern", "The Swing Failure Pattern (SFP) occurs when price attempts to break a previous swing high or low, wicks beyond it to sweep liquidity, but fails to close beyond it. The candle body closes back inside the range. SFP signals a failed breakout — the algorithm swept the liquidity (buy stops above highs or sell stops below lows) but smart money used those orders to fill in the opposite direction. A bearish SFP occurs above a swing high; a bullish SFP occurs below a swing low. SFP is most significant when it occurs at HTF PD arrays and during killzones, confirming the liquidity sweep was intentional rather than a genuine breakout."),
    ]

    for abbr, full_name, answer in abbr_examples:
        # Q using abbreviation
        examples.append({
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": f"What is {abbr} in ICT trading?"},
                {"role": "assistant", "content": answer}
            ]
        })
        # Q using full name
        examples.append({
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": f"Explain {full_name}."},
                {"role": "assistant", "content": answer}
            ]
        })

    # ── Cross-concept relationship examples (BSL + SSL + DOL) ──
    examples.extend([
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What is the relationship between BSL, SSL, and DOL?"},
                {"role": "assistant", "content": "BSL (Buy Side Liquidity), SSL (Sell Side Liquidity), and DOL (Draw on Liquidity) form a complete liquidity framework:\n\n**BSL** — Buy stop orders resting above swing highs, equal highs, and resistance. This is the liquidity pool on the buy side.\n\n**SSL** — Sell stop orders resting below swing lows, equal lows, and support. This is the liquidity pool on the sell side.\n\n**DOL** — The specific liquidity target that the IPDA algorithm is currently drawing price toward. DOL can be either BSL or SSL depending on the current delivery program.\n\n**How they connect:**\n1. Identify where BSL and SSL pools exist on the chart\n2. Determine the HTF bias (bullish = DOL is BSL; bearish = DOL is SSL)\n3. The DOL tells you WHERE price is heading\n4. The opposite liquidity pool tells you what will be SWEPT for entry\n\nExample: If HTF is bearish, DOL = SSL. Price will first sweep BSL (to trap longs and provide sell-side counter-parties), then deliver to SSL. Your entry comes after the BSL sweep, targeting the SSL as your DOL."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "How do OB and FVG combine to create a Unicorn setup?"},
                {"role": "assistant", "content": "The Unicorn setup is one of ICT's highest-probability entry patterns, formed by the overlap of an Order Block (OB) and a Fair Value Gap (FVG):\n\n**How it forms:**\n1. Price creates displacement — a strong, aggressive candle move\n2. The displacement leaves behind an FVG (3-candle pattern with a gap)\n3. Within or adjacent to that FVG, there's an OB (the last opposing candle before the displacement)\n4. When the FVG and OB overlap at the same price level, a Unicorn is born\n\n**Why it's powerful:**\n- The OB represents institutional positioning (where smart money placed orders)\n- The FVG represents an inefficiency the algorithm wants to fill\n- Their overlap creates a double-confirmation zone\n- When price returns to this zone, both the OB reaction AND FVG fill trigger simultaneously\n\n**Trading the Unicorn:**\n- Wait for price to retrace to the OB+FVG overlap zone\n- Enter when you see a reaction (candle rejection or smaller-timeframe CISD)\n- Stop loss goes beyond the OB\n- Target the opposing liquidity pool or next PD array"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are VEX, an ICT trading expert."},
                {"role": "user", "content": "What is the IRL to ERL framework?"},
                {"role": "assistant", "content": "The IRL→ERL (Internal Range Liquidity to External Range Liquidity) framework is ICT's master classification for all price movement:\n\n**Core principle:** Price always moves from IRL to ERL or from ERL to IRL.\n\n**Type A trades (IRL→ERL):**\n- Entry at an internal PD array (FVG, OB inside the range)\n- Target at external liquidity (range high or low)\n- This is the classic continuation trade — price uses internal levels as springboards\n\n**Type B trades (ERL→IRL / Turtle Soup):**\n- Price sweeps external liquidity (breaks the range)\n- Reversal entry targeting internal PD arrays\n- This is the reversal/trap trade — the breakout is fake\n\n**Application:**\n1. Define the current dealing range (recent swing high to swing low)\n2. Mark IRL: FVGs, OBs, breakers inside the range\n3. Mark ERL: the range highs and lows, plus equal highs/lows beyond\n4. Determine if price is moving FROM IRL TO ERL (trend) or FROM ERL TO IRL (reversal)\n5. Enter at the source, target the destination\n\nEvery ICT setup is fundamentally either Type A or Type B."}
            ]
        },
    ])

    return examples


# ── Step 3: Build v8 dataset ─────────────────────────────────────────────────

def main():
    train_file = ROOT / "finetune_data_final" / "train.jsonl"
    valid_file = ROOT / "finetune_data_final" / "valid.jsonl"
    test_file = ROOT / "finetune_data_final" / "test.jsonl"

    print("=" * 60)
    print("  v8 Training Data Pipeline")
    print("=" * 60)

    # Step 1: Find bad examples
    print("\n[1/4] Scanning for bad examples...")
    bad_indices, all_train = find_bad_examples(train_file)
    print(f"\n  Found {len(bad_indices)} bad examples out of {len(all_train)}")

    # Step 2: Remove bad examples
    clean_train = [ex for i, ex in enumerate(all_train) if i not in bad_indices]
    print(f"  Clean base: {len(clean_train)} examples")

    # Step 3: Generate new examples
    print("\n[2/4] Generating v8 examples...")
    new_examples = generate_v8_examples()
    print(f"  Generated {len(new_examples)} new examples")
    print(f"    - CISD fixes: 3")
    print(f"    - IPDA fixes: 3")
    print(f"    - Abbreviation pairs: 14 x 2 = 28")
    print(f"    - Cross-concept: 3")

    # Step 4: Merge and save
    print("\n[3/4] Building v8 dataset...")
    combined = clean_train + new_examples
    random.seed(42)
    random.shuffle(combined)

    # Split: 90% train, 5% valid, 5% test
    n = len(combined)
    n_valid = max(1, n // 20)
    n_test = max(1, n // 20)
    n_train = n - n_valid - n_test

    train_set = combined[:n_train]
    valid_set = combined[n_train:n_train + n_valid]
    test_set = combined[n_train + n_valid:]

    # Also read and preserve existing valid/test (they're separate)
    existing_valid = [json.loads(l) for l in open(valid_file)]
    existing_test = [json.loads(l) for l in open(test_file)]

    # Merge: new valid/test get added to existing
    final_valid = existing_valid + valid_set
    final_test = existing_test + test_set

    # Write v8
    out_dir = ROOT / "finetune_data_v8"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "train.jsonl", "w") as f:
        for ex in train_set:
            f.write(json.dumps(ex) + "\n")

    with open(out_dir / "valid.jsonl", "w") as f:
        for ex in final_valid:
            f.write(json.dumps(ex) + "\n")

    with open(out_dir / "test.jsonl", "w") as f:
        for ex in final_test:
            f.write(json.dumps(ex) + "\n")

    print(f"\n[4/4] v8 dataset written to {out_dir}/")
    print(f"  Train: {len(train_set)}")
    print(f"  Valid: {len(final_valid)}")
    print(f"  Test:  {len(final_test)}")
    print(f"  Total: {len(train_set) + len(final_valid) + len(final_test)}")


if __name__ == "__main__":
    main()
