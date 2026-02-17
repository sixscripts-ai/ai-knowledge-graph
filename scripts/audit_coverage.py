#!/usr/bin/env python3
"""
ICT Knowledge Gap Auditor
==========================
Cross-references:
  1. Graph coverage (ict_graph_final.json — 10K nodes, 23K edges)
  2. Existing synthetic training data (24 concepts in generate_synthetic_data.py)
  3. Master concept inventory from all 3 repos (train-ict, ai-knowledge-graph, Antigravity)

Outputs a prioritized gap report showing:
  - Concepts IN graph but NOT in training data
  - Concepts NOT in graph at all
  - Concepts with thin coverage (few triples)
  - Priority ranking for synthetic data generation
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_FILE = PROJECT_ROOT / "ict_graph_final.json"
TRAIN_ICT = Path.home() / "Documents" / "train-ict"
ANTIGRAVITY = Path.home() / "Antigravity"

# ── Master Concept Inventory ─────────────────────────────────────────────────
# Every ICT concept found across all 3 repos, organized by category
# These are the concepts the system SHOULD know about

MASTER_CONCEPTS = {
    # ── Price Action / PD Arrays ──────────────────────────────────────────
    "price_action": [
        "fair_value_gap", "inversion_fair_value_gap", "first_presented_fvg",
        "order_block", "bullish_order_block", "bearish_order_block",
        "reclaimed_order_block", "breaker_block", "mitigation_block",
        "propulsion_block", "rejection_block", "vacuum_block", "suspension_block",
        "displacement", "balanced_price_range", "volume_imbalance",
        "liquidity_void", "opening_gap", "new_week_opening_gap", "new_day_opening_gap",
        "consequent_encroachment", "mean_threshold", "pd_array_matrix",
        "premium_zone", "discount_zone", "equilibrium",
        "optimal_trade_entry", "fibonacci_retracement",
        "standard_deviation_projections", "expansion_price_swing",
        "separation_principle", "flash_wick", "immediate_rebalance",
        "efficient_price_delivery", "inefficient_price_delivery", "rebalancing",
        "sibi", "bisi", "dealing_range", "micro_dealing_ranges",
        "settlement_levels", "daily_range_expansion",
    ],

    # ── Liquidity ─────────────────────────────────────────────────────────
    "liquidity": [
        "liquidity", "buy_side_liquidity", "sell_side_liquidity",
        "external_range_liquidity", "internal_range_liquidity",
        "low_resistance_liquidity_run", "liquidity_sweep", "liquidity_pool",
        "draw_on_liquidity", "run_on_liquidity", "equal_highs", "equal_lows",
        "relative_equal_highs_lows", "trendline_liquidity", "stop_hunt",
        "buy_side_delivery", "sell_side_delivery",
        "high_resistance_liquidity_zone", "low_support_liquidity_zone",
        "liquidity_exhaustion", "pairing_of_orders",
        "liquidity_distribution_profile",
    ],

    # ── Market Structure ──────────────────────────────────────────────────
    "market_structure": [
        "market_structure", "break_of_structure", "market_structure_shift",
        "change_of_character", "swing_high", "swing_low",
        "higher_high", "higher_low", "lower_high", "lower_low",
        "support_and_resistance_inversion", "reversal_sequence",
        "swing_failure_pattern", "fractal_nature", "fractals",
        "break_vs_stab", "stop_run_vs_true_break",
    ],

    # ── Time / Sessions ───────────────────────────────────────────────────
    "time_sessions": [
        "killzones", "london_killzone", "new_york_am_killzone",
        "new_york_pm_killzone", "asian_killzone", "london_close_killzone",
        "macro_times", "algorithm_macros", "silver_bullet_window",
        "cbdr", "asian_range", "london_open", "london_lunch",
        "new_york_open", "cme_open", "dead_zone",
        "ipda_true_day", "midnight_open", "nine_thirty_open",
        "sunday_opening_price", "tuesday_high_low",
        "two_pm_rule", "ninety_minute_cycles", "quarterly_shifts",
        "seasonal_tendencies", "electronic_trading_hours",
        "regular_trading_hours", "session_analysis",
    ],

    # ── Trading Models ────────────────────────────────────────────────────
    "trading_models": [
        "power_of_three", "accumulation", "manipulation", "distribution",
        "silver_bullet", "judas_swing", "turtle_soup", "turtle_soup_plus_one",
        "market_maker_buy_model", "market_maker_sell_model",
        "ict_2022_model", "unicorn_setup", "a_plus_setup",
        "model_7_universal", "model_8_25_pips", "model_9_one_shot_one_kill",
        "model_11_30_pips_intraday", "model_12_scalping",
        "cbdr_asia_sd_method", "ob_fvg_retrace_entry",
        "htf_sweep_ltf_entry", "breaker_block_reversal",
        "ict_reversal_model", "ict_continuation_model",
        "ict_liquidity_run_model", "ict_stop_hunt_model",
        "ict_displacement_model", "ict_scalp_model",
        "london_open_manipulation_model", "day_trading_model",
    ],

    # ── Algorithmic / Institutional ───────────────────────────────────────
    "institutional": [
        "ipda", "the_algorithm", "smart_money_reversal",
        "institutional_order_flow", "smt_divergence",
        "market_maker_templating", "weekly_profiles",
        "type_a_buy_day", "type_b_sell_day",
        "smart_money", "change_in_state_of_delivery",
        "data_ranges_lookback", "position_cycling", "volatility_injection",
        "economic_calendar_smokescreen", "intermarket_analysis",
        "daily_bias", "confluence", "confluence_scoring",
        "interbank_order_flow", "supply_demand_distinction",
        "reaccumulation", "redistribution", "second_stage_distribution",
    ],

    # ── Risk Management / Execution ───────────────────────────────────────
    "risk_management": [
        "position_size_formula", "r_percent_management", "two_strike_rule",
        "max_three_trades_day", "min_two_to_one_rr", "partial_exits",
        "five_stage_trade_plan", "pre_trade_scoring",
        "adr_targeting", "stop_placement_rules", "psychology_monitoring",
        "equity_curve_leveling",
    ],

    # ── Anti-Patterns ─────────────────────────────────────────────────────
    "anti_patterns": [
        "false_order_block", "chased_move", "asia_breakout_trap",
        "mitigated_fvg_one_time", "fighting_htf", "news_gamble",
        "revenge_trade", "overtrading",
    ],
}

# Concepts already covered by synthetic training data (generate_synthetic_data.py)
ALREADY_TRAINED = {
    "fair_value_gap", "order_block", "liquidity", "displacement",
    "market_structure", "killzones", "judas_swing", "cbdr",
    "smart_money", "power_of_three", "pd_array_matrix", "smt_divergence",
    "market_maker_sell_model", "market_maker_buy_model", "optimal_trade_entry",
    "breaker_block", "mitigation_block", "silver_bullet", "asian_range",
    "institutional_order_flow", "liquidity_pool", "flash_wick",
    "manipulation", "market_structure_shift",
    # Also includes 7 scenario/reasoning examples
}


def normalize(text: str) -> str:
    """Normalize concept name for matching."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9_]', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    return text


def load_graph_subjects() -> Counter:
    """Load all unique subjects from ict_graph_final.json and count triples per subject."""
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE) as f:
        data = json.load(f)

    triples = data if isinstance(data, list) else data.get("triples", data.get("results", []))

    subject_counts = Counter()
    all_terms = set()

    for triple in triples:
        if isinstance(triple, dict):
            subj = normalize(triple.get("subject", ""))
            pred = normalize(triple.get("predicate", ""))
            obj = normalize(triple.get("object", ""))
        elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
            subj = normalize(str(triple[0]))
            pred = normalize(str(triple[1]))
            obj = normalize(str(triple[2]))
        else:
            continue

        if subj:
            subject_counts[subj] += 1
            all_terms.add(subj)
        if obj:
            all_terms.add(obj)

    return subject_counts, all_terms


def load_train_ict_concepts() -> set:
    """Load concepts from train-ict repo's knowledge_base."""
    concepts = set()

    # concept_relationships.yaml
    cr_file = TRAIN_ICT / "knowledge_base" / "concept_relationships.yaml"
    if cr_file.exists():
        text = cr_file.read_text()
        # Extract concept names from YAML keys and values
        for match in re.findall(r'^\s*-?\s*(\w[\w\s]*\w):', text, re.MULTILINE):
            concepts.add(normalize(match))
        for match in re.findall(r'concept:\s*"?([^"\n]+)"?', text):
            concepts.add(normalize(match))

    # terminology.yaml
    term_file = TRAIN_ICT / "knowledge_base" / "definitions" / "terminology.yaml"
    if term_file.exists():
        text = term_file.read_text()
        for match in re.findall(r'^\s*(\w[\w\s]*\w):', text, re.MULTILINE):
            concepts.add(normalize(match))

    # concept .md files
    concept_dir = TRAIN_ICT / "knowledge_base" / "concepts"
    if concept_dir.exists():
        for f in concept_dir.glob("*.md"):
            name = f.stem.replace("-", "_").replace(" ", "_")
            concepts.add(normalize(name))

    # learned_concepts.json
    learned = TRAIN_ICT / "data" / "learning" / "learned_concepts.json"
    if learned.exists():
        try:
            data = json.loads(learned.read_text())
            if isinstance(data, dict):
                for key in data:
                    concepts.add(normalize(key))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        concepts.add(normalize(item))
                    elif isinstance(item, dict):
                        name = item.get("name", item.get("concept", ""))
                        if name:
                            concepts.add(normalize(name))
        except:
            pass

    # playbooks
    playbook_dir = TRAIN_ICT / "data" / "playbooks"
    if playbook_dir.exists():
        for f in playbook_dir.glob("*.yaml"):
            name = f.stem.replace("-", "_").replace(" ", "_")
            concepts.add(normalize(name))

    # concept_graph.json
    cg_file = TRAIN_ICT / "knowledge_base" / "graphs" / "concept_graph.json"
    if cg_file.exists():
        try:
            data = json.loads(cg_file.read_text())
            nodes = data.get("nodes", [])
            for node in nodes:
                if isinstance(node, dict):
                    name = node.get("id", node.get("name", ""))
                    if name:
                        concepts.add(normalize(name))
                elif isinstance(node, str):
                    concepts.add(normalize(node))
        except:
            pass

    return concepts


def load_antigravity_concepts() -> set:
    """Load concepts from Antigravity repo's knowledge_base."""
    concepts = set()

    # concept .md files
    concept_dir = ANTIGRAVITY / "knowledge_base" / "concepts"
    if concept_dir.exists():
        for f in concept_dir.glob("*.md"):
            name = f.stem.replace("-", "_").replace(" ", "_")
            concepts.add(normalize(name))

    # definitions
    for yaml_name in ["terminology.yaml", "concept_relationships.yaml"]:
        yf = ANTIGRAVITY / "knowledge_base" / "definitions" / yaml_name
        if yf.exists():
            text = yf.read_text()
            for match in re.findall(r'^\s*(\w[\w\s]*\w):', text, re.MULTILINE):
                concepts.add(normalize(match))

    # learned_concepts.json
    learned = ANTIGRAVITY / "data" / "learning" / "learned_concepts.json"
    if learned.exists():
        try:
            data = json.loads(learned.read_text())
            if isinstance(data, dict):
                for key in data:
                    concepts.add(normalize(key))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        concepts.add(normalize(item))
                    elif isinstance(item, dict):
                        name = item.get("name", item.get("concept", ""))
                        if name:
                            concepts.add(normalize(name))
        except:
            pass

    # rapid_fire_drill.md
    drill = ANTIGRAVITY / "knowledge_base" / "rapid_fire_drill.md"
    if drill.exists():
        text = drill.read_text()
        # Extract Q&A concept mentions
        for match in re.findall(r'\*\*([^*]+)\*\*', text):
            if len(match) > 3 and len(match) < 60:
                concepts.add(normalize(match))

    # models
    model_dir = ANTIGRAVITY / "knowledge_base" / "models"
    if model_dir.exists():
        for f in model_dir.glob("*.md"):
            name = f.stem.replace("-", "_").replace(" ", "_")
            concepts.add(normalize(name))

    return concepts


def fuzzy_match(concept: str, term_set: set, threshold: float = 0.6) -> list:
    """Find fuzzy matches for a concept in a set of terms."""
    matches = []
    concept_words = set(concept.split("_"))

    for term in term_set:
        # Exact match
        if concept == term:
            matches.append((term, 1.0))
            continue

        # Substring match
        if concept in term or term in concept:
            matches.append((term, 0.9))
            continue

        # Word overlap
        term_words = set(term.split("_"))
        if not concept_words or not term_words:
            continue
        overlap = concept_words & term_words
        score = len(overlap) / max(len(concept_words), len(term_words))
        if score >= threshold:
            matches.append((term, score))

    return sorted(matches, key=lambda x: -x[1])[:3]


def main():
    print("=" * 70)
    print("ICT KNOWLEDGE GAP AUDITOR")
    print("=" * 70)

    # ── 1. Load graph ──────────────────────────────────────────────────────
    graph_counts, graph_terms = load_graph_subjects()
    print(f"\n📊 Graph: {len(graph_counts)} unique subjects, {sum(graph_counts.values())} total triples")
    print(f"   All terms (subjects + objects): {len(graph_terms)}")

    # ── 2. Load repo concepts ─────────────────────────────────────────────
    train_ict_concepts = load_train_ict_concepts()
    antigravity_concepts = load_antigravity_concepts()
    print(f"\n📚 train-ict concepts: {len(train_ict_concepts)}")
    print(f"📚 Antigravity concepts: {len(antigravity_concepts)}")
    print(f"📚 Already in training data: {len(ALREADY_TRAINED)}")

    # ── 3. Flatten master list ────────────────────────────────────────────
    all_master = set()
    for category, concepts in MASTER_CONCEPTS.items():
        all_master.update(concepts)
    print(f"\n🎯 Master concept list: {len(all_master)} unique concepts")

    # ── 4. Coverage Analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS")
    print("=" * 70)

    # For each category, check what's in graph, what's trained, what's missing
    gap_report = {}
    priority_gaps = []  # (concept, category, in_graph, triple_count, in_training)

    for category, concepts in MASTER_CONCEPTS.items():
        cat_in_graph = 0
        cat_in_training = 0
        cat_missing = []

        for concept in concepts:
            # Check if in graph (fuzzy)
            graph_matches = fuzzy_match(concept, graph_terms)
            in_graph = len(graph_matches) > 0 and graph_matches[0][1] >= 0.7
            triple_count = 0
            if in_graph:
                # Get actual triple count
                best_match = graph_matches[0][0]
                triple_count = graph_counts.get(best_match, 0)
                cat_in_graph += 1

            # Check if in training data
            in_training = concept in ALREADY_TRAINED
            if in_training:
                cat_in_training += 1

            if not in_training:
                priority_gaps.append({
                    "concept": concept,
                    "category": category,
                    "in_graph": in_graph,
                    "triple_count": triple_count,
                    "graph_matches": graph_matches[:1] if graph_matches else [],
                    "in_train_ict": any(m[1] >= 0.6 for m in fuzzy_match(concept, train_ict_concepts)),
                    "in_antigravity": any(m[1] >= 0.6 for m in fuzzy_match(concept, antigravity_concepts)),
                })

        total = len(concepts)
        print(f"\n{'─' * 50}")
        print(f"  {category.upper().replace('_', ' ')}")
        print(f"  Graph: {cat_in_graph}/{total} | Trained: {cat_in_training}/{total} | Gaps: {total - cat_in_training}")

    # ── 5. Priority Ranking ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRIORITY GAPS (not in training data)")
    print("=" * 70)

    # Score: higher = more important to add
    # +3 if in graph (we have facts to build from)
    # +2 if in train-ict or antigravity repos (curriculum importance)
    # +1 per 5 triples in graph (depth bonus, capped at +3)
    # +2 if core category (price_action, liquidity, trading_models)
    for gap in priority_gaps:
        score = 0
        if gap["in_graph"]:
            score += 3
        if gap["in_train_ict"]:
            score += 2
        if gap["in_antigravity"]:
            score += 2
        score += min(3, gap["triple_count"] // 5)
        if gap["category"] in ("price_action", "liquidity", "trading_models", "market_structure"):
            score += 2
        gap["priority_score"] = score

    # Sort by priority
    priority_gaps.sort(key=lambda x: -x["priority_score"])

    # Print Tier 1 (score >= 7)
    print(f"\n{'─'*50}")
    print("  TIER 1 — CRITICAL (score ≥ 7)")
    print(f"{'─'*50}")
    tier1 = [g for g in priority_gaps if g["priority_score"] >= 7]
    for g in tier1:
        graph_tag = f"IN GRAPH ({g['triple_count']} triples)" if g["in_graph"] else "NOT IN GRAPH"
        repo_tags = []
        if g["in_train_ict"]:
            repo_tags.append("train-ict")
        if g["in_antigravity"]:
            repo_tags.append("antigravity")
        repos = f" | Sources: {', '.join(repo_tags)}" if repo_tags else ""
        print(f"  [{g['priority_score']:2d}] {g['concept']:<40s} {graph_tag}{repos}")

    # Tier 2 (score 4-6)
    print(f"\n{'─'*50}")
    print("  TIER 2 — IMPORTANT (score 4-6)")
    print(f"{'─'*50}")
    tier2 = [g for g in priority_gaps if 4 <= g["priority_score"] <= 6]
    for g in tier2:
        graph_tag = f"IN GRAPH ({g['triple_count']} triples)" if g["in_graph"] else "NOT IN GRAPH"
        print(f"  [{g['priority_score']:2d}] {g['concept']:<40s} {graph_tag}")

    # Tier 3 (score 1-3)
    print(f"\n{'─'*50}")
    print("  TIER 3 — NICE TO HAVE (score 1-3)")
    print(f"{'─'*50}")
    tier3 = [g for g in priority_gaps if g["priority_score"] <= 3]
    for g in tier3:
        graph_tag = f"IN GRAPH ({g['triple_count']} triples)" if g["in_graph"] else "NOT IN GRAPH"
        print(f"  [{g['priority_score']:2d}] {g['concept']:<40s} {graph_tag}")

    # ── 6. Summary Stats ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_concepts = len(all_master)
    trained = len([g for g in priority_gaps if False]) + len(ALREADY_TRAINED)  # already trained count
    in_graph_not_trained = len([g for g in priority_gaps if g["in_graph"]])
    not_in_graph = len([g for g in priority_gaps if not g["in_graph"]])

    print(f"  Total ICT concepts identified:     {total_concepts}")
    print(f"  Already in training data:           {len(ALREADY_TRAINED)}")
    print(f"  Coverage:                           {len(ALREADY_TRAINED)/total_concepts*100:.1f}%")
    print(f"  Gaps total:                         {len(priority_gaps)}")
    print(f"    → In graph (easy to add):         {in_graph_not_trained}")
    print(f"    → Not in graph (need sourcing):   {not_in_graph}")
    print(f"  Tier 1 critical gaps:               {len(tier1)}")
    print(f"  Tier 2 important gaps:              {len(tier2)}")
    print(f"  Tier 3 nice-to-have:                {len(tier3)}")

    # ── 7. Write JSON report ──────────────────────────────────────────────
    report = {
        "summary": {
            "total_concepts": total_concepts,
            "trained": len(ALREADY_TRAINED),
            "coverage_pct": round(len(ALREADY_TRAINED) / total_concepts * 100, 1),
            "gaps_total": len(priority_gaps),
            "in_graph_not_trained": in_graph_not_trained,
            "not_in_graph": not_in_graph,
            "tier1_count": len(tier1),
            "tier2_count": len(tier2),
            "tier3_count": len(tier3),
        },
        "tier1_gaps": [{"concept": g["concept"], "category": g["category"],
                        "score": g["priority_score"], "in_graph": g["in_graph"],
                        "triple_count": g["triple_count"]} for g in tier1],
        "tier2_gaps": [{"concept": g["concept"], "category": g["category"],
                        "score": g["priority_score"], "in_graph": g["in_graph"],
                        "triple_count": g["triple_count"]} for g in tier2],
        "tier3_gaps": [{"concept": g["concept"], "category": g["category"],
                        "score": g["priority_score"]} for g in tier3],
        "already_trained": sorted(ALREADY_TRAINED),
    }

    report_path = PROJECT_ROOT / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    return priority_gaps


if __name__ == "__main__":
    main()
