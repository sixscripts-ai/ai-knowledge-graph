"""
Training Data Generator V3
===========================
High-quality ICT fine-tuning data with rich, detailed responses (200-800+ chars).

Strategy: Draw from curated YAML/JSON sources instead of noisy graph triples.
Each category uses enriched templates that combine multiple data points per example.

Target: 1,000+ examples across 14 categories:
  1. Core Concept Definitions (~80)     - Deep ICT definitions from ontology
  2. Concept Requirements (~60)         - What each concept needs, enhancers, invalidators
  3. Model Deep Dives (~100)            - 5 models × 20 variations
  4. Causal Chain Reasoning (~80)       - Step-by-step with failure analysis
  5. Anti-Pattern Education (~80)       - What NOT to do, with detailed reasons
  6. Trade Scenario Analysis (~100)     - Real trades decomposed into Q&A
  7. Negative Trade Analysis (~60)      - Failure breakdown
  8. Confluence Scoring (~50)           - Weighted factor walkthroughs
  9. Time & Session Rules (~60)         - Killzones, macro times, session behavior
 10. Pair-Specific Knowledge (~40)      - Per-pair characteristics
 11. Pre-trade Validation (~50)         - Checklist-style reasoning
 12. Multi-TF Alignment (~50)          - HTF→MTF→LTF methodology
 13. Disambiguation (~40)               - ICT-specific term clarification
 14. Existing QA (enriched) (~50)       - Imported + enhanced

All responses minimum 200 chars, target 300-600 chars.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# ── Paths ──────────────────────────────────────────────────────────────────────
_TRAIN_ICT = Path(
    os.environ.get("TRAIN_ICT_ROOT", Path.home() / "Documents" / "train-ict")
)
_OUTPUT_DIR = Path(
    os.environ.get(
        "TRAINING_OUTPUT_V3",
        Path.home()
        / "Documents"
        / "knowledge_graph_ict"
        / "ai-knowledge-graph"
        / "training_output_v3",
    )
)

POSITIVE_DIR = _TRAIN_ICT / "data" / "training" / "positive"
NEGATIVE_DIR = _TRAIN_ICT / "data" / "training" / "negative"
QA_DIR = _TRAIN_ICT / "data" / "training" / "qa"
CONCEPT_RELS_PATH = _TRAIN_ICT / "data" / "schemas" / "concept_relationships.yaml"
ONTOLOGY_PATH = _TRAIN_ICT / "data" / "schemas" / "ict_ontology.yaml"


class TrainingDataGeneratorV3:
    """V3 fine-tuning data generator — rich, detailed ICT training examples."""

    SYSTEM_PROMPT = (
        "You are VEX, an expert ICT (Inner Circle Trader) trading assistant. "
        "You analyze market structure, price delivery arrays, liquidity, and "
        "time-based models to identify high-probability trade setups. "
        "You reason step-by-step through causal chains and confluence factors."
    )

    MIN_RESPONSE_LEN = 150  # chars — reject anything shorter

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.examples: list[dict] = []
        self._concept_rels: Optional[dict] = None
        self._ontology: Optional[dict] = None

    # ── Data Loaders ──────────────────────────────────────────────────────

    def _load_concept_rels(self) -> dict:
        if self._concept_rels is not None:
            return self._concept_rels
        if CONCEPT_RELS_PATH.exists() and yaml:
            with open(CONCEPT_RELS_PATH) as f:
                self._concept_rels = yaml.safe_load(f) or {}
        else:
            self._concept_rels = {}
        return self._concept_rels

    def _load_ontology(self) -> dict:
        if self._ontology is not None:
            return self._ontology
        if ONTOLOGY_PATH.exists() and yaml:
            with open(ONTOLOGY_PATH) as f:
                self._ontology = yaml.safe_load(f) or {}
        else:
            self._ontology = {}
        return self._ontology

    def _load_trades(self, directory: Path) -> list[dict]:
        trades = []
        if not directory.exists():
            return trades
        for f in sorted(directory.glob("*.json")):
            try:
                with open(f) as fh:
                    trades.append(json.load(fh))
            except Exception:
                continue
        return trades

    def _label(self, s: str) -> str:
        """Convert snake_case to Title Case."""
        return s.replace("_", " ").title()

    # ── Main entry ────────────────────────────────────────────────────────

    def generate_all(self) -> int:
        """Generate all v3 training data. Returns total count."""
        self.examples = []

        counts = {}
        generators = [
            ("Core Concepts", self.gen_core_concepts),
            ("Concept Requirements", self.gen_concept_requirements),
            ("Model Deep Dives", self.gen_model_deep_dives),
            ("Causal Chains", self.gen_causal_chains),
            ("Anti-Patterns", self.gen_anti_patterns),
            ("Trade Scenarios", self.gen_trade_scenarios),
            ("Negative Trades", self.gen_negative_trades),
            ("Confluence Scoring", self.gen_confluence_scoring),
            ("Time & Sessions", self.gen_time_sessions),
            ("Pair Knowledge", self.gen_pair_knowledge),
            ("Pre-trade Validation", self.gen_pretrade_validation),
            ("Multi-TF Alignment", self.gen_multi_tf),
            ("Disambiguation", self.gen_disambiguation),
            ("Existing QA", self.gen_existing_qa),
        ]

        for name, gen_fn in generators:
            n = gen_fn()
            counts[name] = n
            print(f"[v3] {name}: {n}")

        total = sum(counts.values())
        print(f"\n[v3] TOTAL: {total} examples")

        # Quality check
        short = sum(
            1 for ex in self.examples if len(ex["output"]) < self.MIN_RESPONSE_LEN
        )
        if short > 0:
            print(
                f"[v3] WARNING: {short} examples below {self.MIN_RESPONSE_LEN} chars — filtering..."
            )
            self.examples = [
                ex for ex in self.examples if len(ex["output"]) >= self.MIN_RESPONSE_LEN
            ]
            print(f"[v3] After filter: {len(self.examples)} examples")

        return len(self.examples)

    # ── 1. Core Concept Definitions ───────────────────────────────────────

    def gen_core_concepts(self) -> int:
        count = 0
        onto = self._load_ontology()
        rels = self._load_concept_rels()

        # Market Structure concepts from ontology
        structures = onto.get("structures", {})
        for name, info in structures.items():
            defn = info.get("definition", "")
            sig = info.get("significance", "")
            rule = info.get("rule", "")
            full_name = info.get("full_name", "")

            label = full_name or self._label(name)
            parts = [
                f"In ICT (Inner Circle Trader) methodology, {label} ({name.upper()}) "
                if full_name
                else f"In ICT methodology, {label} "
            ]
            if defn:
                parts.append(f"is defined as: {defn}. ")
            if sig:
                parts.append(f"Its significance is that it represents {sig.lower()}. ")
            if rule:
                parts.append(f"Key rule: {rule}. ")

            # Enrich with related info
            extra = info.get("valid", info.get("location", info.get("contains", "")))
            if extra:
                parts.append(f"Additional detail: {extra}. ")

            parts.append(
                "Understanding this concept is essential for reading price action correctly in institutional trading."
            )

            for q_template in [
                f"What is {label} in ICT trading?",
                f"Define {label} in the context of ICT methodology.",
            ]:
                self._add(q_template, "".join(parts), "core_concept")
                count += 1

        # Liquidity concepts
        liq = onto.get("liquidity", {})
        for name, info in liq.items():
            if not isinstance(info, dict):
                continue
            full_name = info.get("full_name", "")
            defn = info.get("definition", "")
            loc = info.get("location", "")
            contains = info.get("contains", "")
            sig = info.get("significance", "")
            rule = info.get("rule", "")
            nature = info.get("nature", "")

            label = full_name or self._label(name)
            parts = [f"In ICT methodology, {label} ({name.upper()}) "]
            if defn:
                parts.append(f"refers to {defn.lower()}. ")
            if loc:
                parts.append(f"It is found {loc.lower()}. ")
            if contains:
                parts.append(f"It contains {contains.lower()}. ")
            if sig:
                parts.append(f"Its significance: {sig}. ")
            if rule:
                parts.append(f"Key rule: {rule}. ")
            if nature:
                parts.append(f"Important: {nature}. ")

            # Add behavior if it's the main liquidity entry
            if name == "liquidity":
                behaviors = info.get("behavior", [])
                if behaviors:
                    parts.append("Core principles: " + "; ".join(behaviors) + ". ")
                seq = info.get("sequence_after_sweep", {})
                if seq:
                    parts.append(
                        "After a sweep: "
                        + " → ".join(f"{k}. {v}" for k, v in sorted(seq.items()))
                        + ". "
                    )
                ki = info.get("key_insight", "")
                if ki:
                    parts.append(f"Key insight: {ki}")

            self._add(
                f"What is {label} in ICT trading?", "".join(parts), "core_concept"
            )
            count += 1

        # PD Arrays
        pd = onto.get("pd_arrays", {})
        for name, info in pd.items():
            if not isinstance(info, dict):
                continue
            full_name = info.get("full_name", "")
            defn = info.get("definition", "")
            label = full_name or self._label(name)

            parts = [
                f"{label} is a Price Delivery Array (PD Array) in ICT methodology. "
            ]
            if defn:
                parts.append(f"Definition: {defn}. ")
            for field in [
                "measurement",
                "priority",
                "entry_rule",
                "entry",
                "stop",
                "bullish",
                "bearish",
                "mitigated_when",
                "action",
            ]:
                val = info.get(field, "")
                if val:
                    parts.append(f"{self._label(field)}: {val}. ")
            for field in ["status"]:
                val = info.get(field, [])
                if val:
                    parts.append(f"Can be: {', '.join(val)}. ")

            parts.append(
                "PD Arrays are the core building blocks of ICT price analysis — they tell you WHERE to trade."
            )

            self._add(
                f"Explain {label} as a PD Array in ICT.", "".join(parts), "core_concept"
            )
            count += 1

        # Timing concepts
        timing = onto.get("timing", {})
        for name, info in timing.items():
            if not isinstance(info, dict):
                continue
            full_name = info.get("full_name", "")
            defn = info.get("definition", "")
            label = full_name or self._label(name)

            parts = [f"In ICT methodology, {label} "]
            if defn:
                parts.append(f"is {defn.lower()}. ")
            for field in [
                "time",
                "ideal_range",
                "rule",
                "timing",
                "purpose",
                "significance",
                "tight_cbdr",
            ]:
                val = info.get(field, "")
                if val:
                    parts.append(f"{self._label(field)}: {val}. ")

            # Special handling for timing entries with sub-dicts
            if isinstance(info, dict):
                for sub_key in ["london", "ny", "cme"]:
                    sub_val = info.get(sub_key, "")
                    if sub_val and isinstance(sub_val, str):
                        parts.append(f"{self._label(sub_key)}: {sub_val}. ")

            parts.append(
                "Time is one of the three pillars of ICT trading — knowing WHEN to trade is as important as WHERE."
            )

            self._add(
                f"What is {label} in ICT timing methodology?",
                "".join(parts),
                "core_concept",
            )
            count += 1

        # Model stages (AMD)
        stages = onto.get("model_stages", {})
        for name, info in stages.items():
            if not isinstance(info, dict):
                continue
            label = self._label(name)
            defn = info.get("definition", "")
            appearance = info.get("appearance", "")
            action = info.get("action", "")

            resp = (
                f"In the ICT AMD (Accumulation, Manipulation, Distribution) framework, "
                f"{label} is the {'first' if name == 'accumulation' else 'second' if name == 'manipulation' else 'third'} phase. "
                f"Definition: {defn}. "
                f"What it looks like on the chart: {appearance}. "
                f"What you should do: {action}. "
                f"The AMD cycle repeats every session — understanding where you are in the cycle "
                f"determines whether you should be waiting, preparing, or executing. "
                f"The key insight is that the Manipulation phase IS the entry signal — it sweeps "
                f"liquidity and creates the displacement that forms your entry zone."
            )
            self._add(
                f"Explain the {label} phase in ICT's AMD model.", resp, "core_concept"
            )
            count += 1

        # Entry models from ontology
        entries = onto.get("entry_models", {})
        for name, info in entries.items():
            if not isinstance(info, dict):
                continue
            label = self._label(name)
            seq = info.get("sequence", [])
            entry = info.get("entry", "")
            stop = info.get("stop", "")
            timing_val = info.get("timing", "")
            trigger = info.get("trigger", "")
            action = info.get("action", "")
            rule = info.get("rule", "")
            model_name = info.get("name", label)
            conditions = info.get("conditions", [])

            parts = [f"The {model_name} entry model in ICT methodology "]
            if seq:
                parts.append(f"follows this sequence: {' → '.join(seq)}. ")
            if conditions:
                parts.append(f"Requires: {', '.join(conditions)}. ")
            if entry:
                parts.append(f"Entry point: {entry}. ")
            if stop:
                parts.append(f"Stop loss: {stop}. ")
            if timing_val:
                parts.append(f"Timing: {timing_val}. ")
            if trigger:
                parts.append(f"Trigger: {trigger}. ")
            if action:
                parts.append(f"Action: {action}. ")
            if rule:
                parts.append(f"Rule: {rule}. ")
            parts.append(
                "Following the exact sequence is critical — skipping steps leads to poor entries and unnecessary losses."
            )

            self._add(
                f"How does the {model_name} entry model work in ICT?",
                "".join(parts),
                "core_concept",
            )
            count += 1

        # Trade management
        mgmt = onto.get("trade_management", {})
        if mgmt:
            # Stop management
            stop_mgmt = mgmt.get("stop_management", {})
            if stop_mgmt:
                parts = ["ICT stop loss management follows strict rules: "]
                for level, desc in stop_mgmt.items():
                    parts.append(f"{self._label(level)}: {desc}. ")
                parts.append(
                    "The golden rule in ICT is NEVER widen your stop — only tighten or leave it. Your stop represents the invalidation of your thesis."
                )
                self._add(
                    "How should I manage stop losses in ICT methodology?",
                    "".join(parts),
                    "core_concept",
                )
                count += 1

            # Partial strategies
            partials = mgmt.get("partial_strategies", {})
            if partials:
                parts = [
                    "ICT trade management includes several partial exit strategies: "
                ]
                for name, info in partials.items():
                    desc = info.get("description", "")
                    example = info.get("example", "")
                    use = info.get("use_case", "")
                    parts.append(f"{self._label(name)}: {desc}. ")
                    if example:
                        parts.append(f"Example: {example}. ")
                    if use:
                        parts.append(f"Best used: {use}. ")
                parts.append(
                    "Choose your strategy BEFORE entering — adjusting mid-trade leads to emotional decisions."
                )
                self._add(
                    "What partial exit strategies does ICT recommend?",
                    "".join(parts),
                    "core_concept",
                )
                count += 1

        # Validation rules
        validation = onto.get("validation", {})
        if validation:
            parts = [
                "ICT validation rules determine whether a setup or re-entry is still valid: "
            ]
            for name, info in validation.items():
                if isinstance(info, dict):
                    cond = info.get("condition", "")
                    meaning = info.get("meaning", "")
                    parts.append(f"{self._label(name)}: {cond}. Meaning: {meaning}. ")
            parts.append(
                "Always define your invalidation BEFORE entering. If invalidation occurs, exit immediately — no exceptions."
            )
            self._add(
                "What are the validation rules for ICT setups?",
                "".join(parts),
                "core_concept",
            )
            count += 1

        # Narrative reasoning
        narrative = onto.get("narrative_reasoning", {})
        if narrative:
            parts = [
                "ICT narrative reasoning requires answering three questions before every trade: "
            ]
            for name, info in narrative.items():
                if isinstance(info, dict):
                    desc = info.get("description", "")
                    examples = info.get("examples", [])
                    parts.append(f"\n\n{self._label(name)}: {desc}")
                    if examples:
                        parts.append(" Examples: " + "; ".join(examples[:3]) + ".")
            parts.append(
                "\n\nIf you cannot answer all three questions with specific, concrete reasons, you do not have a trade — you have a guess."
            )
            self._add(
                "What is ICT narrative reasoning and how do I apply it?",
                "".join(parts),
                "core_concept",
            )
            count += 1

        return count

    # ── 2. Concept Requirements ───────────────────────────────────────────

    def gen_concept_requirements(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        reqs = rels.get("concept_requirements", {})

        for concept, data in reqs.items():
            label = self._label(concept)

            # What does it require?
            requires = data.get("requires", [])
            if requires:
                parts = [
                    f"{label} has specific prerequisites in ICT methodology that must be confirmed before it's considered valid:\n\n"
                ]
                for req in requires:
                    c = req.get("concept", req) if isinstance(req, dict) else req
                    why = req.get("why", "") if isinstance(req, dict) else ""
                    parts.append(f"• Requires {self._label(c)}")
                    if why:
                        parts.append(f" — {why}")
                    parts.append("\n")
                parts.append(
                    f"\nWithout these prerequisites, {label} is unreliable. Many traders lose money by identifying {label} on the chart without confirming these requirements first."
                )

                self._add(
                    f"What does {label} require to be valid in ICT?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # What enhances it?
            enhanced = data.get("enhanced_by", [])
            if enhanced:
                parts = [
                    f"{label} becomes a higher-probability setup when combined with these confluence factors:\n\n"
                ]
                for enh in enhanced:
                    c = enh.get("concept", enh) if isinstance(enh, dict) else enh
                    bonus = enh.get("bonus", "") if isinstance(enh, dict) else ""
                    why = enh.get("why", "") if isinstance(enh, dict) else ""
                    parts.append(f"• {self._label(c)}")
                    if bonus:
                        parts.append(f" (+{bonus} confluence)")
                    if why:
                        parts.append(f" — {why}")
                    parts.append("\n")
                parts.append(
                    f"\nStacking these enhancers creates what ICT calls an 'A+ setup' — the highest probability trade with the best risk-to-reward ratio."
                )

                self._add(
                    f"What factors enhance {label} in ICT?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # What invalidates it?
            invalidated = data.get("invalidated_by", [])
            if invalidated:
                parts = [f"{label} is invalidated (no longer tradeable) when:\n\n"]
                for inv in invalidated:
                    cond = inv.get("condition", inv) if isinstance(inv, dict) else inv
                    why = inv.get("why", "") if isinstance(inv, dict) else ""
                    parts.append(f"• {cond}")
                    if why:
                        parts.append(f" — {why}")
                    parts.append("\n")
                parts.append(
                    f"\nRecognizing invalidation is critical. Trading an invalidated {label} is one of the most common mistakes and leads to consistent losses. When invalidated, move on to the next setup."
                )

                self._add(
                    f"When is {label} invalidated in ICT?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # Entry rules
            entry_rules = data.get("entry_rules", [])
            if entry_rules:
                parts = [f"The entry rules for {label} in ICT methodology are:\n\n"]
                for i, rule in enumerate(entry_rules, 1):
                    parts.append(f"{i}. {rule}\n")
                parts.append(
                    f"\nThese rules must be followed exactly. Deviating — entering early, entering late, or ignoring the SL placement — significantly reduces your edge."
                )
                self._add(
                    f"What are the entry rules for {label}?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # Targets
            targets = data.get("targets", [])
            if targets:
                parts = [f"When trading {label}, your target should be:\n\n"]
                for tgt in targets:
                    c = tgt.get("concept", tgt) if isinstance(tgt, dict) else tgt
                    why = tgt.get("why", "") if isinstance(tgt, dict) else ""
                    parts.append(f"• {self._label(c)}")
                    if why:
                        parts.append(f" — {why}")
                    parts.append("\n")
                parts.append(
                    f"\nIn ICT, entries use Internal Range Liquidity (IRL) like FVGs and OBs, while targets use External Range Liquidity (ERL) like equal highs/lows and old swing points."
                )
                self._add(
                    f"What should I target when trading {label}?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # Identification (for order_block)
            identification = data.get("identification", {})
            if identification:
                parts = [f"How to identify {label} on the chart:\n\n"]
                for direction, desc in identification.items():
                    parts.append(f"• {self._label(direction)}: {desc}\n")
                parts.append(
                    f"\nThe key is displacement — without a strong momentum candle leaving the zone, it's not a valid {label}. Just seeing a candle at a level is not enough."
                )
                self._add(
                    f"How do I identify {label} on a chart?",
                    "".join(parts),
                    "concept_requirement",
                )
                count += 1

            # Special: liquidity types
            types = data.get("types", {})
            if types and concept == "liquidity":
                for side, pools in types.items():
                    parts = [
                        f"{self._label(side)} liquidity in ICT includes these specific pool types:\n\n"
                    ]
                    for pool in pools:
                        parts.append(f"• {self._label(pool)}\n")
                    parts.append(
                        f"\nThese are NOT support/resistance levels — they are TARGETS. Price is drawn to liquidity. Smart money sweeps these pools to fuel their moves. The key insight: don't trade at these levels, trade AFTER they get swept."
                    )
                    self._add(
                        f"What types of {self._label(side)} liquidity exist in ICT?",
                        "".join(parts),
                        "concept_requirement",
                    )
                    count += 1

            # Special: market structure types
            ms_types = data.get("types", {})
            if ms_types and concept == "market_structure":
                for ms_name, ms_info in ms_types.items():
                    if isinstance(ms_info, dict):
                        meaning = ms_info.get("meaning", "")
                        action = ms_info.get("action", "")
                        ms_label = ms_name.upper()
                        resp = (
                            f"In ICT market structure analysis, {ms_label} ({meaning}) is a critical concept. "
                            f"When you identify {ms_label}: {action}. "
                            f"The sequence of market structure shifts matters: Liquidity sweep → CHoCH → MSS → Entry. "
                            f"CHoCH is the first warning that the trend may change, while MSS confirms the reversal. "
                            f"Never trade based on CHoCH alone — wait for MSS with displacement to confirm the shift. "
                            f"Many retail traders confuse BOS (continuation) with CHoCH (warning) — knowing the difference "
                            f"keeps you on the right side of institutional order flow."
                        )
                        self._add(
                            f"What is {ms_label} in ICT market structure?",
                            resp,
                            "concept_requirement",
                        )
                        count += 1

            # SMT divergence special handling
            if concept == "smt_divergence":
                defn = data.get("definition", "")
                pairs = data.get("pairs", [])
                signal = data.get("signal", "")
                weight = data.get("weight", "")
                ki = data.get("key_insight", "")

                pairs_text = (
                    "; ".join(f"{p[0]} vs {p[1]}" for p in pairs) if pairs else ""
                )
                resp = (
                    f"SMT (Smart Money Technique) Divergence in ICT methodology: {defn}. "
                    f"Correlated pairs to watch: {pairs_text}. "
                    f"How to read it: {signal}. "
                    f"Confluence weight: {weight}. "
                    f"Key insight: {ki}. "
                    f"SMT divergence is one of the most powerful confirmation tools in ICT because it reveals "
                    f"what institutional order flow is actually doing — when one pair makes a new extreme but its "
                    f"correlated partner doesn't follow, smart money is already positioned in the opposite direction."
                )
                self._add(
                    "Explain SMT Divergence in ICT methodology.",
                    resp,
                    "concept_requirement",
                )
                count += 1

        return count

    # ── 3. Model Deep Dives ───────────────────────────────────────────────

    def gen_model_deep_dives(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        models = rels.get("models", {})

        for model_name, data in models.items():
            label = self._label(model_name)
            desc = data.get("description", "")
            required = data.get("required", [])
            entry = data.get("entry", "")
            target = data.get("target", "")
            avoid = data.get("avoid_when", [])
            time_windows = data.get("time_windows", [])
            time_ctx = data.get("time_context", "")
            rarity = data.get("rarity", "")
            probability = data.get("probability", "")
            key = data.get("key", "")
            insight = data.get("insight", "")
            time_val = data.get("time", "")

            # 1. Complete model overview
            parts = [f"The {label} is an ICT trade model"]
            if desc:
                parts.append(f" — {desc.lower()}")
            parts.append(".\n\n")

            if required:
                parts.append("Requirements:\n")
                for r in required:
                    parts.append(f"• {r}\n")
                parts.append("\n")

            if time_windows:
                parts.append("Time windows:\n")
                for tw in time_windows:
                    parts.append(
                        f"• {tw.get('name', '')}: {tw.get('start', '')} - {tw.get('end', '')} {tw.get('timezone', '')}\n"
                    )
                parts.append("\n")
            elif time_ctx:
                parts.append(f"Timing: {time_ctx}\n\n")
            elif time_val:
                parts.append(f"Timing: {time_val}\n\n")

            if entry:
                parts.append(f"Entry: {entry}\n")
            if target:
                parts.append(f"Target: {target}\n")
            if key:
                parts.append(f"Key factor: {key}\n")
            if insight:
                parts.append(f"Critical insight: {insight}\n")
            if rarity:
                parts.append(f"Frequency: {rarity}\n")
            if probability:
                parts.append(f"Probability: {probability}\n")

            parts.append(
                f"\nAll requirements must be confirmed before entering. If any condition is missing, do not force the {label} — wait for the next one."
            )

            self._add(
                f"Explain the {label} model in ICT trading.",
                "".join(parts),
                "model_overview",
            )
            count += 1

            # 2. When to use
            if required:
                req_text = "; ".join(required)
                resp = (
                    f"You should use the {label} model when all of these conditions are present: {req_text}. "
                    f"{'Time context: ' + (time_ctx or time_val or '') + '. ' if time_ctx or time_val else ''}"
                    f"{'Entry: ' + entry + '. ' if entry else ''}"
                    f"{'Target: ' + target + '. ' if target else ''}"
                    f"The {label} is not a setup you force — it's a setup you recognize. "
                    f"If you find yourself trying to make the chart fit the model, walk away. "
                    f"Patience is the edge."
                )
                self._add(f"When should I use the {label} model?", resp, "model_usage")
                count += 1

            # 3. When to avoid
            if avoid:
                avoid_text = "\n".join(f"• {a}" for a in avoid)
                resp = (
                    f"Avoid the {label} model in these conditions:\n\n{avoid_text}\n\n"
                    f"These conditions significantly reduce the win rate of the {label}. "
                    f"Trading a setup in unfavorable conditions is one of the most common "
                    f"mistakes. The edge comes from SELECTIVITY — only taking the model "
                    f"when conditions are optimal."
                )
                self._add(
                    f"When should I avoid the {label} model?", resp, "model_avoidance"
                )
                count += 1

            # 4. Scenario: conditions met
            if len(required) >= 2:
                scenario_items = required[:3]
                scenario = ", ".join(scenario_items)
                resp = (
                    f"Based on these conditions — {scenario} — the {label} model is applicable. "
                    f"{'Description: ' + desc + '. ' if desc else ''}"
                    f"{'Your entry should be: ' + entry + '. ' if entry else ''}"
                    f"{'Your target: ' + target + '. ' if target else ''}"
                    f"Before executing, verify ALL remaining requirements are also met: "
                    f"{'; '.join(r for r in required if r not in scenario_items)}. "
                    f"Also check for red flags: no major news imminent, you're within the correct "
                    f"time window, and your R:R is at least 2:1."
                )
                self._add(
                    f"I see these conditions: {scenario}. Which ICT model fits?",
                    resp,
                    "model_selection",
                )
                count += 1

            # 5. Missing conditions scenario
            if len(required) >= 2:
                missing = required[-1]
                present = required[:-1]
                resp = (
                    f"You have: {', '.join(present)}. However, the {label} model also requires: {missing}. "
                    f"Without this condition, the setup is incomplete. "
                    f"Entering an incomplete {label} significantly reduces your win rate. "
                    f"The discipline to wait for ALL conditions is what separates consistently "
                    f"profitable ICT traders from those who give back their gains. "
                    f"Either wait for the missing condition to appear, or look for a different model "
                    f"that matches your current conditions."
                )
                self._add(
                    f"I have {', '.join(present)} but I'm missing {missing}. Should I trade the {label}?",
                    resp,
                    "model_incomplete",
                )
                count += 1

            # 6. Common mistakes with this model
            model_mistakes = {
                "silver_bullet": (
                    "Common mistakes with the Silver Bullet model:\n\n"
                    "1. Trading outside the time window — the Silver Bullet is TIME-SPECIFIC (10-11 AM or 2-3 PM ET). "
                    "An FVG at 9:30 AM is NOT a Silver Bullet, even if it looks identical.\n"
                    "2. Ignoring HTF bias — the FVG must align with the higher timeframe direction. "
                    "A bullish FVG in a bearish HTF context is a counter-trend trap.\n"
                    "3. Entering already-filled FVGs — if the FVG is 50%+ filled when you find it, "
                    "the setup is dead. You need a FRESH FVG within the window.\n"
                    "4. No prior liquidity sweep — the Silver Bullet requires liquidity to have been "
                    "swept before the FVG forms. Without the sweep, there's no institutional catalyst.\n\n"
                    "Fix: Use a checklist. Time window ✓, HTF bias ✓, liquidity swept ✓, fresh FVG ✓."
                ),
                "unicorn": (
                    "Common mistakes with the Unicorn model:\n\n"
                    "1. Forcing it — Unicorn setups (OB with FVG inside) are RARE, maybe 2-3 per week. "
                    "If you're seeing them daily, you're not identifying them correctly.\n"
                    "2. Wrong OB identification — the Order Block must form AFTER a liquidity sweep. "
                    "An OB without a prior sweep is just a candle.\n"
                    "3. Ignoring HTF alignment — even a perfect Unicorn fails if it's against the "
                    "higher timeframe bias. HTF direction is non-negotiable.\n"
                    "4. SL too tight — the stop must be beyond the entire OB, not just the FVG portion. "
                    "The Unicorn's strength comes from the full OB zone holding.\n\n"
                    "The Unicorn has the highest probability of all ICT setups precisely because it's rare "
                    "and requires multiple confluences to align."
                ),
                "ict_2022_model": (
                    "Common mistakes with the ICT 2022 Model (AMD cycle):\n\n"
                    "1. Entering during Accumulation — the accumulation phase is for WAITING, not trading. "
                    "Range-bound price action is where liquidity builds, not where you enter.\n"
                    "2. Getting trapped by Manipulation — the false breakout IS the manipulation. "
                    "Many traders buy the breakout and become the liquidity. You should be preparing "
                    "to trade the OPPOSITE direction after the sweep.\n"
                    "3. Chasing Distribution — if you missed the entry, don't chase. The distribution "
                    "phase moves fast and entering late gives terrible R:R.\n"
                    "4. Wrong session — the AMD cycle works best at session opens (London, NY) where "
                    "institutional order flow creates clean manipulation.\n\n"
                    "The edge: identify Accumulation, prepare for Manipulation, execute in Distribution."
                ),
                "turtle_soup": (
                    "Common mistakes with the Turtle Soup model:\n\n"
                    "1. Confusing a real breakout with a sweep — Turtle Soup requires an IMMEDIATE "
                    "reversal after the liquidity sweep. If price continues beyond the sweep for "
                    "multiple candles, it's a legitimate breakout, not a Turtle Soup.\n"
                    "2. No clear liquidity pool — you need well-defined equal highs/lows or a clean "
                    "swing point. Random price levels don't count.\n"
                    "3. SL too close — the stop must be beyond the sweep extreme. The sweep often "
                    "pushes 20-30 pips past the level before reversing.\n"
                    "4. Ignoring speed — the KEY to Turtle Soup is speed. The faster the reversal "
                    "after the sweep, the better. Slow, grinding reversals often fail.\n\n"
                    "Turtle Soup is a pure liquidity play — smart money sweeps stops, fills orders, "
                    "then drives price in the real direction."
                ),
                "judas_swing": (
                    "Common mistakes with the Judas Swing model:\n\n"
                    "1. Wrong timing — the Judas Swing happens in the FIRST 30-60 minutes of London "
                    "or NY open. A move 2 hours into the session is not a Judas Swing.\n"
                    "2. Trading the Judas itself — the Judas Swing is the FALSE move. You don't trade "
                    "WITH it, you trade AGAINST it after it sweeps liquidity.\n"
                    "3. No prior liquidity target — the Judas must sweep identifiable liquidity "
                    "(Asian range, overnight highs/lows). Random moves don't qualify.\n"
                    "4. Missing displacement — after the Judas sweep, you need displacement in the "
                    "real direction. Without it, the reversal isn't confirmed.\n\n"
                    "The Judas Swing IS the manipulation phase of AMD — it exists to trap counter-trend "
                    "traders and grab their liquidity for the real move."
                ),
            }
            if model_name in model_mistakes:
                self._add(
                    f"What are common mistakes when trading the {label}?",
                    model_mistakes[model_name],
                    "model_mistakes",
                )
                count += 1

            # 7. Step-by-step execution
            if required and entry:
                steps = [f"Step-by-step {label} execution:\n"]
                for i, req in enumerate(required, 1):
                    steps.append(f"\n{i}. Confirm: {req}")
                steps.append(f"\n{len(required) + 1}. Entry: {entry}")
                if target:
                    steps.append(f"\n{len(required) + 2}. Target: {target}")
                steps.append(
                    f"\n{len(required) + 3}. Risk management: Set SL at invalidation level. Never risk more than 1-2% per trade."
                )
                steps.append(
                    f"\n\nExecute these steps IN ORDER. Do not skip ahead. If any step fails, abandon the setup. Discipline is the edge."
                )
                self._add(
                    f"Walk me through executing a {label} trade step by step.",
                    "".join(steps),
                    "model_execution",
                )
                count += 1

        return count

    # ── 4. Causal Chain Reasoning ─────────────────────────────────────────

    def gen_causal_chains(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        chains = rels.get("causal_chains", {})

        for chain_name, chain_data in chains.items():
            label = self._label(chain_name)
            desc = chain_data.get("description", label)
            steps = chain_data.get("steps", {})
            failures = chain_data.get("failure_at_step", {})
            ki = chain_data.get("key_insight", "")

            sorted_steps = sorted(steps.items(), key=lambda x: int(x[0]))

            # 1. Full chain walkthrough
            parts = [f"The {desc} in ICT follows a strict sequential process:\n\n"]
            for num, step in sorted_steps:
                concept = step.get("concept", step.get("phase", f"Step {num}"))
                signal = step.get("signal", step.get("action", step.get("time", "")))
                parts.append(f"Step {num}: {self._label(concept)} — {signal}\n")

                # Add timeframes/concepts if present
                tfs = step.get("timeframes", [])
                if tfs:
                    parts.append(f"  Timeframes: {', '.join(tfs)}\n")
                concepts = step.get("concepts", [])
                if concepts:
                    parts.append(f"  Look for: {', '.join(concepts)}\n")
                patience = step.get("patience", "")
                if patience:
                    parts.append(f"  Note: {patience}\n")

            parts.append(
                f"\nEach step MUST complete before moving to the next. Skipping any step breaks the causal chain and significantly reduces the probability of a successful trade."
            )
            if ki:
                parts.append(f"\n\nKey insight: {ki}")

            self._add(
                f"Walk me through the {desc} step by step.",
                "".join(parts),
                "causal_chain",
            )
            count += 1

            # 2. Individual step failures
            for step_num, failure_msg in failures.items():
                step_data = steps.get(str(step_num), {})
                concept = step_data.get(
                    "concept", step_data.get("phase", f"step {step_num}")
                )

                resp = (
                    f"If step {step_num} ({self._label(concept)}) fails in the {desc}: {failure_msg}\n\n"
                    f"This breaks the causal chain at a critical point. Without this step completing, "
                    f"all subsequent steps lack the prerequisite confirmation they need. "
                    f"The correct action is to ABANDON this setup entirely and wait for a fresh sequence to begin. "
                    f"Do not try to 'make it work' by skipping to a later step — the ICT framework is sequential "
                    f"because each step provides the context and confirmation for the next."
                )
                self._add(
                    f"What happens if {self._label(concept)} fails in the {desc}?",
                    resp,
                    "causal_failure",
                )
                count += 1

            # 3. "I see X, what's next?" for each step
            for i, (num, step) in enumerate(sorted_steps):
                concept = step.get("concept", step.get("phase", f"Step {num}"))
                signal = step.get("signal", step.get("action", ""))

                if i < len(sorted_steps) - 1:
                    next_num, next_step = sorted_steps[i + 1]
                    next_concept = next_step.get("concept", next_step.get("phase", ""))
                    next_signal = next_step.get("signal", next_step.get("action", ""))

                    resp = (
                        f"You've confirmed {self._label(concept)} ({signal}). Good — this is step {num} of the {desc}.\n\n"
                        f"Next step (step {next_num}): Look for {self._label(next_concept)} — {next_signal}.\n\n"
                        f"Important: Do NOT enter yet. "
                    )
                    remaining = [s for _, s in sorted_steps[i + 1 :]]
                    if remaining:
                        rem_names = [
                            s.get("concept", s.get("phase", "")).replace("_", " ")
                            for s in remaining
                        ]
                        resp += f"You still need: {', '.join(rem_names)}. "
                    resp += "Patience at this stage is critical — premature entry is one of the top reasons ICT setups fail."

                    self._add(
                        f"I see {self._label(concept)} in the {desc}. What's next?",
                        resp,
                        "causal_next_step",
                    )
                    count += 1

            # 4. Complete chain vs incomplete
            if len(sorted_steps) >= 3:
                all_concepts = [
                    s.get("concept", s.get("phase", "")).replace("_", " ")
                    for _, s in sorted_steps
                ]
                missing_idx = self.rng.randint(0, len(all_concepts) - 1)
                present = [c for i, c in enumerate(all_concepts) if i != missing_idx]
                missing = all_concepts[missing_idx]

                resp = (
                    f"You have: {', '.join(present)}. But you're missing: {missing}.\n\n"
                    f"In the {desc}, ALL steps must be present in sequence. {self._label(missing)} "
                    f"is {'a critical early step that provides the foundation' if missing_idx < 2 else 'a later confirmation step'} "
                    f"for the entire chain.\n\n"
                    f"Without {missing}, this is NOT a valid setup. Do not enter. Wait for the complete sequence, "
                    f"or look for a different setup that matches your current market conditions."
                )
                self._add(
                    f"I have {', '.join(present)} but no {missing}. Can I still trade?",
                    resp,
                    "causal_incomplete",
                )
                count += 1

        return count

    # ── 5. Anti-Pattern Education ─────────────────────────────────────────

    def gen_anti_patterns(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        onto = self._load_ontology()

        # From concept_relationships.yaml
        rels_patterns = rels.get("anti_patterns", {})
        for name, data in rels_patterns.items():
            label = self._label(name)
            desc = data.get("description", "")
            why_fails = data.get("why_fails", "")
            fix = data.get("fix", "")
            symptom = data.get("symptom", "")
            winrate = data.get("historical_winrate", None)

            # Main explanation
            parts = [f"**{label}** is a dangerous anti-pattern in ICT trading.\n\n"]
            parts.append(f"What it looks like: {desc}\n\n")
            parts.append(f"Why it fails: {why_fails}\n\n")
            if symptom:
                parts.append(f"Telltale symptom: {symptom}\n\n")
            if winrate is not None:
                parts.append(
                    f"Historical win rate: {int(winrate * 100)}% — well below profitable threshold.\n\n"
                )
            parts.append(f"How to fix it: {fix}\n\n")
            parts.append(
                "Recognizing anti-patterns is just as important as recognizing valid setups. "
                "The ability to say 'this is NOT a trade' protects your capital on days when "
                "the market is designed to take it from you."
            )

            self._add(
                f"What is the '{label}' anti-pattern in ICT?",
                "".join(parts),
                "anti_pattern",
            )
            count += 1

            # "I just did this, what went wrong?"
            resp = (
                f"You've fallen into the '{label}' anti-pattern. Here's what happened:\n\n"
                f"{desc}\n\n"
                f"This fails because: {why_fails}\n\n"
                f"{'You probably noticed: ' + symptom + chr(10) + chr(10) if symptom else ''}"
                f"Going forward: {fix}\n\n"
                f"This is a learning moment, not a failure. Every profitable ICT trader has made this mistake. "
                f"The difference is they built a rule to prevent repeating it."
            )
            self._add(
                f"I {desc.lower()[:-1] if desc.endswith('.') else desc.lower()}. What went wrong?",
                resp,
                "anti_pattern_scenario",
            )
            count += 1

        # From ontology anti-patterns
        onto_patterns = onto.get("anti_patterns", {})
        for name, data in onto_patterns.items():
            if not isinstance(data, dict):
                continue
            label = self._label(name)
            looks_like = data.get("looks_like", "")
            problem = data.get("problem", "")
            lesson = data.get("lesson", "")

            if not looks_like:
                continue

            resp = (
                f"The {label} anti-pattern in ICT trading:\n\n"
                f"What it looks like: {looks_like}\n"
                f"The problem: {problem}\n"
                f"The lesson: {lesson}\n\n"
                f"Every experienced ICT trader has a 'never again' list. {label} should be on yours. "
                f"When you catch yourself in this pattern, close the charts, take a break, and come back "
                f"with fresh eyes. The next A+ setup is always coming — you just need to be patient enough to wait for it."
            )
            self._add(f"Explain the {label} anti-pattern in ICT.", resp, "anti_pattern")
            count += 1

        return count

    # ── 6. Trade Scenario Analysis ────────────────────────────────────────

    def gen_trade_scenarios(self) -> int:
        count = 0
        trades = self._load_trades(POSITIVE_DIR)

        for trade in trades:
            trade_id = trade.get("id", "unknown")
            pair = trade.get("market", {}).get("pair", "Unknown")
            session = trade.get("time", {}).get("session", "Unknown")
            killzone = trade.get("time", {}).get("killzone", "Unknown")
            htf_bias = trade.get("context", {}).get("htf_bias", "Unknown")
            env_notes = trade.get("context", {}).get("environment_notes", "")
            mm_model = trade.get("mm_model", {})
            setup = trade.get("setup", {})
            execution = trade.get("execution", {})
            management = trade.get("management", {})
            reasoning = trade.get("reasoning", {})
            pd_arrays = trade.get("pd_arrays", {})
            context = trade.get("context", {})
            labels = trade.get("labels", {})
            time_info = trade.get("time", {})

            # 1. Full trade analysis
            parts = [f"Trade Analysis: {pair} — {session} session\n\n"]
            parts.append(f"Context: {env_notes}\n\n")

            if mm_model:
                mm_type = mm_model.get("type", "")
                mm_phase = mm_model.get("phase", "")
                target_liq = mm_model.get("target_liquidity", "")
                parts.append(f"Market Maker Model: {mm_type} in {mm_phase} phase\n")
                if target_liq:
                    parts.append(f"Target liquidity: {target_liq}\n")

            parts.append(f"HTF Bias: {htf_bias}\n")
            if setup:
                bias = setup.get("bias", "")
                setup_type = setup.get("setup_type", "")
                parts.append(f"Trade direction: {bias}, Setup type: {setup_type}\n")

            if execution:
                entry = execution.get("entry_price", "")
                sl = execution.get("stop_loss", "")
                risk_pips = execution.get("risk_pips", "")
                parts.append(f"Entry: {entry}, SL: {sl}, Risk: {risk_pips} pips\n")

            if management and management.get("final_outcome"):
                outcome = management["final_outcome"]
                parts.append(
                    f"\nResult: {outcome.get('result', '').upper()} — {outcome.get('pnl_pips', '')} pips, {outcome.get('r_achieved', '')}R\n"
                )
                parts.append(f"Exit reason: {outcome.get('exit_reason', '')}\n")

            quality = labels.get("quality_tag", "")
            if quality:
                parts.append(f"\nSetup quality: {quality}\n")

            parts.append(
                "\nThis trade demonstrates proper ICT methodology: HTF bias aligned, liquidity swept, confluence stacked, and risk managed."
            )

            self._add(
                f"Analyze this {pair} trade during the {session} session.",
                "".join(parts),
                "trade_analysis",
            )
            count += 1

            # 2. Why here? Why now?
            if reasoning:
                why_here = reasoning.get("why_here", "")
                why_now = reasoning.get("why_now", "")
                invalidates = reasoning.get("what_invalidates", "")
                notes = reasoning.get("notes", "")

                if why_here or why_now:
                    resp = (
                        f"Narrative reasoning for this {pair} trade:\n\n"
                        f"WHY HERE (price level significance): {why_here}\n\n"
                        f"WHY NOW (timing justification): {why_now}\n\n"
                        f"WHAT WOULD KILL IT (invalidation): {invalidates}\n\n"
                        f"{'Notes: ' + notes + chr(10) + chr(10) if notes else ''}"
                        f"This is the ICT narrative framework in action — every trade must answer these three "
                        f"questions with specific, concrete reasons. If you can't articulate WHY you're entering "
                        f"at THIS price at THIS time, you don't have a trade."
                    )
                    self._add(
                        f"Explain the reasoning behind the {pair} {session} trade.",
                        resp,
                        "trade_reasoning",
                    )
                    count += 1

            # 3. PD Array analysis
            if pd_arrays:
                htf_obs = pd_arrays.get("htf_order_blocks", [])
                ltf_obs = pd_arrays.get("ltf_order_blocks", [])
                fvgs = pd_arrays.get("fair_value_gaps", [])

                elements = []
                if htf_obs:
                    for ob in htf_obs:
                        elements.append(
                            f"HTF {ob.get('direction', '')} OB on {ob.get('tf', '')} at {ob.get('price_zone', '')}"
                        )
                if ltf_obs:
                    for ob in ltf_obs:
                        elements.append(
                            f"LTF {ob.get('direction', '')} OB on {ob.get('tf', '')} at {ob.get('price_zone', '')}"
                        )
                if fvgs:
                    for fvg in fvgs:
                        elements.append(
                            f"{fvg.get('tf', '')} FVG ({fvg.get('direction', '')}) at {fvg.get('price_zone', '')} — {fvg.get('origin_leg_strength', '')} origin"
                        )

                if elements:
                    resp = (
                        f"PD Array analysis for {pair} trade:\n\n"
                        + "\n".join(f"• {e}" for e in elements)
                        + f"\n\nThese PD Arrays provided the entry zones for this trade. "
                        f"When multiple PD Arrays overlap at the same price level (multi-timeframe confluence), "
                        f"it creates a high-probability entry zone. This trade had "
                        f"{len(elements)} PD Arrays identified, "
                        f"{'with clear multi-TF confluence' if len(elements) >= 3 else 'providing a solid entry zone'}."
                    )
                    self._add(
                        f"What PD Arrays were present in this {pair} trade?",
                        resp,
                        "trade_pd_arrays",
                    )
                    count += 1

            # 4. Liquidity analysis
            liq_map = context.get("liquidity_map", {})
            if liq_map:
                bsl = liq_map.get("buy_side_levels", [])
                ssl = liq_map.get("sell_side_levels", [])
                range_info = context.get("range", {})

                parts = [f"Liquidity map for {pair} trade:\n\n"]
                if bsl:
                    parts.append("Buy-side liquidity (above price):\n")
                    for level in bsl:
                        parts.append(
                            f"  • {level.get('price', '')} — {level.get('type', '').replace('_', ' ')}\n"
                        )
                if ssl:
                    parts.append("Sell-side liquidity (below price):\n")
                    for level in ssl:
                        parts.append(
                            f"  • {level.get('price', '')} — {level.get('type', '').replace('_', ' ')}\n"
                        )
                if range_info:
                    ah = range_info.get("asia_high", "")
                    al = range_info.get("asia_low", "")
                    if ah and al:
                        parts.append(f"\nAsia range: {al} — {ah}\n")
                    terminus = range_info.get("terminus_level", "")
                    if terminus:
                        parts.append(f"Terminus level: {terminus}\n")

                parts.append(
                    "\nIn ICT, liquidity levels are TARGETS, not support/resistance. Price is drawn to "
                    "these pools. The trade plan uses IRL (FVGs, OBs) for entry and ERL (liquidity pools) for targets."
                )
                self._add(
                    f"Map the liquidity for this {pair} trade.",
                    "".join(parts),
                    "trade_liquidity",
                )
                count += 1

            # 5. Model identification
            if mm_model and mm_model.get("type"):
                mm_type = mm_model.get("type", "")
                mm_phase = mm_model.get("phase", "")
                dealing_range = mm_model.get("dealing_range", {})
                entry_rel = mm_model.get("entry_relative_to_eq", "")

                parts = [
                    f"This {pair} trade used the {mm_type} (Market Maker {'Sell' if 'S' in mm_type else 'Buy'} Model).\n\n"
                ]
                if mm_phase:
                    parts.append(f"Phase: {mm_phase}\n")
                if dealing_range:
                    parts.append(
                        f"Dealing range: {dealing_range.get('low', '')} — {dealing_range.get('high', '')}\n"
                    )
                    eq = dealing_range.get("equilibrium", "")
                    if eq:
                        parts.append(f"Equilibrium: {eq}\n")
                if entry_rel:
                    parts.append(f"Entry relative to equilibrium: {entry_rel}\n")

                parts.append(
                    f"\nThe Market Maker Model describes how institutional order flow creates "
                    f"predictable price patterns. {mm_type} means smart money is {'distributing (selling)' if 'S' in mm_type else 'accumulating (buying)'}. "
                    f"Understanding which MM model is active tells you the probable direction of the next move."
                )
                self._add(
                    f"What market maker model was used in this {pair} trade?",
                    "".join(parts),
                    "trade_model",
                )
                count += 1

            # 6. Confirmation checklist
            if setup and setup.get("confirmation"):
                conf = setup["confirmation"]
                bos = conf.get("bos_or_choch", "")
                disp = conf.get("displacement", False)
                entry_model = conf.get("entry_model", "")
                extra = conf.get("extra_filters", [])

                parts = [f"Confirmation checklist for {pair} trade:\n\n"]
                if bos:
                    parts.append(f"✓ Structure: {bos}\n")
                parts.append(
                    f"{'✓' if disp else '✗'} Displacement: {'confirmed' if disp else 'NOT confirmed'}\n"
                )
                if entry_model:
                    parts.append(f"✓ Entry model: {entry_model.replace('_', ' ')}\n")
                for f in extra:
                    parts.append(f"✓ {f.replace('_', ' ')}\n")

                parts.append(
                    f"\nEvery checkmark represents a confluence factor that increases the probability of success. "
                    f"This trade had {2 + len(extra)} confirmations — a strong setup. "
                    f"Minimum for entry should be 3-4 confirmations."
                )
                self._add(
                    f"What confirmations were present in this {pair} trade?",
                    "".join(parts),
                    "trade_confirmation",
                )
                count += 1

            # 7. Execution review
            if execution and management:
                entry_price = execution.get("entry_price", "")
                sl = execution.get("stop_loss", "")
                risk_pips = execution.get("risk_pips", "")
                mae = execution.get("max_adverse_excursion_pips", "")
                mfe = execution.get("max_favorable_excursion_pips", "")
                targets = execution.get("targets", [])
                outcome = management.get("final_outcome", {})

                parts = [f"Execution review for {pair} trade:\n\n"]
                parts.append(f"Entry: {entry_price}\n")
                parts.append(f"Stop loss: {sl} ({risk_pips} pips risk)\n")
                if targets:
                    for tgt in targets:
                        parts.append(
                            f"Target: {tgt.get('label', '').replace('_', ' ')} at {tgt.get('price', '')} ({tgt.get('expected_rr', '')}R)\n"
                        )
                if mae:
                    parts.append(
                        f"Max adverse excursion: {mae} pips (how far it went against)\n"
                    )
                if mfe:
                    parts.append(
                        f"Max favorable excursion: {mfe} pips (maximum gain available)\n"
                    )

                if outcome:
                    pnl = outcome.get("pnl_pips", "")
                    rr = outcome.get("r_achieved", "")
                    parts.append(f"\nResult: {pnl} pips, {rr}R achieved\n")

                parts.append(
                    "\nProper execution means entry at the PD Array, stop at invalidation, and target at liquidity. "
                    "The R:R ratio must be at least 2:1 before entering."
                )
                self._add(
                    f"Review the execution of this {pair} trade.",
                    "".join(parts),
                    "trade_execution",
                )
                count += 1

        return count

    # ── 7. Negative Trade Analysis ────────────────────────────────────────

    def gen_negative_trades(self) -> int:
        count = 0
        trades = self._load_trades(NEGATIVE_DIR)
        rels = self._load_concept_rels()
        anti_patterns = rels.get("anti_patterns", {})

        for trade in trades:
            trade_id = trade.get("id", "unknown")
            pair = trade.get("market", {}).get("pair", "Unknown")
            env_notes = trade.get("context", {}).get("environment_notes", "")
            tags = trade.get("meta", {}).get("tags", [])
            failure = trade.get("failure_analysis", {}) or {}
            reasoning = trade.get("reasoning", {}) or {}
            execution = trade.get("execution", {}) or {}
            management = trade.get("management", {}) or {}
            outcome = management.get("final_outcome", {}) or {}

            # 1. Full failure analysis
            parts = [f"Failure analysis: {pair} trade ({trade_id})\n\n"]
            if env_notes:
                parts.append(f"Context: {env_notes}\n\n")

            if outcome:
                parts.append(
                    f"Result: {outcome.get('result', 'loss').upper()} — {outcome.get('pnl_pips', '')} pips"
                )
                if outcome.get("pnl_usd"):
                    parts.append(f" (${outcome['pnl_usd']})")
                parts.append(
                    f"\nExit reason: {outcome.get('exit_reason', 'unknown')}\n\n"
                )

            # Match tags to anti-patterns
            matched_patterns = []
            for tag in tags:
                tag_lower = tag.lower().replace(" ", "_")
                for ap_name, ap_data in anti_patterns.items():
                    if tag_lower in ap_name or ap_name in tag_lower:
                        matched_patterns.append((ap_name, ap_data))

            if matched_patterns:
                parts.append("Anti-patterns identified:\n")
                for ap_name, ap_data in matched_patterns:
                    parts.append(
                        f"• {self._label(ap_name)}: {ap_data.get('description', '')}\n"
                    )
                    parts.append(f"  Why it fails: {ap_data.get('why_fails', '')}\n")
                    parts.append(f"  Fix: {ap_data.get('fix', '')}\n")
            elif tags:
                parts.append(f"Issues tagged: {', '.join(tags)}\n")

            if failure:
                root = failure.get("root_cause", "")
                if root:
                    parts.append(f"\nRoot cause: {root}\n")

            # Reasoning if available
            if reasoning:
                invalidates = reasoning.get("what_invalidates", "")
                if invalidates:
                    parts.append(
                        f"\nWhat should have been the invalidation: {invalidates}\n"
                    )

            parts.append(
                "\nEvery losing trade is a tuition payment. The lesson here: follow the checklist, "
                "respect your invalidation levels, and never force a setup that doesn't meet all criteria."
            )

            self._add(
                f"Analyze why this {pair} trade failed.",
                "".join(parts),
                "trade_failure",
            )
            count += 1

            # 2. What should I have done differently?
            if matched_patterns:
                fixes = [
                    ap_data.get("fix", "")
                    for _, ap_data in matched_patterns
                    if ap_data.get("fix")
                ]
                if fixes:
                    resp = (
                        f"In this {pair} trade, the key mistakes and corrections:\n\n"
                        + "\n".join(f"• Instead: {fix}" for fix in fixes)
                        + f"\n\nThe common theme: patience and discipline. The market rewards traders who wait "
                        f"for their setup, not traders who force entries. Had these corrections been applied, "
                        f"either the trade would have been profitable or — equally importantly — "
                        f"you would have stayed on the sidelines and preserved capital."
                    )
                    self._add(
                        f"What should I have done differently in this {pair} trade?",
                        resp,
                        "trade_correction",
                    )
                    count += 1

            # 3. Tags-based questions
            for tag in tags[:3]:
                tag_label = tag.replace("_", " ").title()
                resp = (
                    f"This trade was tagged with '{tag_label}', which indicates a specific type of trading error.\n\n"
                    f"In ICT methodology, {tag_label.lower()} is considered a rule violation that undermines "
                    f"the statistical edge of the system. Even when the analysis is correct, poor execution "
                    f"or discipline failures turn winning trades into losers.\n\n"
                    f"The fix: add '{tag_label}' to your pre-trade checklist as a red flag. "
                    f"If you catch yourself about to make this mistake, close the chart and take a 30-minute break. "
                    f"Protecting capital is more important than any single trade."
                )
                self._add(
                    f"My {pair} trade was flagged for '{tag_label}'. What does this mean?",
                    resp,
                    "trade_tag_analysis",
                )
                count += 1

        return count

    # ── 8. Confluence Scoring ─────────────────────────────────────────────

    def gen_confluence_scoring(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        weights = rels.get("confluence_weights", {})

        if not weights:
            return 0

        # Build factor list
        all_factors = {}
        for tier, factors in weights.items():
            if tier == "thresholds":
                continue
            if isinstance(factors, dict):
                for factor, weight in factors.items():
                    all_factors[factor] = (weight, tier)
            elif isinstance(factors, list):
                for item in factors:
                    if isinstance(item, dict):
                        for factor, weight in item.items():
                            all_factors[factor] = (weight, tier)

        thresholds = weights.get("thresholds", {})
        min_score = thresholds.get("minimum_for_trade", 5.0)
        good_score = thresholds.get("good_setup", 7.0)
        aplus_score = thresholds.get("a_plus_setup", 9.0)

        # 1. Full scoring explanation
        parts = [
            "ICT confluence scoring uses weighted factors to objectively evaluate trade setups:\n\n"
        ]

        # Group by positive/negative
        positive = {k: v for k, v in all_factors.items() if v[0] > 0}
        negative = {k: v for k, v in all_factors.items() if v[0] < 0}

        parts.append("POSITIVE FACTORS:\n")
        for factor, (weight, tier) in sorted(positive.items(), key=lambda x: -x[1][0]):
            parts.append(f"  +{weight} — {self._label(factor)} ({tier})\n")

        parts.append("\nNEGATIVE FACTORS (penalties):\n")
        for factor, (weight, tier) in sorted(negative.items(), key=lambda x: x[1][0]):
            parts.append(f"  {weight} — {self._label(factor)} ({tier})\n")

        parts.append(f"\nThresholds:\n")
        parts.append(f"  {min_score}+ = Minimum for any trade\n")
        parts.append(f"  {good_score}+ = Good setup\n")
        parts.append(f"  {aplus_score}+ = A+ setup (highest conviction)\n")
        parts.append(
            f"\nNever trade below {min_score}. Be selective — quality over quantity."
        )

        self._add(
            "How does ICT confluence scoring work?",
            "".join(parts),
            "confluence_scoring",
        )
        count += 1

        # 2. Scenario scoring examples
        positive_factors = list(positive.keys())
        negative_factors = list(negative.keys())

        # A+ scenario
        if len(positive_factors) >= 5:
            chosen = self.rng.sample(positive_factors, min(6, len(positive_factors)))
            score = sum(all_factors[f][0] for f in chosen)
            parts = [f"Scoring this setup:\n\n"]
            for f in chosen:
                w = all_factors[f][0]
                parts.append(f"  +{w} — {self._label(f)}\n")
            parts.append(f"\nTotal score: {score}\n")
            rating = (
                "A+ setup"
                if score >= aplus_score
                else "Good setup"
                if score >= good_score
                else "Minimum met"
                if score >= min_score
                else "DO NOT TRADE"
            )
            parts.append(f"Rating: {rating}\n\n")
            parts.append(
                f"This is {'a high-conviction trade — full size' if score >= aplus_score else 'tradeable but consider reduced size' if score >= good_score else 'borderline — only with perfect execution'}."
            )

            self._add(
                f"Score this setup: I have {', '.join(self._label(f) for f in chosen[:3])} and more.",
                "".join(parts),
                "confluence_scenario",
            )
            count += 1

        # Weak scenario with penalties
        if positive_factors and negative_factors:
            pos_chosen = self.rng.sample(
                positive_factors, min(3, len(positive_factors))
            )
            neg_chosen = self.rng.sample(
                negative_factors, min(2, len(negative_factors))
            )
            score = sum(all_factors[f][0] for f in pos_chosen + neg_chosen)

            parts = [f"Scoring this setup:\n\n"]
            for f in pos_chosen:
                w = all_factors[f][0]
                parts.append(f"  +{w} — {self._label(f)}\n")
            for f in neg_chosen:
                w = all_factors[f][0]
                parts.append(f"  {w} — {self._label(f)} ⚠️\n")
            parts.append(f"\nTotal score: {score}\n")
            if score < min_score:
                parts.append(f"Rating: BELOW MINIMUM ({min_score}) — DO NOT TRADE\n\n")
                parts.append(
                    "The penalties from the negative factors drag this below the trading threshold. Even though some positive factors are present, the risks outweigh the edge."
                )
            else:
                parts.append(
                    f"Rating: {'Borderline' if score < good_score else 'Acceptable'} — {'proceed with caution' if score < good_score else 'tradeable'}\n\n"
                )
                parts.append(
                    "Be aware of the penalty factors — they reduce your expected win rate."
                )

            self._add(
                f"I have {', '.join(self._label(f) for f in pos_chosen)} but also {', '.join(self._label(f) for f in neg_chosen)}. Should I trade?",
                "".join(parts),
                "confluence_scenario",
            )
            count += 1

        # 3. Individual factor explanations
        important_factors = [
            (
                "liquidity_swept",
                "Liquidity swept is the highest-weighted positive factor (+2.5) in ICT confluence scoring. A liquidity sweep means smart money has collected the orders it needs (stop losses, breakout orders) to fuel its move. Without a sweep, there's no institutional catalyst — the setup lacks the 'fuel' for a sustained directional move. Always wait for the sweep.",
            ),
            (
                "htf_bias_aligned",
                "HTF bias alignment carries +2.0 weight. When your trade aligns with the Daily/4H trend direction, you're trading WITH institutional order flow. This is non-negotiable in ICT — taking LTF signals against HTF bias is 'swimming upstream.' Even a perfect setup against the HTF fails more often than not.",
            ),
            (
                "displacement_confirmed",
                "Displacement confirmation (+2.0) means you've seen a large-bodied candle showing institutional conviction. In ICT, displacement creates the FVGs and OBs that become your entry zones. Without displacement, the 'setup' you see is just random noise — smart money hasn't committed yet.",
            ),
            (
                "no_liquidity_sweep",
                "No liquidity sweep is a -2.0 penalty because it means the market hasn't collected the fuel it needs. Entering before a sweep often means YOU become the liquidity. The most common stop-out scenario: entering at an OB, getting swept, then watching price reverse in your direction without you. Always wait for the sweep.",
            ),
            (
                "against_htf_bias",
                "Trading against HTF bias is a -2.0 penalty. This is one of the strongest warnings in the confluence system. Even a setup with 3-4 positive factors will be penalized below the trading threshold if it's against the higher timeframe direction. The HTF determines where price is GOING — the LTF determines where to GET IN.",
            ),
        ]

        for factor, explanation in important_factors:
            if factor in all_factors:
                w = all_factors[factor][0]
                self._add(
                    f"Why is '{self._label(factor)}' weighted {'+' if w > 0 else ''}{w} in confluence scoring?",
                    explanation,
                    "confluence_factor",
                )
                count += 1

        return count

    # ── 9. Time & Session Rules ───────────────────────────────────────────

    def gen_time_sessions(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        onto = self._load_ontology()

        time_rules = rels.get("time_rules", {})
        session_behaviors = onto.get("session_behaviors", {})

        # Killzone deep dives
        killzones = time_rules.get("killzones", {})
        for kz_name, kz_data in killzones.items():
            label = self._label(kz_name)
            time_val = kz_data.get("time", "")
            behavior = kz_data.get("behavior", "")
            style = kz_data.get("trade_style", "")
            best = kz_data.get("best_setups", [])
            liq = kz_data.get("liquidity_builds", [])
            contains = kz_data.get("contains", [])

            parts = [f"The {label} killzone in ICT methodology:\n\n"]
            parts.append(f"Time window: {time_val}\n")
            parts.append(f"Market behavior: {behavior}\n")
            parts.append(f"How to trade it: {style}\n")
            if best:
                parts.append(f"Best setups: {', '.join(best)}\n")
            if liq:
                parts.append(f"Liquidity that builds: {', '.join(liq)}\n")
            if contains:
                parts.append(f"Contains: {', '.join(contains)}\n")

            # Add session behavior if available
            sb = session_behaviors.get(kz_name, {})
            if sb:
                typical = sb.get("typical_action", "")
                chars = sb.get("characteristics", [])
                trap = sb.get("trap_type", "")
                if typical:
                    parts.append(f"\nTypical action: {typical}\n")
                if chars:
                    parts.append(f"Characteristics: {', '.join(chars)}\n")
                if trap:
                    parts.append(f"Common trap: {trap}\n")

            parts.append(
                f"\nKillzones are the WHEN of ICT trading. Even a perfect setup taken outside the killzone has significantly lower probability."
            )

            self._add(
                f"Tell me about the {label} killzone in ICT.",
                "".join(parts),
                "time_session",
            )
            count += 1

        # Avoid times
        avoid = time_rules.get("avoid_times", [])
        if avoid:
            parts = ["Times to AVOID trading in ICT methodology:\n\n"]
            for item in avoid:
                time_val = item.get("time", "")
                reason = item.get("reason", "")
                parts.append(f"• {time_val}: {reason}\n")
            parts.append(
                "\nTrading during these times is a common mistake. The market is designed to take money "
                "from traders who are active when they shouldn't be. The killzones exist because institutional "
                "order flow is concentrated in specific windows. Outside those windows, price action is "
                "noise — random chop that stops out directional traders."
            )
            self._add(
                "When should I avoid trading in ICT?", "".join(parts), "time_avoid"
            )
            count += 1

        # Macro times
        macro = time_rules.get("macro_times", {})
        if macro:
            desc = macro.get("description", "")
            times = macro.get("times", [])
            usage = macro.get("usage", "")

            parts = [f"ICT Macro Times — {desc}:\n\n"]
            for t in times:
                parts.append(f"• {t}\n")
            parts.append(f"\nUsage: {usage}\n")
            parts.append(
                "\nMacro times are micro session opens where institutional algorithms become active. "
                "These are precision timing tools — if you're in a killzone with a valid setup, "
                "a macro time gives you the exact moment to look for your entry trigger (displacement, FVG formation)."
            )
            self._add(
                "What are ICT macro times and how do I use them?",
                "".join(parts),
                "time_macro",
            )
            count += 1

        # Session flow patterns
        day_flow = session_behaviors.get("day_flow_pattern", {})
        if day_flow:
            parts = ["ICT daily session flow patterns:\n\n"]
            for pattern, desc in day_flow.items():
                parts.append(f"• {self._label(pattern)}: {desc}\n")
            parts.append(
                "\nRecognizing which flow pattern is playing out helps you anticipate "
                "what's likely to happen next. On a normal day, Asia accumulates, London manipulates, "
                "and NY distributes. On a reversal day, London overextends and NY snaps back. "
                "Identifying the pattern early gives you an edge in positioning."
            )
            self._add(
                "What are the daily session flow patterns in ICT?",
                "".join(parts),
                "time_session_flow",
            )
            count += 1

        # Individual session behaviors
        for sess_name, sess_data in session_behaviors.items():
            if sess_name == "day_flow_pattern" or not isinstance(sess_data, dict):
                continue
            label = self._label(sess_name)
            typical = sess_data.get("typical_action", "")
            chars = sess_data.get("characteristics", [])
            trap = sess_data.get("trap_type", "")
            bias = sess_data.get("trade_bias", "")

            resp = (
                f"The {label} session in ICT trading:\n\n"
                f"Typical action: {typical}\n"
                f"Key characteristics: {', '.join(chars) if chars else 'N/A'}\n"
                f"Common trap to avoid: {trap}\n"
                f"Trading bias: {bias}\n\n"
                f"Each session has a role in the daily price delivery cycle. Understanding {label}'s "
                f"role helps you know whether to be aggressive, patient, or flat during this window."
            )
            self._add(
                f"How does the {label} session behave in ICT?", resp, "time_session"
            )
            count += 1

        # Optimal trade days
        optimal_days = onto.get("timing", {}).get("optimal_trade_days", {})
        if optimal_days:
            best = optimal_days.get("best", [])
            good = optimal_days.get("good", [])
            avoid_days = optimal_days.get("avoid", [])
            resp = (
                f"ICT optimal trading days:\n\n"
                f"Best days: {', '.join(best)} — highest probability setups, cleanest price action\n"
                f"Good days: {', '.join(good)} — tradeable but slightly less reliable\n"
                f"Avoid: {', '.join(avoid_days)} — position squaring, unpredictable moves\n\n"
                f"Tuesday and Wednesday are the best because institutional algorithms have the full "
                f"weekly range to deliver. Monday can be slow (establishing direction), and Friday "
                f"sees profit-taking and position squaring that distorts price action. "
                f"Trading only on optimal days is a simple filter that improves your win rate significantly."
            )
            self._add(
                "Which days of the week are best for ICT trading?", resp, "time_days"
            )
            count += 1

        return count

    # ── 10. Pair-Specific Knowledge ───────────────────────────────────────

    def gen_pair_knowledge(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        pair_rules = rels.get("pair_rules", {})

        for pair, data in pair_rules.items():
            pair_label = pair.replace("_", "/")
            correlations = data.get("correlations", [])
            sessions = data.get("best_sessions", [])
            chars = data.get("characteristics", "")
            smt = data.get("smt_partner", "")
            daily_range = data.get("typical_daily_range", "")
            warning = data.get("warning", "")
            pip_size = data.get("pip_size", "")

            parts = [f"{pair_label} in ICT trading methodology:\n\n"]
            if chars:
                parts.append(f"Characteristics: {chars}\n")
            if daily_range:
                parts.append(f"Typical daily range: {daily_range}\n")
            if sessions:
                parts.append(f"Best sessions: {', '.join(sessions)}\n")
            if correlations:
                parts.append(
                    f"Correlations: {', '.join(c.replace('_', ' ') for c in correlations)}\n"
                )
            if smt:
                parts.append(f"SMT divergence partner: {smt.replace('_', '/')}\n")
            if pip_size:
                parts.append(f"Pip size: {pip_size}\n")
            if warning:
                parts.append(f"⚠️ Warning: {warning}\n")

            parts.append(
                f"\nKnowing your pair's personality is crucial. {pair_label} has specific "
                f"behavioral patterns that differ from other pairs. Trading it during its best "
                f"sessions and understanding its correlations gives you an informational edge."
            )

            self._add(
                f"How should I trade {pair_label} using ICT methodology?",
                "".join(parts),
                "pair_knowledge",
            )
            count += 1

            # SMT-specific question
            if smt:
                smt_label = smt.replace("_", "/")
                resp = (
                    f"When trading {pair_label}, {smt_label} is your SMT divergence partner. "
                    f"SMT divergence occurs when one of these pairs makes a new high/low but the other doesn't. "
                    f"The pair that DOESN'T make the new extreme reveals the true direction.\n\n"
                    f"Example: If {pair_label} makes a new high but {smt_label} fails to follow, "
                    f"the {pair_label} high is likely a liquidity grab and price will reverse down.\n\n"
                    f"SMT at a liquidity sweep level is one of the highest-probability reversal signals "
                    f"in ICT methodology. Always check your SMT partner before confirming a sweep."
                )
                self._add(
                    f"How do I use {smt_label} for SMT divergence with {pair_label}?",
                    resp,
                    "pair_smt",
                )
                count += 1

        return count

    # ── 11. Pre-trade Validation ──────────────────────────────────────────

    def gen_pretrade_validation(self) -> int:
        count = 0
        rels = self._load_concept_rels()
        validation = rels.get("pre_trade_validation", {})

        if not validation:
            return 0

        must_all = validation.get("must_have_all", [])
        must_one = validation.get("must_have_one", [])
        should_have = validation.get("should_have", [])
        red_flags = validation.get("red_flags", [])

        # 1. Full checklist
        parts = ["ICT pre-trade validation checklist:\n\n"]
        if must_all:
            parts.append("MUST HAVE ALL (non-negotiable):\n")
            for item in must_all:
                parts.append(f"  □ {item.replace('_', ' ')}\n")
        if must_one:
            parts.append("\nMUST HAVE AT LEAST ONE:\n")
            for item in must_one:
                parts.append(f"  □ {item.replace('_', ' ')}\n")
        if should_have:
            parts.append("\nSHOULD HAVE (improves probability):\n")
            for item in should_have:
                parts.append(f"  □ {item.replace('_', ' ')}\n")
        if red_flags:
            parts.append("\nRED FLAGS (absolute disqualifiers):\n")
            for item in red_flags:
                parts.append(f"  ✗ {item.replace('_', ' ')}\n")

        parts.append(
            "\nRun this checklist BEFORE every trade. If any must-have is missing, no trade. "
            "If any red flag is present, no trade. No exceptions. The checklist exists to protect you "
            "from yourself — to override the emotional urge to enter when the setup isn't complete."
        )
        self._add(
            "What is the ICT pre-trade validation checklist?",
            "".join(parts),
            "pretrade_checklist",
        )
        count += 1

        # 2. Red flag scenarios
        for flag in red_flags:
            flag_label = flag.replace("_", " ")
            resp = (
                f"'{self._label(flag)}' is a red flag in the ICT pre-trade checklist — an absolute disqualifier.\n\n"
                f"No matter how good the rest of the setup looks, this condition means NO TRADE. "
                f"Red flags exist because they represent conditions where even valid setups have "
                f"a significantly reduced win rate.\n\n"
                f"The correct action: walk away and wait. The next clean setup is always coming. "
                f"Protecting capital on red flag days is what keeps you in the game long enough "
                f"to capitalize on the A+ setups when they appear.\n\n"
                f"Many traders rationalize away red flags. 'It looks so good otherwise.' "
                f"This is exactly when the checklist matters most — when your emotions want to override your rules."
            )
            self._add(
                f"Should I trade when I see '{flag_label}'?", resp, "pretrade_red_flag"
            )
            count += 1

        # 3. "All conditions met" scenario
        if must_all and must_one:
            conds = [c.replace("_", " ") for c in must_all]
            one_of = must_one[0].replace("_", " ")
            resp = (
                f"Validation check: You have {', '.join(conds)} (all must-haves) plus {one_of}.\n\n"
                f"Result: ✓ VALID SETUP — this trade passes pre-trade validation.\n\n"
                f"Next steps:\n"
                f"1. Check for red flags: {', '.join(f.replace('_', ' ') for f in red_flags[:3])}\n"
                f"2. Score the confluence to determine position size\n"
                f"3. Define your exact entry, stop loss, and take profit BEFORE entering\n"
                f"4. Execute with discipline — no adjustments after entry\n\n"
                f"Passing the checklist means you have permission to enter. It does NOT guarantee a win. "
                f"You're playing probabilities, and the checklist ensures you only take high-probability trades."
            )
            self._add(
                f"I have {', '.join(conds)} and {one_of}. Is this valid?",
                resp,
                "pretrade_valid",
            )
            count += 1

        # 4. Missing must-have scenarios
        for item in must_all[:3]:
            item_label = item.replace("_", " ")
            remaining = [c.replace("_", " ") for c in must_all if c != item]
            resp = (
                f"You have: {', '.join(remaining)}. But you're missing: {item_label}.\n\n"
                f"Result: ✗ INVALID — {item_label} is a must-have condition. Without it, "
                f"this trade does not pass pre-trade validation.\n\n"
                f"This is NOT a suggestion — it's a hard rule. The must-have conditions exist because "
                f"historical analysis shows trades without them have significantly lower win rates. "
                f"Taking trades that fail validation erodes your edge over time.\n\n"
                f"Wait for {item_label} to appear, or find a different setup that meets all requirements."
            )
            self._add(
                f"My setup has everything except {item_label}. Can I still trade?",
                resp,
                "pretrade_missing",
            )
            count += 1

        return count

    # ── 12. Multi-TF Alignment ────────────────────────────────────────────

    def gen_multi_tf(self) -> int:
        count = 0
        onto = self._load_ontology()
        rels = self._load_concept_rels()

        tf_hierarchy = onto.get("timeframe_hierarchy", {})

        # Full multi-TF explanation
        if tf_hierarchy:
            parts = ["ICT Multi-Timeframe analysis hierarchy:\n\n"]
            for tier, data in tf_hierarchy.items():
                if not isinstance(data, dict) or tier == "alignment_states":
                    continue
                label = self._label(tier)
                tfs = data.get("timeframes", [])
                purpose = data.get("purpose", "")
                identifies = data.get("identifies", [])
                rule = data.get("rule", "")

                parts.append(f"{label}:\n")
                if tfs:
                    parts.append(f"  Timeframes: {', '.join(tfs)}\n")
                parts.append(f"  Purpose: {purpose}\n")
                if identifies:
                    parts.append(f"  Identifies: {', '.join(identifies)}\n")
                if rule:
                    parts.append(f"  Rule: {rule}\n")
                parts.append("\n")

            alignment = tf_hierarchy.get("alignment_states", {})
            if alignment:
                parts.append("Alignment states:\n")
                for state, desc in alignment.items():
                    parts.append(f"  • {self._label(state)}: {desc}\n")

            parts.append(
                "\nThe multi-TF hierarchy is THE framework for ICT analysis. "
                "HTF tells you WHERE price is going. MTF shows you WHERE to enter. "
                "LTF gives you WHEN to execute. Skipping any level creates blind spots."
            )

            self._add(
                "How does multi-timeframe analysis work in ICT?",
                "".join(parts),
                "multi_tf",
            )
            count += 1

        # HTF to LTF chain from concept_rels
        chains = rels.get("causal_chains", {})
        htf_ltf = chains.get("htf_to_ltf", {})
        if htf_ltf:
            desc = htf_ltf.get("description", "")
            steps = htf_ltf.get("steps", {})
            failure = htf_ltf.get("failure_mode", "")

            parts = [f"The HTF-to-LTF alignment process ({desc}):\n\n"]
            for num, step in sorted(steps.items(), key=lambda x: int(x[0])):
                action = step.get("action", "")
                tfs = step.get("timeframes", [])
                concepts = step.get("concepts", [])
                patience = step.get("patience", "")

                parts.append(f"Step {num}: {action}")
                if tfs:
                    parts.append(f" [{', '.join(tfs)}]")
                if concepts:
                    parts.append(f" — look for: {', '.join(concepts)}")
                if patience:
                    parts.append(f" ({patience})")
                parts.append("\n")

            if failure:
                parts.append(f"\n⚠️ Failure mode: {failure}\n")

            parts.append(
                "\nThis is the most important process in ICT trading. Every successful trade "
                "starts with HTF direction, narrows to MTF zones, and executes on LTF confirmation. "
                "The discipline to follow this sequence separates profitable ICT traders from gamblers."
            )

            self._add(
                "Walk me through the HTF-to-LTF alignment process in ICT.",
                "".join(parts),
                "multi_tf",
            )
            count += 1

        # Individual TF tier questions
        for tier in ["htf_bias", "mtf_bridge", "ltf_execution"]:
            data = tf_hierarchy.get(tier, {})
            if not data:
                continue
            label = self._label(tier)
            tfs = data.get("timeframes", [])
            purpose = data.get("purpose", "")
            identifies = data.get("identifies", [])
            rule = data.get("rule", "")

            resp = (
                f"The {label} tier in ICT multi-timeframe analysis uses {', '.join(tfs)} timeframes.\n\n"
                f"Purpose: {purpose}\n"
                f"What it identifies: {', '.join(i.replace('_', ' ') for i in identifies)}\n"
                f"Rule: {rule}\n\n"
                f"This tier is {'the foundation — get the direction wrong and nothing else matters' if 'htf' in tier else 'the bridge that refines HTF zones into tradeable areas' if 'mtf' in tier else 'where you pull the trigger — but ONLY when HTF and MTF are aligned'}. "
                f"Many traders skip straight to {'the entry timeframe' if 'htf' in tier or 'mtf' in tier else 'the higher timeframes'}, "
                f"which is like reading the last page of a book and trying to understand the plot."
            )
            self._add(
                f"What is the role of {label} in ICT analysis?", resp, "multi_tf_tier"
            )
            count += 1

        # Alignment states
        alignment = tf_hierarchy.get("alignment_states", {})
        for state, desc in alignment.items():
            label = self._label(state)
            resp = (
                f"'{label}' in ICT multi-timeframe alignment means: {desc}\n\n"
                f"{'This is your green light — highest probability, full position size.' if 'full' in state else ''}"
                f"{'Proceed with caution — wait for clarity or reduce size.' if 'partial' in state else ''}"
                f"{'DO NOT TRADE. Conflicting signals = random outcome.' if 'no_' in state else ''}\n\n"
                f"The alignment state determines not just IF you trade, but HOW MUCH you risk. "
                f"Full alignment = standard risk. Partial = half size or wait. No alignment = no trade. "
                f"This position sizing based on alignment is a professional risk management technique "
                f"that many retail traders ignore — and it costs them."
            )
            self._add(
                f"What does '{label}' mean in ICT multi-TF analysis?",
                resp,
                "multi_tf_alignment",
            )
            count += 1

        return count

    # ── 13. Disambiguation ────────────────────────────────────────────────

    def gen_disambiguation(self) -> int:
        """Generate examples that explicitly disambiguate ICT terms from general usage."""
        count = 0

        disambiguations = [
            {
                "term": "Displacement",
                "wrong": "In physics, displacement refers to the change in position of an object — a vector quantity measuring the shortest distance from initial to final position.",
                "right": "In ICT (Inner Circle Trader) methodology, displacement is a large-bodied candle that shows institutional conviction, typically 2x or more the average candle range. Displacement indicates that smart money (institutional order flow) has committed to a direction. It is required to validate Fair Value Gaps (FVGs) and Order Blocks (OBs). Without displacement, these structures are unreliable. A displacement candle creates imbalances in price (FVGs) and marks the beginning of a directional move. It is the confirmation that a liquidity sweep was successful and that the real move has begun.",
            },
            {
                "term": "Order Block",
                "wrong": "In traditional trading, an order block might refer to any large institutional order or block trade on an exchange.",
                "right": "In ICT methodology, an Order Block (OB) is specifically defined as the last candle of opposite color before a displacement move. A bullish OB is the last DOWN candle before a bullish displacement. A bearish OB is the last UP candle before a bearish displacement. The OB forms AFTER a liquidity sweep and is validated by displacement. Entry is at the 50% (midpoint) of the OB candle body. Stop loss goes below the OB. An OB without a prior liquidity sweep and displacement is NOT a valid Order Block — it's just a candle.",
            },
            {
                "term": "Fair Value Gap",
                "wrong": "In fundamental analysis, fair value refers to the estimated true worth of an asset based on financial metrics, earnings, and intrinsic value calculations.",
                "right": "In ICT methodology, a Fair Value Gap (FVG) is a three-candle pattern where the wick of candle 1 and the wick of candle 3 don't overlap, creating a gap or imbalance. This gap represents an area where price moved too fast — only one side of the market was filled. The algorithm is programmed to return to this void to rebalance. FVGs are measured from WICKS, not bodies. An FVG is mitigated (used up) once price returns to its 50% level. FVGs have higher priority than Order Blocks — the algorithm returns to THE VOID first. A valid FVG requires displacement to form.",
            },
            {
                "term": "Liquidity",
                "wrong": "In traditional finance, liquidity refers to how easily an asset can be bought or sold without affecting its price — market depth and trading volume.",
                "right": "In ICT methodology, liquidity refers to clusters of resting orders — specifically stop losses and pending orders — sitting at predictable price levels. Buy-side liquidity (BSL) sits above equal highs, swing highs, and range highs (stop losses from shorts and buy orders from breakout traders). Sell-side liquidity (SSL) sits below equal lows, swing lows, and range lows. In ICT, price is DRAWN to liquidity — it's the fuel for smart money moves. Liquidity pools are targets, NOT support/resistance. The key rule: don't trade UNTIL liquidity is swept.",
            },
            {
                "term": "Premium and Discount",
                "wrong": "In retail, premium means expensive and discount means cheap — simple price comparisons relative to perceived value.",
                "right": "In ICT methodology, premium and discount are precisely defined relative to the equilibrium (50%) of the current dealing range. Price ABOVE the 50% level is in premium — this is where you look to SELL. Price BELOW the 50% level is in discount — this is where you look to BUY. This is rooted in the concept that smart money buys at wholesale (discount) and sells at retail (premium). Your entry should always be in discount for longs and premium for shorts. The Optimal Trade Entry (OTE) zone at the 62-79% retracement is the sweet spot within the premium/discount framework.",
            },
            {
                "term": "Manipulation",
                "wrong": "In general markets discussion, manipulation implies illegal activity — spoofing, wash trading, or insider trading that distorts prices.",
                "right": "In ICT methodology, manipulation is a natural, expected phase of the AMD (Accumulation, Manipulation, Distribution) cycle. It refers to the false move at session opens that sweeps liquidity from the accumulation phase. The Judas Swing at London or NY open is a classic manipulation move. It's not illegal activity — it's how institutional order flow naturally creates optimal entry conditions. Smart money needs liquidity (other traders' orders) to fill their large positions, so they push price to where those orders sit (equal highs/lows), sweep them, then move in the real direction. Understanding manipulation as a feature, not a bug, is key to ICT.",
            },
            {
                "term": "Mitigation",
                "wrong": "In risk management, mitigation means reducing the severity or probability of a risk event through preventive measures.",
                "right": "In ICT methodology, mitigation specifically refers to price returning to a PD Array (FVG or Order Block) and touching it. An FVG is 'mitigated' when price returns to its 50% level after formation. An Order Block is mitigated when price taps its zone. Once mitigated, the PD Array is considered 'used up' and should not be used for another entry. Trading already-mitigated levels is a common anti-pattern — the zone has already served its purpose. Only unmitigated PD Arrays are valid for entries. The distinction between mitigated and unmitigated is critical for identifying fresh vs. stale entry zones.",
            },
            {
                "term": "Imbalance",
                "wrong": "In trading, imbalance generally refers to a mismatch between buy and sell orders in the order book — visible in Level 2 data or delta.",
                "right": "In ICT methodology, imbalance is synonymous with Fair Value Gap (FVG). It refers to a price area where the market moved too fast, leaving only one side's orders filled. A buy-side imbalance creates a sell-side inefficiency (BISI) — price moved up too fast and will return to fill the gap. A sell-side imbalance creates a buy-side inefficiency (SIBI) — price moved down too fast and will return. The key: the algorithm is programmed to return to these imbalances to 'rebalance' the delivery of price. These areas become your entry zones.",
            },
            {
                "term": "Structure",
                "wrong": "In technical analysis, structure typically means chart patterns like head and shoulders, triangles, or channels — geometric shapes formed by price.",
                "right": "In ICT methodology, market structure refers specifically to the sequence of swing highs and swing lows, and how they break. Break of Structure (BOS) means price closes beyond a previous swing high/low — this indicates trend continuation. Change of Character (CHoCH) is the FIRST break against the prevailing trend — a warning signal. Market Structure Shift (MSS) is a confirmed reversal after a liquidity sweep. The key rule: the BODY must close beyond the level, not just the wick. A wick through a level is a stop hunt, not a real break. Structure tells you the story of institutional order flow.",
            },
            {
                "term": "Sweep",
                "wrong": "In everyday trading slang, a sweep might mean scanning multiple stocks or a broad market move.",
                "right": "In ICT methodology, a sweep (or liquidity sweep) is the specific event where price reaches a liquidity pool, takes out the resting orders (stop losses), and then reverses. Sweeps typically push 20-30 pips through the level and can happen via wick or body close. After the sweep, look for displacement in the opposite direction — this confirms the reversal. A sweep is 'fluid' and 'destined' — it's the natural result of smart money needing to fill orders. The sequence is: sweep happens → displacement follows → FVG/OB forms → you enter on the retracement. Never enter BEFORE the sweep.",
            },
            {
                "term": "Silver Bullet",
                "wrong": "A silver bullet generally means a simple guaranteed solution to a complex problem.",
                "right": "In ICT methodology, the Silver Bullet is a specific time-based trading model. It occurs in two windows: the AM Silver Bullet (10:00-11:00 ET) and the PM Silver Bullet (2:00-3:00 PM ET). The setup requires: an FVG that forms WITHIN the time window, in the direction of HTF bias, AFTER liquidity has been swept. Entry is on the first tap of the FVG. Target is opposite liquidity or 15-20 pips. The Silver Bullet is strictly time-gated — an identical setup at 9:30 AM or 3:30 PM does NOT qualify. The time window captures specific institutional order flow patterns that repeat daily.",
            },
            {
                "term": "Terminus",
                "wrong": "A terminus is an endpoint — the final station of a railway line or the end point of a journey.",
                "right": "In ICT methodology, the terminus is the final objective of the Market Maker Model — a PREDETERMINED target identified BEFORE price reaches it. The terminus is located at discount PD arrays for buys and premium PD arrays for sells. The critical rule: you must identify the terminus before the move, not after. If you're naming the terminus after price arrives, you're not predicting — you're narrating. The terminus anchors your entire trade plan — your entry, stop, and target are all relative to where you believe smart money is delivering price to.",
            },
        ]

        for d in disambiguations:
            term = d["term"]
            right = d["right"]
            wrong = d["wrong"]

            # Standard definition
            self._add(
                f"What is {term} in ICT trading methodology?", right, "disambiguation"
            )
            count += 1

            # Explicit disambiguation
            resp = (
                f"Important distinction — do NOT confuse ICT's use of '{term}' with other contexts.\n\n"
                f"General/other meaning: {wrong}\n\n"
                f"In ICT methodology: {right}"
            )
            self._add(
                f"How is '{term}' different in ICT compared to general trading?",
                resp,
                "disambiguation",
            )
            count += 1

        return count

    # ── 14. Existing QA (enriched) ────────────────────────────────────────

    def gen_existing_qa(self) -> int:
        count = 0

        for qa_file in sorted(QA_DIR.glob("*.json")):
            try:
                with open(qa_file) as fh:
                    qa_data = json.load(fh)
            except Exception:
                continue

            if not isinstance(qa_data, list):
                continue

            for item in qa_data:
                question = item.get("question", "")
                explanation = item.get("explanation", "")
                correct = item.get("correct_answer", "")
                options = item.get("options", {})
                category = item.get("category", "general")

                if not question or not explanation:
                    continue

                # Build enriched response
                parts = []
                if correct and options:
                    correct_text = options.get(correct, "")
                    parts.append(
                        f"The correct answer is {correct}) {correct_text}.\n\n"
                    )

                parts.append(f"{explanation}\n\n")

                # Add wrong answer analysis for richer training
                if options and correct:
                    wrong_opts = {k: v for k, v in options.items() if k != correct}
                    if wrong_opts:
                        parts.append("Why the other options are wrong:\n")
                        for k, v in wrong_opts.items():
                            parts.append(
                                f"  {k}) {v} — incorrect because this doesn't match the ICT definition.\n"
                            )
                        parts.append("\n")

                parts.append(
                    f"This concept is part of the {category.replace('_', ' ')} area of ICT methodology."
                )

                self._add(question, "".join(parts), "existing_qa")
                count += 1

        return count

    # ── Export ─────────────────────────────────────────────────────────────

    def _add(
        self,
        instruction: str,
        output: str,
        category: str,
        input_text: Optional[str] = None,
    ):
        """Add a training example."""
        self.examples.append(
            {
                "instruction": instruction.strip(),
                "input": (input_text or "").strip(),
                "output": output.strip(),
                "category": category,
            }
        )

    def export(
        self,
        filename: str = "ict_v3_training.jsonl",
        fmt: str = "chatml",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Export to JSONL."""
        output_dir = output_dir or _OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        examples = list(self.examples)
        self.rng.shuffle(examples)

        with open(output_path, "w") as f:
            for ex in examples:
                if fmt == "chatml":
                    messages = [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": ex["instruction"]},
                    ]
                    if ex.get("input"):
                        messages[-1]["content"] += f"\n\n{ex['input']}"
                    messages.append({"role": "assistant", "content": ex["output"]})
                    record = {"messages": messages}
                elif fmt == "alpaca":
                    record = {
                        "instruction": ex["instruction"],
                        "input": ex.get("input", ""),
                        "output": ex["output"],
                    }
                elif fmt == "sharegpt":
                    convos = [
                        {"from": "system", "value": self.SYSTEM_PROMPT},
                        {
                            "from": "human",
                            "value": ex["instruction"]
                            + (f"\n\n{ex['input']}" if ex.get("input") else ""),
                        },
                        {"from": "gpt", "value": ex["output"]},
                    ]
                    record = {"conversations": convos}
                else:
                    raise ValueError(f"Unknown format: {fmt}")

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[v3] Exported {len(examples)} examples → {output_path} ({fmt})")
        return output_path

    def export_train_valid_split(
        self,
        valid_ratio: float = 0.1,
        fmt: str = "chatml",
        output_dir: Optional[Path] = None,
    ) -> tuple:
        """Export with train/valid split."""
        output_dir = output_dir or _OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = list(self.examples)
        self.rng.shuffle(examples)

        split_idx = int(len(examples) * (1 - valid_ratio))
        train = examples[:split_idx]
        valid = examples[split_idx:]

        orig = self.examples

        self.examples = train
        train_path = self.export("train.jsonl", fmt=fmt, output_dir=output_dir)

        self.examples = valid
        valid_path = self.export("valid.jsonl", fmt=fmt, output_dir=output_dir)

        self.examples = orig

        print(f"[v3] Split: {len(train)} train / {len(valid)} valid")
        return train_path, valid_path

    def get_stats(self) -> dict:
        """Return detailed statistics."""
        by_category = defaultdict(int)
        lengths = []

        for ex in self.examples:
            by_category[ex.get("category", "unknown")] += 1
            lengths.append(len(ex["output"]))

        lengths.sort()
        return {
            "total_examples": len(self.examples),
            "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
            "response_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "median": lengths[len(lengths) // 2] if lengths else 0,
                "mean": int(sum(lengths) / len(lengths)) if lengths else 0,
                "under_200": sum(1 for l in lengths if l < 200),
                "200_500": sum(1 for l in lengths if 200 <= l < 500),
                "500_800": sum(1 for l in lengths if 500 <= l < 800),
                "over_800": sum(1 for l in lengths if l >= 800),
            },
            "estimated_tokens": sum(lengths) // 4,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate ICT V3 training data")
    parser.add_argument(
        "--format", "-f", default="chatml", choices=["alpaca", "chatml", "sharegpt"]
    )
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument(
        "--split", action="store_true", help="Generate train/valid split"
    )
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = parser.parse_args()

    gen = TrainingDataGeneratorV3()
    gen.generate_all()

    out_dir = Path(args.output) if args.output else None

    if args.split:
        gen.export_train_valid_split(fmt=args.format, output_dir=out_dir)
    else:
        gen.export(fmt=args.format, output_dir=out_dir)

    if args.stats:
        import pprint

        pprint.pprint(gen.get_stats())


if __name__ == "__main__":
    main()
