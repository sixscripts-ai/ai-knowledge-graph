"""
Phase 5: Graph-Driven Trade Logic Engine
==========================================
Replaces hardcoded model selection in VEX with graph-traversal-based
reasoning. The graph IS the logic.

Flow:
    1. Detectors fire → detected patterns become graph nodes
    2. Traverse edges to find which models require these patterns
    3. Score confluence by counting satisfied prerequisites
    4. Select highest-confluence model
    5. Generate explainable decision trace

Usage:
    from graph_rag import ICTGraphStore, TradeReasoner
    
    store = ICTGraphStore().load_all()
    reasoner = TradeReasoner(store)
    
    # Simulate detector output
    signals = {
        "patterns": ["fvg", "displacement", "order_block"],
        "liquidity_swept": True,
        "htf_bias": "bullish",
        "session": "ny_am",
        "time": "10:15",
    }
    
    result = reasoner.evaluate(signals)
    print(result.recommendation)  # "silver_bullet"
    print(result.score)           # 8.5
    print(result.explanation)     # Step-by-step reasoning
    print(result.go_no_go)        # True / False
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .graph_store import ICTGraphStore, _normalize

try:
    import yaml
except ImportError:
    yaml = None

_TRAIN_ICT = Path(os.environ.get(
    "TRAIN_ICT_ROOT",
    Path.home() / "Documents" / "train-ict"
))
CONCEPT_RELS_PATH = _TRAIN_ICT / "data" / "schemas" / "concept_relationships.yaml"


@dataclass
class TradeDecision:
    """Result of the logic engine's evaluation."""
    recommendation: Optional[str] = None
    score: float = 0.0
    go_no_go: bool = False
    explanation: list[str] = field(default_factory=list)
    model_scores: dict[str, float] = field(default_factory=dict)
    satisfied_prerequisites: list[str] = field(default_factory=list)
    missing_prerequisites: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    confluence_factors: dict[str, float] = field(default_factory=dict)
    graph_path: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable decision summary."""
        status = "GO" if self.go_no_go else "NO-GO"
        lines = [
            f"=== Trade Decision: {status} ===",
            f"Recommended Model: {self.recommendation or 'None'}",
            f"Confluence Score: {self.score:.1f}",
            "",
            "--- Reasoning ---",
        ]
        for step in self.explanation:
            lines.append(f"  {step}")

        if self.red_flags:
            lines.append("\n--- RED FLAGS ---")
            for flag in self.red_flags:
                lines.append(f"  !! {flag}")

        if self.missing_prerequisites:
            lines.append("\n--- Missing ---")
            for m in self.missing_prerequisites:
                lines.append(f"  - {m}")

        if self.model_scores:
            lines.append("\n--- Model Rankings ---")
            for model, score in sorted(self.model_scores.items(),
                                       key=lambda x: -x[1]):
                marker = " <--" if model == self.recommendation else ""
                lines.append(f"  {model}: {score:.1f}{marker}")

        return "\n".join(lines)


class TradeReasoner:
    """Graph-driven trade reasoning engine for VEX."""

    def __init__(self, store: ICTGraphStore):
        self.store = store
        self._concept_rels = None
        self._confluence_weights = {}
        self._thresholds = {}
        self._validation_rules = {}
        self._model_blueprints = {}
        self._killzone_models = {}

        self._load_rules()

    def _load_rules(self):
        """Load trading rules from concept_relationships.yaml."""
        if not CONCEPT_RELS_PATH.exists() or yaml is None:
            return

        with open(CONCEPT_RELS_PATH, "r") as f:
            data = yaml.safe_load(f) or {}

        self._concept_rels = data

        # Confluence weights
        weights = data.get("confluence_weights", {})
        for tier in ("critical", "high", "moderate", "bonuses"):
            if isinstance(weights.get(tier), dict):
                for factor, weight in weights[tier].items():
                    self._confluence_weights[_normalize(factor)] = float(weight)

        # Penalties (negative weights)
        penalties = weights.get("penalties", {})
        if isinstance(penalties, dict):
            for factor, weight in penalties.items():
                self._confluence_weights[_normalize(factor)] = float(weight)

        # Thresholds
        self._thresholds = weights.get("thresholds", {
            "minimum_for_trade": 5.0,
            "good_setup": 7.0,
            "a_plus_setup": 9.0,
        })

        # Pre-trade validation
        self._validation_rules = data.get("pre_trade_validation", {})

        # Model blueprints
        self._model_blueprints = data.get("models", {})

        # Killzone → model mapping
        killzones = data.get("time_rules", {}).get("killzones", {})
        for kz_name, kz_data in killzones.items():
            for setup in (kz_data.get("best_setups") or []):
                self._killzone_models.setdefault(_normalize(kz_name), []).append(
                    _normalize(setup)
                )

    # ── Main evaluation ───────────────────────────────────────────────────

    def evaluate(self, signals: dict) -> TradeDecision:
        """Evaluate a set of market signals and return a trade decision.
        
        Args:
            signals: Dictionary with detected market conditions:
                - patterns: list[str]  — detected ICT patterns (fvg, ob, etc.)
                - liquidity_swept: bool
                - htf_bias: str  — "bullish", "bearish", or "neutral"
                - session: str  — current trading session
                - time: str  — current time (HH:MM)
                - displacement: bool
                - in_killzone: bool
                - smt_divergence: bool
                - at_ote: bool
                - ... any other confluence factors
        
        Returns:
            TradeDecision with recommendation, score, and explanation
        """
        decision = TradeDecision()

        # Step 1: Map signals to graph nodes
        detected = self._map_signals_to_graph(signals, decision)

        # Step 2: Check red flags (absolute disqualifiers)
        self._check_red_flags(signals, decision)

        # Step 3: Score confluence
        self._score_confluence(signals, detected, decision)

        # Step 4: Find matching models via graph traversal
        self._find_matching_models(detected, signals, decision)

        # Step 5: Apply thresholds
        min_score = self._thresholds.get("minimum_for_trade", 5.0)
        good_score = self._thresholds.get("good_setup", 7.0)
        aplus_score = self._thresholds.get("a_plus_setup", 9.0)

        if decision.red_flags:
            decision.go_no_go = False
            decision.explanation.append(
                f"NO-GO: {len(decision.red_flags)} red flag(s) detected"
            )
        elif decision.score >= min_score and decision.recommendation:
            decision.go_no_go = True
            quality = ("A+" if decision.score >= aplus_score
                       else "Good" if decision.score >= good_score
                       else "Minimum")
            decision.explanation.append(
                f"GO: {quality} setup ({decision.score:.1f}) — "
                f"Model: {decision.recommendation}"
            )
        else:
            decision.go_no_go = False
            reason = (f"Score {decision.score:.1f} < minimum {min_score}"
                      if decision.score < min_score
                      else "No matching model found")
            decision.explanation.append(f"NO-GO: {reason}")

        return decision

    # ── Step 1: Signal → Graph mapping ────────────────────────────────────

    def _map_signals_to_graph(self, signals: dict,
                               decision: TradeDecision) -> set[str]:
        """Map detector signals to graph node names."""
        detected = set()

        # Direct pattern names
        for pattern in signals.get("patterns", []):
            norm = _normalize(pattern)
            if norm in self.store.G:
                detected.add(norm)
                decision.explanation.append(f"Detected: {norm}")
            else:
                # Try fuzzy match
                matches = self.store.search(pattern, top_k=1)
                if matches:
                    detected.add(matches[0]["node"])
                    decision.explanation.append(
                        f"Detected: {pattern} (matched to {matches[0]['node']})"
                    )

        # Boolean signals
        bool_mapping = {
            "liquidity_swept": "liquidity_swept",
            "displacement": "displacement_confirmed",
            "in_killzone": "in_killzone",
            "smt_divergence": "smt_divergence",
            "at_ote": "at_ote_level",
            "htf_aligned": "htf_bias_aligned",
        }

        for signal_key, graph_node in bool_mapping.items():
            if signals.get(signal_key):
                detected.add(_normalize(graph_node))
                decision.explanation.append(f"Confirmed: {graph_node}")

        # HTF bias
        htf_bias = signals.get("htf_bias", "").lower()
        if htf_bias in ("bullish", "bearish"):
            detected.add("htf_bias_aligned")
            decision.explanation.append(f"HTF bias: {htf_bias}")

        # Session/killzone
        session = _normalize(signals.get("session", ""))
        if session:
            kz_node = f"killzone_{session}" if not session.startswith("killzone") else session
            if kz_node in self.store.G or _normalize(session) in self.store.G:
                detected.add(kz_node)
                decision.explanation.append(f"Session: {session}")

        return detected

    # ── Step 2: Red flags ─────────────────────────────────────────────────

    def _check_red_flags(self, signals: dict, decision: TradeDecision):
        """Check for absolute disqualifiers."""
        red_flags = self._validation_rules.get("red_flags", [])

        flag_checks = {
            "against_htf_bias": (
                signals.get("htf_bias") == "neutral" or
                (signals.get("trade_direction") and
                 signals.get("htf_bias") and
                 signals.get("trade_direction") != signals.get("htf_bias"))
            ),
            "no_displacement": not signals.get("displacement", False),
            "news_in_30_min": signals.get("news_imminent", False),
            "already_2_trades_today": signals.get("trades_today", 0) >= 2,
            "revenge_trading": signals.get("revenge_trading", False),
            "outside_all_killzones": (
                not signals.get("in_killzone") and
                signals.get("session") not in ("london", "ny_am", "ny_pm")
            ),
            "no_clear_invalidation": not signals.get("stop_loss_defined", True),
        }

        # Always check no_displacement even if not in the YAML red_flags list
        all_flags_to_check = set(red_flags) | {"no_displacement"}

        for flag in all_flags_to_check:
            flag_norm = _normalize(flag)
            if flag_checks.get(flag_norm, False):
                decision.red_flags.append(flag)
                decision.explanation.append(f"RED FLAG: {flag}")

    # ── Step 3: Confluence scoring ────────────────────────────────────────

    def _score_confluence(self, signals: dict, detected: set[str],
                          decision: TradeDecision):
        """Score the setup using graph-derived confluence weights."""
        total_score = 0.0

        for factor, weight in self._confluence_weights.items():
            # Check if this factor is in the detected set
            factor_present = (
                factor in detected or
                any(factor in d for d in detected) or
                signals.get(factor.replace("_", ""), False) or
                signals.get(factor, False)
            )

            if factor_present:
                total_score += weight
                decision.confluence_factors[factor] = weight
                if weight > 0:
                    decision.satisfied_prerequisites.append(
                        f"{factor} (+{weight})"
                    )
                else:
                    decision.explanation.append(
                        f"Penalty: {factor} ({weight})"
                    )

        # Check for unicorn setup (OB + FVG together)
        if "order_block" in detected and ("fvg" in detected or
                                           "fair_value_gap" in detected):
            bonus = self._confluence_weights.get("unicorn_setup", 2.0)
            total_score += bonus
            decision.confluence_factors["unicorn_setup"] = bonus
            decision.explanation.append(f"Bonus: Unicorn setup (+{bonus})")

        # Check for multi-TF confluence
        if signals.get("multi_tf_aligned"):
            bonus = self._confluence_weights.get("multi_timeframe_confluence", 1.5)
            total_score += bonus
            decision.confluence_factors["multi_timeframe_confluence"] = bonus
            decision.explanation.append(f"Bonus: Multi-TF confluence (+{bonus})")

        decision.score = total_score

    # ── Step 4: Model matching via graph traversal ────────────────────────

    def _find_matching_models(self, detected: set[str], signals: dict,
                               decision: TradeDecision):
        """Find the best matching ICT model by traversing the graph.
        
        For each model, check how many of its prerequisites are satisfied
        by the detected patterns.  Scores are *specificity-normalized*:
        requirements that appear in many models contribute less than
        requirements unique to a single model.  This prevents generic
        models (e.g. silver_bullet) from always winning.
        """

        # ── Pre-compute requirement frequency across ALL models ──────────
        # requirement text (lowered) → set of model names that include it
        req_models: dict[str, set[str]] = defaultdict(set)
        for model_name, blueprint in self._model_blueprints.items():
            for req_str in blueprint.get("required", []):
                req_models[req_str.lower()].add(model_name)

        num_models = max(len(self._model_blueprints), 1)

        for model_name, blueprint in self._model_blueprints.items():
            model_norm = _normalize(model_name)
            required = blueprint.get("required", [])
            if not required:
                continue

            # Count how many requirements are satisfied
            satisfied = 0
            weighted_score = 0.0
            total = len(required)
            missing = []

            for req_str in required:
                req_lower = req_str.lower()
                # Check if any detected pattern matches this requirement
                is_met = False

                for det in detected:
                    det_label = det.replace("_", " ")
                    if det_label in req_lower or det in req_lower:
                        is_met = True
                        break

                # Also check via graph — does any detected concept
                # have a path to a concept mentioned in this requirement?
                if not is_met:
                    for det in detected:
                        concepts_in_req = self.store._extract_concepts_from_text(req_str)
                        for concept in concepts_in_req:
                            if concept in detected:
                                is_met = True
                                break
                            # Check 1-hop neighbors
                            neighbors = self.store.get_neighbors(det)
                            for n in neighbors:
                                if n["node"] == concept:
                                    is_met = True
                                    break
                            if is_met:
                                break
                        if is_met:
                            break

                if is_met:
                    satisfied += 1
                    # Specificity weight: unique reqs contribute more
                    # shared-by-all → 1/num_models, unique → 1.0
                    sharing = len(req_models.get(req_lower, {model_name}))
                    specificity = 1.0 / sharing
                    weighted_score += specificity
                else:
                    missing.append(req_str)

            # Compute blended score:
            #   base   = percentage satisfied  (0-10 range)
            #   spec   = specificity-weighted  (rewards unique matches)
            #   bonus  = perfect-fit bonus when ALL requirements are met
            if total > 0:
                max_specificity = sum(
                    1.0 / len(req_models.get(r.lower(), {model_name}))
                    for r in required
                )
                base_score = (satisfied / total) * 7.0
                spec_score = (weighted_score / max(max_specificity, 0.01)) * 3.0
                model_score = base_score + spec_score
                # Perfect-fit bonus: all requirements met
                if satisfied == total:
                    model_score += 1.5
            else:
                model_score = 0

            # Time window bonus
            time_windows = blueprint.get("time_windows", [])
            current_time = signals.get("time", "")
            if time_windows and current_time:
                for tw in time_windows:
                    start = tw.get("start", "")
                    end = tw.get("end", "")
                    if start and end and start <= current_time <= end:
                        model_score += 2.0
                        break

            # Session bonus from killzone mapping
            session = _normalize(signals.get("session", ""))
            if session in self._killzone_models:
                if model_norm in self._killzone_models[session]:
                    model_score += 1.5

            decision.model_scores[model_name] = model_score

        # Select best model
        if decision.model_scores:
            best_model = max(decision.model_scores, key=decision.model_scores.get)
            best_score = decision.model_scores[best_model]

            if best_score > 0:
                decision.recommendation = best_model
                decision.explanation.append(
                    f"Best model: {best_model} (fit score: {best_score:.1f})"
                )

                # Find missing prerequisites for the chosen model
                blueprint = self._model_blueprints.get(best_model, {})
                for req in blueprint.get("required", []):
                    req_lower = req.lower()
                    is_met = any(
                        det.replace("_", " ") in req_lower or det in req_lower
                        for det in detected
                    )
                    if not is_met:
                        decision.missing_prerequisites.append(req)

                # Build graph path for explainability
                for det in detected:
                    path = self.store.find_path_with_relations(
                        det, _normalize(best_model)
                    )
                    if path:
                        decision.graph_path.extend(path)
                        break

    # ── Convenience methods ───────────────────────────────────────────────

    def quick_check(self, patterns: list[str],
                    htf_bias: str = "bullish",
                    session: str = "ny_am") -> TradeDecision:
        """Quick evaluation with minimal signal input."""
        signals = {
            "patterns": patterns,
            "htf_bias": htf_bias,
            "session": session,
            "htf_aligned": htf_bias in ("bullish", "bearish"),
            "in_killzone": session in ("london", "ny_am", "ny_pm"),
        }

        # Infer booleans from patterns
        pattern_set = {_normalize(p) for p in patterns}
        if "displacement" in pattern_set:
            signals["displacement"] = True
        if any(p in pattern_set for p in ("fvg", "fair_value_gap")):
            signals["patterns"].append("in_fvg")
        if "order_block" in pattern_set:
            signals["patterns"].append("at_order_block")
        if "liquidity_sweep" in pattern_set or "liquidity_swept" in pattern_set:
            signals["liquidity_swept"] = True

        return self.evaluate(signals)

    def explain_model(self, model_name: str) -> dict:
        """Get full graph-derived explanation of a model's requirements."""
        model_norm = _normalize(model_name)
        blueprint = self._model_blueprints.get(model_name,
                    self._model_blueprints.get(model_norm, {}))

        if not blueprint:
            return {"error": f"Model '{model_name}' not found"}

        result = {
            "model": model_name,
            "description": blueprint.get("description", ""),
            "required": blueprint.get("required", []),
            "time_windows": blueprint.get("time_windows", []),
            "avoid_when": blueprint.get("avoid_when", []),
            "graph_connections": {},
        }

        # Get graph connections for each requirement
        for req in blueprint.get("required", []):
            concepts = self.store._extract_concepts_from_text(req)
            for concept in concepts:
                neighbors = self.store.get_neighbors(concept)
                if neighbors:
                    result["graph_connections"][concept] = [
                        {"node": n["node"], "relation": n["relation"]}
                        for n in neighbors[:5]
                    ]

        return result

    def what_if(self, base_signals: dict,
                add_pattern: Optional[str] = None,
                remove_pattern: Optional[str] = None) -> tuple[TradeDecision, TradeDecision]:
        """Compare two scenarios: with and without a specific pattern.
        
        Useful for understanding marginal impact of each signal.
        """
        # Base case
        base_decision = self.evaluate(base_signals)

        # Modified case
        modified_signals = dict(base_signals)
        if add_pattern:
            modified_signals.setdefault("patterns", [])
            modified_signals["patterns"] = list(modified_signals["patterns"]) + [add_pattern]
        if remove_pattern:
            modified_signals["patterns"] = [
                p for p in modified_signals.get("patterns", [])
                if _normalize(p) != _normalize(remove_pattern)
            ]

        modified_decision = self.evaluate(modified_signals)

        return base_decision, modified_decision


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """CLI: test the logic engine with sample scenarios."""
    from .graph_store import ICTGraphStore

    store = ICTGraphStore().load_all()
    reasoner = TradeReasoner(store)

    print("\n=== ICT Trade Logic Engine ===\n")

    # Scenario 1: Strong setup
    print("--- Scenario 1: Strong Silver Bullet Setup ---")
    result = reasoner.quick_check(
        patterns=["fvg", "displacement", "liquidity_sweep"],
        htf_bias="bullish",
        session="ny_am",
    )
    print(result.summary())

    # Scenario 2: Weak setup
    print("\n--- Scenario 2: Weak Setup (no liquidity sweep) ---")
    result = reasoner.quick_check(
        patterns=["fvg"],
        htf_bias="neutral",
        session="ny_am",
    )
    print(result.summary())

    # Scenario 3: Unicorn
    print("\n--- Scenario 3: Unicorn Setup ---")
    result = reasoner.quick_check(
        patterns=["order_block", "fvg", "displacement", "liquidity_sweep"],
        htf_bias="bearish",
        session="london",
    )
    print(result.summary())

    # Interactive mode
    print("\n\n=== Interactive Mode ===")
    print("Enter patterns separated by commas (e.g.: fvg, displacement, liquidity_sweep)")
    print("Type 'exit' to quit\n")

    while True:
        try:
            inp = input("Patterns: ").strip()
            if inp.lower() in ("exit", "quit", "q"):
                break
            if not inp:
                continue

            patterns = [p.strip() for p in inp.split(",")]
            bias = input("HTF bias (bullish/bearish/neutral): ").strip() or "bullish"
            session = input("Session (london/ny_am/ny_pm): ").strip() or "ny_am"

            result = reasoner.quick_check(patterns, htf_bias=bias, session=session)
            print("\n" + result.summary() + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
