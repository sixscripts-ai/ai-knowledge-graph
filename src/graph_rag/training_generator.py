"""
Phase 3: Fine-Tuning Data Generator
=====================================
Generates structured training data from the knowledge graph + existing
trade examples for LoRA fine-tuning.

Output formats:
    - Alpaca: {"instruction", "input", "output"}
    - ChatML: {"messages": [{"role": ..., "content": ...}]}
    - ShareGPT: {"conversations": [{"from": ..., "value": ...}]}

Data types generated:
    1. Concept Q&A — definitions + relationships from graph
    2. Reasoning chains — multi-hop graph traversals as CoT
    3. Trade analysis — annotated trade examples with graph context
    4. Counterfactuals — why trades failed + graph-derived fixes
    5. Model selection — given patterns, which model applies and why
    6. Pre-trade validation — checklist-style reasoning

Usage:
    from graph_rag import ICTGraphStore, TrainingDataGenerator
    
    store = ICTGraphStore().load_all()
    gen = TrainingDataGenerator(store)
    gen.generate_all()
    gen.export("training_data.jsonl", fmt="chatml")
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .graph_store import ICTGraphStore, _normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
_TRAIN_ICT = Path(os.environ.get(
    "TRAIN_ICT_ROOT",
    Path.home() / "Documents" / "train-ict"
))
_OUTPUT_DIR = Path(os.environ.get(
    "TRAINING_OUTPUT",
    Path.home() / "Documents" / "knowledge_graph_ict" / "ai-knowledge-graph" / "training_output"
))

POSITIVE_DIR = _TRAIN_ICT / "data" / "training" / "positive"
NEGATIVE_DIR = _TRAIN_ICT / "data" / "training" / "negative"
BLIND_TEST_DIR = _TRAIN_ICT / "data" / "training" / "blind_test"
QA_DIR = _TRAIN_ICT / "data" / "training" / "qa"

try:
    import yaml
except ImportError:
    yaml = None

CONCEPT_RELS_PATH = _TRAIN_ICT / "data" / "schemas" / "concept_relationships.yaml"


class TrainingDataGenerator:
    """Generate fine-tuning data from ICT knowledge graph."""

    SYSTEM_PROMPT = (
        "You are VEX, an expert ICT (Inner Circle Trader) trading assistant. "
        "You analyze market structure, price delivery arrays, liquidity, and "
        "time-based models to identify high-probability trade setups. "
        "You reason step-by-step through causal chains and confluence factors."
    )

    def __init__(self, store: ICTGraphStore, seed: int = 42):
        self.store = store
        self.rng = random.Random(seed)
        self.examples: list[dict] = []
        self._concept_rels = None

    def _load_concept_rels(self) -> dict:
        """Load concept_relationships.yaml for template generation."""
        if self._concept_rels is not None:
            return self._concept_rels
        if CONCEPT_RELS_PATH.exists() and yaml is not None:
            with open(CONCEPT_RELS_PATH, "r") as f:
                self._concept_rels = yaml.safe_load(f) or {}
        else:
            self._concept_rels = {}
        return self._concept_rels

    # ── Generate all ───────────────────────────────────────────────────────

    def generate_all(self) -> int:
        """Generate all training data types. Returns total example count."""
        self.examples = []

        n1 = self.generate_concept_qa()
        n2 = self.generate_relationship_qa()
        n3 = self.generate_reasoning_chains()
        n4 = self.generate_model_selection()
        n5 = self.generate_trade_analysis()
        n6 = self.generate_counterfactuals()
        n7 = self.generate_pretrade_validation()
        n8 = self.generate_anti_pattern_qa()
        n9 = self.generate_existing_qa()
        n10 = self.generate_neighborhood_summaries()
        n11 = self.generate_concept_comparisons()

        total = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11
        print(f"\n[training_gen] Total examples: {total}")
        print(f"  Concept Q&A:            {n1}")
        print(f"  Relationship Q&A:       {n2}")
        print(f"  Reasoning chains:       {n3}")
        print(f"  Model selection:        {n4}")
        print(f"  Trade analysis:         {n5}")
        print(f"  Counterfactuals:        {n6}")
        print(f"  Pre-trade validation:   {n7}")
        print(f"  Anti-pattern Q&A:       {n8}")
        print(f"  Existing Q&A import:    {n9}")
        print(f"  Neighborhood summaries: {n10}")
        print(f"  Concept comparisons:    {n11}")
        return total

    # ── Type 1: Concept definitions + properties ──────────────────────────

    def generate_concept_qa(self) -> int:
        """Generate Q&A pairs about concept definitions from graph nodes."""
        count = 0
        for node, data in self.store.G.nodes(data=True):
            defn = data.get("definition", "")
            node_label = node.replace("_", " ").title()

            if defn and isinstance(defn, str) and len(defn) >= 10:
                # Has explicit definition
                self._add(
                    instruction=f"What is {node_label} in ICT trading methodology?",
                    output=f"{node_label} is defined as: {defn}",
                    category="concept_definition",
                )
                count += 1
            else:
                # No definition — synthesize from graph relationships
                out_edges = list(self.store.G.out_edges(node, data=True))
                in_edges = list(self.store.G.in_edges(node, data=True))
                if len(out_edges) >= 2:
                    # Build context from outgoing edges
                    facts = []
                    for _, tgt, edata in out_edges[:5]:
                        rel = edata.get("relation", "relates to")
                        tgt_label = tgt.replace("_", " ").title()
                        facts.append(f"- {rel.replace('_', ' ')} {tgt_label}")
                    facts_text = "\n".join(facts)
                    self._add(
                        instruction=f"What is {node_label} in ICT trading?",
                        output=f"{node_label} is an ICT concept with these key relationships:\n{facts_text}",
                        category="concept_from_graph",
                    )
                    count += 1

            # If has type info
            node_type = data.get("type", "")
            if node_type and node_type not in ("concept", "unknown"):
                self._add(
                    instruction=f"What category does {node_label} belong to in ICT?",
                    output=f"{node_label} is classified as a {node_type} in the ICT framework.",
                    category="concept_classification",
                )
                count += 1

        print(f"[training_gen] Generated {count} concept Q&A examples")
        return count

    # ── Type 2: Relationship-based Q&A ────────────────────────────────────

    def generate_relationship_qa(self) -> int:
        """Generate Q&A about concept relationships from graph edges."""
        count = 0

        # Group edges by relation type
        by_relation: dict[str, list] = defaultdict(list)
        for src, tgt, data in self.store.G.edges(data=True):
            rel = data.get("relation", "related")
            by_relation[rel].append((src, tgt, data))

        # Templates for each relation type
        templates = {
            "requires": [
                ("What does {src} require?",
                 "{src_label} requires {tgt_label}. {reason}"),
                ("What are the prerequisites for {src}?",
                 "A key prerequisite for {src_label} is {tgt_label}. {reason}"),
            ],
            "enhances": [
                ("What enhances {tgt}?",
                 "{src_label} enhances {tgt_label}. {reason}"),
                ("How can {tgt} be strengthened?",
                 "{tgt_label} is strengthened when {src_label} is present. {reason}"),
            ],
            "invalidates": [
                ("What invalidates {tgt}?",
                 "{src_label} invalidates {tgt_label}. {reason}"),
                ("When should you avoid trading {tgt}?",
                 "You should avoid {tgt_label} when {src_label} occurs. {reason}"),
            ],
            "precedes": [
                ("What must happen before {tgt}?",
                 "{src_label} must precede {tgt_label} in the ICT sequence."),
                ("What comes after {src} in the sequence?",
                 "After {src_label}, the next step is {tgt_label}."),
            ],
            "is_type_of": [
                ("What type of concept is {src}?",
                 "{src_label} is a type of {tgt_label} in the ICT framework."),
            ],
            "belongs_to": [
                ("What category does {src} belong to?",
                 "{src_label} belongs to the {tgt_label} category."),
            ],
            # ── Expanded relation templates (matches cleaned graph predicates) ──
            "is_a": [
                ("What is {src} in the ICT framework?",
                 "{src_label} is a {tgt_label} in ICT trading methodology."),
                ("How would you classify {src}?",
                 "{src_label} is classified as a {tgt_label}."),
            ],
            "includes": [
                ("What does {src} include?",
                 "{src_label} includes {tgt_label} as a key component."),
                ("What are the components of {src}?",
                 "One of the components of {src_label} is {tgt_label}."),
            ],
            "contains": [
                ("What does {src} contain?",
                 "{src_label} contains {tgt_label}."),
            ],
            "has_component": [
                ("What are the components of {src}?",
                 "{src_label} has {tgt_label} as a component."),
                ("What makes up {src}?",
                 "{tgt_label} is a component of {src_label}."),
            ],
            "uses": [
                ("What does {src} use?",
                 "{src_label} uses {tgt_label} in its methodology."),
                ("How is {tgt} applied?",
                 "{tgt_label} is used by {src_label} as part of the ICT approach."),
            ],
            "shows": [
                ("What does {src} show?",
                 "{src_label} shows {tgt_label}."),
            ],
            "occurs": [
                ("When does {src} occur?",
                 "{src_label} occurs with {tgt_label}."),
            ],
            "indicates": [
                ("What does {src} indicate?",
                 "{src_label} indicates {tgt_label} in the market structure."),
                ("How can you identify {tgt}?",
                 "{tgt_label} is indicated by {src_label}."),
            ],
            "relates_to": [
                ("How does {src} relate to {tgt}?",
                 "{src_label} relates to {tgt_label} in ICT methodology."),
            ],
            "related_to": [
                ("What is {src} related to?",
                 "{src_label} is related to {tgt_label} in the ICT framework."),
            ],
            "consists_of": [
                ("What does {src} consist of?",
                 "{src_label} consists of {tgt_label}."),
            ],
            "focuses_on": [
                ("What does {src} focus on?",
                 "{src_label} focuses on {tgt_label}."),
            ],
            "provides": [
                ("What does {src} provide?",
                 "{src_label} provides {tgt_label} in the trading context."),
            ],
            "represents": [
                ("What does {src} represent?",
                 "{src_label} represents {tgt_label} in ICT methodology."),
            ],
            "involves": [
                ("What does {src} involve?",
                 "{src_label} involves {tgt_label}."),
            ],
            "identifies": [
                ("What does {src} identify?",
                 "{src_label} identifies {tgt_label}."),
            ],
            "targets": [
                ("What does {src} target?",
                 "{src_label} targets {tgt_label}."),
            ],
            "follows": [
                ("What follows {src}?",
                 "{tgt_label} follows {src_label} in the sequence."),
            ],
            "results_in": [
                ("What does {src} result in?",
                 "{src_label} results in {tgt_label}."),
            ],
            "confirms": [
                ("What does {src} confirm?",
                 "{src_label} confirms {tgt_label}."),
                ("How is {tgt} confirmed?",
                 "{tgt_label} is confirmed by {src_label}."),
            ],
            "leads_to": [
                ("What does {src} lead to?",
                 "{src_label} leads to {tgt_label} in the ICT setup flow."),
            ],
            "aligns_with": [
                ("What does {src} align with?",
                 "{src_label} aligns with {tgt_label}, providing confluence."),
            ],
            "defines": [
                ("How is {tgt} defined?",
                 "{tgt_label} is defined by {src_label}."),
            ],
            "creates": [
                ("What does {src} create?",
                 "{src_label} creates {tgt_label}."),
            ],
            "triggers": [
                ("What triggers {tgt}?",
                 "{src_label} triggers {tgt_label}."),
            ],
            "exists_in": [
                ("Where does {src} exist?",
                 "{src_label} exists in {tgt_label}."),
            ],
            "describes": [
                ("What does {src} describe?",
                 "{src_label} describes {tgt_label}."),
            ],
            "refers_to": [
                ("What does {src} refer to?",
                 "{src_label} refers to {tgt_label}."),
            ],
            "based_on": [
                ("What is {src} based on?",
                 "{src_label} is based on {tgt_label}."),
            ],
            "occurs_at": [
                ("Where does {src} occur?",
                 "{src_label} occurs at {tgt_label}."),
            ],
            "is_part_of": [
                ("What is {src} part of?",
                 "{src_label} is part of {tgt_label}."),
            ],
            "supports": [
                ("What does {src} support?",
                 "{src_label} supports {tgt_label}."),
            ],
            "violates": [
                ("What does {src} violate?",
                 "{src_label} violates {tgt_label}. This is a critical error that should prevent trade entry."),
            ],
        }

        # Catch-all for any relation type not explicitly templated
        _catch_all = [
            ("How does {src} relate to {tgt} in ICT?",
             "In ICT methodology, {src_label} {rel_natural} {tgt_label}."),
            ("What is the connection between {src} and {tgt}?",
             "{src_label} {rel_natural} {tgt_label} within the ICT framework."),
        ]

        # Noise predicates from transcript extraction — skip these
        _noise_words = {
            "we", "we're", "you", "they", "i", "he", "she", "it",
            "don't", "doesn't", "can't", "won't", "wouldn't", "couldn't",
            "should", "would", "could", "will", "shall", "might",
            "what", "where", "when", "how", "why", "who",
            "look", "looking", "looked", "see", "seen", "saw",
            "going", "doing", "having", "being", "getting",
        }

        def _is_clean_predicate(pred: str) -> bool:
            """Return True if predicate looks like a real relationship type."""
            parts = pred.split("_")
            # Too many words → likely a sentence fragment
            if len(parts) > 4:
                return False
            # Contains noise/conversational words
            if any(w.lower() in _noise_words for w in parts):
                return False
            # Contains emoji or special chars
            if any(not c.isalnum() and c != "_" for c in pred):
                return False
            # Very short single-char predicates
            if len(pred) < 3:
                return False
            return True

        MAX_PER_REL = 30  # cap per relation type to avoid imbalance
        MAX_PER_CATCHALL = 5  # lower cap for catch-all (less trustworthy)
        MIN_EDGES_CATCHALL = 3  # only use catch-all if rel has enough edges

        for rel_type, edges in by_relation.items():
            if rel_type in templates:
                tmpls = templates[rel_type]
                cap = MAX_PER_REL
            elif _is_clean_predicate(rel_type) and len(edges) >= MIN_EDGES_CATCHALL:
                tmpls = _catch_all
                cap = MAX_PER_CATCHALL
            else:
                continue  # skip noise predicates

            # Sample if too many edges for this relation
            if len(edges) > cap:
                edges = list(self.rng.sample(edges, cap))

            for src, tgt, data in edges:
                src_label = src.replace("_", " ").title()
                tgt_label = tgt.replace("_", " ").title()
                reason = data.get("reason", "")
                if reason:
                    reason = f"Reason: {reason}"

                # Pick a random template for variety
                rel_natural = rel_type.replace("_", " ")
                tmpl = self.rng.choice(tmpls)
                question = tmpl[0].format(src=src_label, tgt=tgt_label, rel_natural=rel_natural)
                answer = tmpl[1].format(
                    src_label=src_label, tgt_label=tgt_label, reason=reason,
                    rel_natural=rel_natural,
                ).strip()

                self._add(
                    instruction=question,
                    output=answer,
                    category=f"relationship_{rel_type}",
                )
                count += 1

        print(f"[training_gen] Generated {count} relationship Q&A examples")
        return count

    # ── Type 3: Multi-hop reasoning chains ────────────────────────────────

    def generate_reasoning_chains(self) -> int:
        """Generate chain-of-thought reasoning from graph paths.
        
        Example: liquidity_sweep → displacement → fvg → entry
        Becomes: "Walk me through a reversal entry..."
        """
        count = 0
        rels = self._load_concept_rels()

        # Generate from causal chains in concept_relationships.yaml
        for chain_name, chain in (rels.get("causal_chains") or {}).items():
            steps = chain.get("steps", {})
            desc = chain.get("description", chain_name)
            sorted_steps = sorted(steps.items(), key=lambda x: int(x[0]))

            if len(sorted_steps) < 3:
                continue

            # Build the reasoning chain
            step_texts = []
            for i, (num, step) in enumerate(sorted_steps, 1):
                concept = step.get("concept", step.get("phase", f"step {i}"))
                signal = step.get("signal", step.get("action", ""))
                step_texts.append(f"Step {i}: {concept.replace('_', ' ').title()} — {signal}")

            chain_text = "\n".join(step_texts)

            # Forward reasoning
            self._add(
                instruction=f"Walk me through the {desc} step by step.",
                output=f"The {desc} follows this sequence:\n\n{chain_text}\n\n"
                       f"Each step must complete before moving to the next. "
                       f"Skipping any step significantly reduces the probability of success.",
                category="reasoning_chain",
            )
            count += 1

            # Failure analysis for each step
            failure_info = chain.get("failure_at_step", {})
            for step_num, failure in failure_info.items():
                step_data = steps.get(str(step_num), steps.get(int(step_num), {}))
                concept = step_data.get("concept", step_data.get("phase", f"step {step_num}"))

                self._add(
                    instruction=f"What happens if step {step_num} ({concept.replace('_', ' ')}) fails in the {desc}?",
                    output=f"If {concept.replace('_', ' ')} fails: {failure}\n\n"
                           f"This breaks the causal chain. Without this step completing, "
                           f"subsequent steps lack the prerequisite confirmation and the "
                           f"setup should be abandoned.",
                    category="reasoning_failure",
                )
                count += 1

        # Generate from graph paths between important concept pairs
        important_pairs = [
            ("liquidity_sweep", "silver_bullet"),
            ("liquidity_sweep", "entry"),
            ("accumulation", "distribution"),
            ("displacement", "fair_value_gap"),
            ("order_block", "entry"),
            ("htf_bias", "ltf_execution"),
            ("asian_range", "judas_swing"),
            ("cbdr", "silver_bullet"),
            ("smt_divergence", "entry"),
            ("equal_highs", "liquidity_sweep"),
        ]

        for src, tgt in important_pairs:
            path = self.store.find_path_with_relations(src, tgt)
            if len(path) < 2:
                continue

            path_text = " → ".join(
                f"{s['from'].replace('_', ' ')} --[{s['relation']}]--> {s['to'].replace('_', ' ')}"
                for s in path
            )

            src_label = src.replace("_", " ").title()
            tgt_label = tgt.replace("_", " ").title()

            self._add(
                instruction=f"How does {src_label} connect to {tgt_label} in ICT methodology?",
                output=f"The relationship chain from {src_label} to {tgt_label} is:\n\n"
                       f"{path_text}\n\n"
                       f"This shows how {src_label} ultimately leads to or enables {tgt_label} "
                       f"through a series of causal and prerequisite relationships.",
                category="reasoning_path",
            )
            count += 1

        print(f"[training_gen] Generated {count} reasoning chain examples")
        return count

    # ── Type 4: Model selection reasoning ─────────────────────────────────

    def generate_model_selection(self) -> int:
        """Generate examples of selecting the right ICT model given detected patterns."""
        count = 0
        rels = self._load_concept_rels()
        models = rels.get("models", {})

        for model_name, model_data in models.items():
            model_label = model_name.replace("_", " ").title()
            required = model_data.get("required", [])
            desc = model_data.get("description", "")
            time_windows = model_data.get("time_windows", [])

            if not required:
                continue

            # "When to use this model" question
            req_text = "\n".join(f"  - {r}" for r in required)
            time_text = ""
            if time_windows:
                time_text = "\n\nTime windows:\n" + "\n".join(
                    f"  - {tw.get('name', 'Window')}: {tw.get('start', '')} - {tw.get('end', '')} {tw.get('timezone', '')}"
                    for tw in time_windows
                )

            self._add(
                instruction=f"When should I use the {model_label} model?",
                output=f"The {model_label} model{' — ' + desc if desc else ''} requires:\n\n"
                       f"{req_text}{time_text}\n\n"
                       f"All conditions must be met. If any requirement is missing, "
                       f"do not force the setup.",
                category="model_requirements",
            )
            count += 1

            # "I see these patterns, what model?" question
            if len(required) >= 2:
                subset = required[:3]
                scenario = ", ".join(subset)
                self._add(
                    instruction=f"I'm seeing these market conditions: {scenario}. Which ICT model applies?",
                    output=f"Based on these conditions, the {model_label} model is the best fit.\n\n"
                           f"This model specifically requires:\n{req_text}\n\n"
                           f"Make sure all remaining requirements are also confirmed before entering.",
                    category="model_selection",
                )
                count += 1

            # Avoid conditions
            avoid = model_data.get("avoid_when", [])
            if avoid:
                avoid_text = "\n".join(f"  - {a}" for a in avoid)
                self._add(
                    instruction=f"When should I avoid the {model_label} model?",
                    output=f"Avoid the {model_label} when:\n\n{avoid_text}\n\n"
                           f"These conditions reduce the win rate significantly.",
                    category="model_avoidance",
                )
                count += 1

        print(f"[training_gen] Generated {count} model selection examples")
        return count

    # ── Type 5: Trade analysis from real examples ─────────────────────────

    def generate_trade_analysis(self) -> int:
        """Annotate existing positive trade examples with graph context."""
        count = 0

        for trade_file in sorted(POSITIVE_DIR.glob("*.json")):
            try:
                with open(trade_file, "r") as f:
                    trade = json.load(f)
            except Exception:
                continue

            trade_id = trade.get("id", trade_file.stem)
            pair = trade.get("market", {}).get("pair", "Unknown")
            session = trade.get("time", {}).get("session", "Unknown")
            htf_bias = trade.get("context", {}).get("htf_bias", "Unknown")
            env_notes = trade.get("context", {}).get("environment_notes", "")

            # Extract ICT elements mentioned
            pd_arrays = trade.get("pd_arrays", {})
            elements = []
            if pd_arrays.get("htf_order_blocks"):
                elements.append("HTF Order Block")
            if pd_arrays.get("ltf_order_blocks"):
                elements.append("LTF Order Block")
            if pd_arrays.get("fair_value_gaps"):
                elements.append("Fair Value Gap")

            # Get graph context for these elements
            graph_annotations = []
            for elem in elements:
                concept = _normalize(elem)
                related = self.store.get_related_concepts(concept, max_hops=1)
                if related:
                    for rel_type, concepts in list(related.items())[:3]:
                        graph_annotations.append(
                            f"  - {elem} {rel_type}: {', '.join(concepts[:3])}"
                        )

            elements_text = ", ".join(elements) if elements else "Standard setup"
            graph_text = "\n".join(graph_annotations) if graph_annotations else "  (No additional graph context)"

            self._add(
                instruction=f"Analyze this ICT trade setup: {pair} {session} session with {htf_bias} HTF bias.",
                input_text=env_notes or f"Trade {trade_id} on {pair} during {session}.",
                output=f"Trade Analysis ({trade_id}):\n\n"
                       f"Pair: {pair}\n"
                       f"Session: {session}\n"
                       f"HTF Bias: {htf_bias}\n"
                       f"ICT Elements Present: {elements_text}\n\n"
                       f"Graph-Derived Context:\n{graph_text}\n\n"
                       f"This trade used confirmed ICT structure with proper "
                       f"prerequisite alignment. The presence of {elements_text} "
                       f"provided sufficient confluence for entry.",
                category="trade_analysis_positive",
            )
            count += 1

        print(f"[training_gen] Generated {count} trade analysis examples")
        return count

    # ── Type 6: Counterfactual analysis from negative examples ────────────

    def generate_counterfactuals(self) -> int:
        """Generate 'why did this fail?' reasoning from negative trade examples."""
        count = 0

        for trade_file in sorted(NEGATIVE_DIR.glob("*.json")):
            try:
                with open(trade_file, "r") as f:
                    trade = json.load(f)
            except Exception:
                continue

            trade_id = trade.get("id", trade_file.stem)
            pair = trade.get("market", {}).get("pair", "Unknown")
            env_notes = trade.get("context", {}).get("environment_notes", "")
            tags = trade.get("meta", {}).get("tags", [])

            # Identify what went wrong using tags
            failure_reasons = []
            fix_suggestions = []

            for tag in tags:
                tag_norm = _normalize(tag)
                # Check if this tag corresponds to an anti-pattern in the graph
                if tag_norm in self.store.G:
                    defn = self.store.get_concept_definition(tag_norm)
                    if defn:
                        failure_reasons.append(f"- {tag}: {defn}")
                    neighbors = self.store.get_neighbors(tag_norm, edge_type="violates")
                    for n in neighbors:
                        fix_suggestions.append(f"- Follow proper {n['node'].replace('_', ' ')} protocol")

            if not failure_reasons:
                failure_reasons.append(f"- Tags indicate: {', '.join(tags)}")
                fix_suggestions.append("- Review pre-trade checklist before entering")

            failures_text = "\n".join(failure_reasons)
            fixes_text = "\n".join(fix_suggestions) if fix_suggestions else "- Apply stricter entry criteria"

            self._add(
                instruction=f"Why did this {pair} trade fail?",
                input_text=env_notes or f"Trade {trade_id}: {', '.join(tags)}",
                output=f"Trade Failure Analysis ({trade_id}):\n\n"
                       f"Issues identified:\n{failures_text}\n\n"
                       f"What to do differently:\n{fixes_text}\n\n"
                       f"The graph-based analysis shows this trade violated key "
                       f"prerequisites in the ICT causal chain. Following the proper "
                       f"sequence would have either produced a valid entry or correctly "
                       f"kept you on the sidelines.",
                category="trade_analysis_negative",
            )
            count += 1

        print(f"[training_gen] Generated {count} counterfactual examples")
        return count

    # ── Type 7: Pre-trade validation reasoning ────────────────────────────

    def generate_pretrade_validation(self) -> int:
        """Generate pre-trade checklist reasoning examples."""
        count = 0
        rels = self._load_concept_rels()
        validation = rels.get("pre_trade_validation", {})

        if not validation:
            return 0

        must_have_one = validation.get("must_have_one", [])
        must_have_all = validation.get("must_have_all", [])
        should_have = validation.get("should_have", [])
        red_flags = validation.get("red_flags", [])

        # Generate "should I take this trade?" scenarios
        # Scenario 1: All conditions met
        if must_have_all:
            conditions = [c.replace("_", " ") for c in must_have_all]
            self._add(
                instruction="Should I take this trade? I have: " + ", ".join(conditions) + ".",
                output="Yes, this trade meets the minimum requirements.\n\n"
                       f"Must-have conditions confirmed: {', '.join(conditions)}\n\n"
                       "However, also verify:\n"
                       f"- At least one of: {', '.join(c.replace('_', ' ') for c in must_have_one)}\n"
                       f"- Ideally also: {', '.join(c.replace('_', ' ') for c in should_have)}\n\n"
                       f"Red flags to check: {', '.join(c.replace('_', ' ') for c in red_flags[:3])}",
                category="pretrade_validation",
            )
            count += 1

        # Scenario 2: Missing critical conditions
        for flag in red_flags[:5]:
            flag_label = flag.replace("_", " ")
            self._add(
                instruction=f"Should I take this trade? The setup looks good but I notice {flag_label}.",
                output=f"No. {flag_label.title()} is a red flag.\n\n"
                       f"Red flags are absolute disqualifiers — no matter how good the rest of "
                       f"the setup looks. The purpose of the pre-trade checklist is to protect "
                       f"capital on days when discipline matters most.\n\n"
                       f"Wait for the red flag to clear, or look for the next setup.",
                category="pretrade_red_flag",
            )
            count += 1

        # Scenario 3: Confluence scoring
        weights = rels.get("confluence_weights", {})
        thresholds = weights.get("thresholds", {})
        if thresholds:
            min_score = thresholds.get("minimum_for_trade", 5.0)
            good_score = thresholds.get("good_setup", 7.0)
            aplus_score = thresholds.get("a_plus_setup", 9.0)

            self._add(
                instruction="How do I score a setup's confluence in ICT methodology?",
                output=f"ICT confluence scoring uses weighted factors:\n\n"
                       f"Critical factors (2.0-2.5 weight): liquidity swept, HTF bias aligned, displacement confirmed\n"
                       f"High value (1.5): in FVG, at order block, SMT divergence, in killzone\n"
                       f"Moderate (1.0): at OTE level, structure break, targeting liquidity\n"
                       f"Bonuses: unicorn setup (+2.0), HTF POI alignment (+1.5)\n"
                       f"Penalties: against HTF bias (-2.0), no liquidity sweep (-2.0)\n\n"
                       f"Thresholds:\n"
                       f"  Minimum for trade: {min_score}\n"
                       f"  Good setup: {good_score}\n"
                       f"  A+ setup: {aplus_score}\n\n"
                       f"Never trade below {min_score}. Be selective.",
                category="confluence_scoring",
            )
            count += 1

        print(f"[training_gen] Generated {count} pre-trade validation examples")
        return count

    # ── Type 8: Anti-pattern recognition ──────────────────────────────────

    def generate_anti_pattern_qa(self) -> int:
        """Generate Q&A about common trading mistakes from graph anti-patterns."""
        count = 0

        anti_pattern_nodes = self.store.get_nodes_by_type("anti_pattern")
        for ap_node in anti_pattern_nodes:
            data = self.store.G.nodes[ap_node]
            desc = data.get("description", "")
            why_fails = data.get("why_fails", "")
            fix = data.get("fix", "")

            if not desc:
                continue

            ap_label = ap_node.replace("_", " ").title()

            self._add(
                instruction=f"What is the '{ap_label}' anti-pattern in ICT trading?",
                output=f"**{ap_label}** is a common trading mistake.\n\n"
                       f"What it looks like: {desc}\n"
                       f"Why it fails: {why_fails}\n"
                       f"How to fix it: {fix}",
                category="anti_pattern",
            )
            count += 1

        print(f"[training_gen] Generated {count} anti-pattern Q&A examples")
        return count

    # ── Type 9: Import existing Q&A ───────────────────────────────────────

    def generate_existing_qa(self) -> int:
        """Import existing Q&A datasets from train-ict/data/training/qa/."""
        count = 0

        for qa_file in sorted(QA_DIR.glob("*.json")):
            try:
                with open(qa_file, "r") as f:
                    qa_data = json.load(f)
            except Exception:
                continue

            if not isinstance(qa_data, list):
                continue

            for item in qa_data:
                question = item.get("question", "")
                explanation = item.get("explanation", "")
                correct = item.get("correct_answer", "")
                options = item.get("options", {})

                if not question or not explanation:
                    continue

                # Format with options for richer training
                options_text = ""
                if options:
                    options_text = "\n" + "\n".join(
                        f"  {k}) {v}" for k, v in options.items()
                    )

                correct_text = ""
                if correct and options:
                    correct_text = f"\n\nCorrect answer: {correct}) {options.get(correct, '')}"

                self._add(
                    instruction=question,
                    input_text=options_text if options_text else None,
                    output=f"{explanation}{correct_text}",
                    category="existing_qa",
                )
                count += 1

        print(f"[training_gen] Generated {count} existing Q&A examples")
        return count

    # ── Type 10: Neighborhood summaries ───────────────────────────────────

    def generate_neighborhood_summaries(self) -> int:
        """Generate comprehensive summaries of a concept's graph neighborhood.

        For well-connected nodes, produce a rich description that covers
        incoming and outgoing relationships to teach the model about the
        concept's role in the broader ICT framework.
        """
        count = 0

        # Only nodes with enough connections to produce a useful summary
        MIN_DEGREE = 4
        MAX_EXAMPLES = 500  # cap to avoid overwhelm

        candidates = [
            (node, self.store.G.degree(node))
            for node in self.store.G.nodes()
            if self.store.G.degree(node) >= MIN_DEGREE
        ]
        # Sort by degree (most connected first)
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:MAX_EXAMPLES]

        for node, degree in candidates:
            node_label = node.replace("_", " ").title()

            # Collect outgoing
            out_by_rel: dict[str, list[str]] = defaultdict(list)
            for _, tgt, edata in self.store.G.out_edges(node, data=True):
                rel = edata.get("relation", "related_to").replace("_", " ")
                out_by_rel[rel].append(tgt.replace("_", " ").title())

            # Collect incoming
            in_by_rel: dict[str, list[str]] = defaultdict(list)
            for src, _, edata in self.store.G.in_edges(node, data=True):
                rel = edata.get("relation", "related_to").replace("_", " ")
                in_by_rel[rel].append(src.replace("_", " ").title())

            # Build summary text
            parts = []
            if out_by_rel:
                for rel, targets in sorted(out_by_rel.items()):
                    targets_str = ", ".join(targets[:5])
                    if len(targets) > 5:
                        targets_str += f" (and {len(targets) - 5} more)"
                    parts.append(f"- {rel}: {targets_str}")

            if in_by_rel:
                for rel, sources in sorted(in_by_rel.items()):
                    sources_str = ", ".join(sources[:5])
                    if len(sources) > 5:
                        sources_str += f" (and {len(sources) - 5} more)"
                    parts.append(f"- Is {rel} by: {sources_str}")

            if not parts:
                continue

            summary = "\n".join(parts)

            self._add(
                instruction=f"Describe the role of {node_label} in ICT trading and its key relationships.",
                output=f"{node_label} plays a significant role in the ICT framework "
                       f"(connected to {degree} concepts).\n\n"
                       f"Key relationships:\n{summary}",
                category="neighborhood_summary",
            )
            count += 1

        print(f"[training_gen] Generated {count} neighborhood summary examples")
        return count

    # ── Type 11: Concept comparisons ──────────────────────────────────────

    def generate_concept_comparisons(self) -> int:
        """Generate comparison Q&A between related ICT concepts.

        Picks pairs of nodes that share a common neighbor and asks
        the model to compare/contrast them.
        """
        count = 0
        MAX_EXAMPLES = 200

        # Find pairs that share neighbors
        seen_pairs: set[tuple[str, str]] = set()
        comparisons = []

        # Get nodes with enough connections
        connected_nodes = [
            n for n in self.store.G.nodes()
            if self.store.G.degree(n) >= 3
        ]

        for node in connected_nodes:
            neighbors_out = set(tgt for _, tgt in self.store.G.out_edges(node))
            neighbors_in = set(src for src, _ in self.store.G.in_edges(node))
            all_neighbors = neighbors_out | neighbors_in

            # For each pair of neighbors, they share `node` as a common connection
            neighbor_list = list(all_neighbors)
            for i in range(min(len(neighbor_list), 10)):
                for j in range(i + 1, min(len(neighbor_list), 10)):
                    a, b = neighbor_list[i], neighbor_list[j]
                    pair_key = tuple(sorted([a, b]))
                    if pair_key in seen_pairs:
                        continue
                    # Both neighbors must have minimum connectivity
                    if self.store.G.degree(a) < 2 or self.store.G.degree(b) < 2:
                        continue
                    seen_pairs.add(pair_key)
                    comparisons.append((a, b, node))

                    if len(comparisons) >= MAX_EXAMPLES * 3:
                        break
                if len(comparisons) >= MAX_EXAMPLES * 3:
                    break
            if len(comparisons) >= MAX_EXAMPLES * 3:
                break

        # Sample down
        self.rng.shuffle(comparisons)
        comparisons = comparisons[:MAX_EXAMPLES]

        for a, b, shared in comparisons:
            a_label = a.replace("_", " ").title()
            b_label = b.replace("_", " ").title()
            shared_label = shared.replace("_", " ").title()

            # Get unique relationships for each
            a_rels = set()
            for _, tgt, edata in self.store.G.out_edges(a, data=True):
                if tgt != b:
                    rel = edata.get("relation", "related_to").replace("_", " ")
                    a_rels.add(f"{rel} {tgt.replace('_', ' ').title()}")

            b_rels = set()
            for _, tgt, edata in self.store.G.out_edges(b, data=True):
                if tgt != a:
                    rel = edata.get("relation", "related_to").replace("_", " ")
                    b_rels.add(f"{rel} {tgt.replace('_', ' ').title()}")

            a_unique = list(a_rels - b_rels)[:3]
            b_unique = list(b_rels - a_rels)[:3]

            a_text = "\n".join(f"  - {r}" for r in a_unique) if a_unique else "  - (simpler concept)"
            b_text = "\n".join(f"  - {r}" for r in b_unique) if b_unique else "  - (simpler concept)"

            self._add(
                instruction=f"Compare {a_label} and {b_label} in ICT methodology.",
                output=f"Both {a_label} and {b_label} relate to {shared_label}, "
                       f"but they differ:\n\n"
                       f"{a_label} specifically:\n{a_text}\n\n"
                       f"{b_label} specifically:\n{b_text}\n\n"
                       f"Understanding these differences helps identify which concept "
                       f"applies in a given market context.",
                category="concept_comparison",
            )
            count += 1

        print(f"[training_gen] Generated {count} concept comparison examples")
        return count

    # ── Export ─────────────────────────────────────────────────────────────

    def export(self, filename: str = "ict_training_data.jsonl",
               fmt: str = "chatml",
               output_dir: Optional[Path] = None) -> Path:
        """Export training data to JSONL file.
        
        Args:
            filename: Output filename
            fmt: Format — "alpaca", "chatml", or "sharegpt"
            output_dir: Output directory (default: training_output/)
        
        Returns:
            Path to the output file
        """
        output_dir = output_dir or _OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Shuffle for training
        examples = list(self.examples)
        self.rng.shuffle(examples)

        with open(output_path, "w") as f:
            for ex in examples:
                if fmt == "alpaca":
                    record = {
                        "instruction": ex["instruction"],
                        "input": ex.get("input", ""),
                        "output": ex["output"],
                    }
                elif fmt == "chatml":
                    messages = [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": ex["instruction"]},
                    ]
                    if ex.get("input"):
                        messages[-1]["content"] += f"\n\n{ex['input']}"
                    messages.append({"role": "assistant", "content": ex["output"]})
                    record = {"messages": messages}
                elif fmt == "sharegpt":
                    convos = [
                        {"from": "system", "value": self.SYSTEM_PROMPT},
                        {"from": "human", "value": ex["instruction"] + (
                            f"\n\n{ex['input']}" if ex.get("input") else "")},
                        {"from": "gpt", "value": ex["output"]},
                    ]
                    record = {"conversations": convos}
                else:
                    raise ValueError(f"Unknown format: {fmt}")

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[training_gen] Exported {len(examples)} examples to {output_path} ({fmt} format)")

        # Also export stats
        stats = self.get_stats()
        stats_path = output_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[training_gen] Stats saved to {stats_path}")

        return output_path

    def export_train_test_split(self, test_ratio: float = 0.1,
                                fmt: str = "chatml",
                                output_dir: Optional[Path] = None) -> tuple[Path, Path]:
        """Export with train/test split."""
        output_dir = output_dir or _OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = list(self.examples)
        self.rng.shuffle(examples)

        split_idx = int(len(examples) * (1 - test_ratio))
        train = examples[:split_idx]
        test = examples[split_idx:]

        # Temporarily swap examples for export
        orig = self.examples

        self.examples = train
        train_path = self.export("ict_train.jsonl", fmt=fmt, output_dir=output_dir)

        self.examples = test
        test_path = self.export("ict_test.jsonl", fmt=fmt, output_dir=output_dir)

        self.examples = orig

        print(f"[training_gen] Split: {len(train)} train, {len(test)} test")
        return train_path, test_path

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return statistics about generated training data."""
        by_category: dict[str, int] = defaultdict(int)
        total_tokens_est = 0

        for ex in self.examples:
            by_category[ex.get("category", "unknown")] += 1
            # Rough token estimate: 1 token ≈ 4 chars
            text = ex["instruction"] + ex.get("input", "") + ex["output"]
            total_tokens_est += len(text) // 4

        return {
            "total_examples": len(self.examples),
            "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
            "estimated_tokens": total_tokens_est,
            "estimated_cost_gpt4_training": round(total_tokens_est * 0.00003, 2),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _add(self, instruction: str, output: str,
             category: str = "general",
             input_text: Optional[str] = None):
        """Add a training example."""
        self.examples.append({
            "instruction": instruction.strip(),
            "input": (input_text or "").strip(),
            "output": output.strip(),
            "category": category,
        })


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """CLI: generate training data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate ICT fine-tuning data")
    parser.add_argument("--format", "-f", default="chatml",
                        choices=["alpaca", "chatml", "sharegpt"])
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--split", action="store_true",
                        help="Generate train/test split")
    args = parser.parse_args()

    from .graph_store import ICTGraphStore
    store = ICTGraphStore().load_all()
    gen = TrainingDataGenerator(store)
    gen.generate_all()

    if args.split:
        gen.export_train_test_split(fmt=args.format,
                                    output_dir=Path(args.output) if args.output else None)
    else:
        gen.export(fmt=args.format,
                   output_dir=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
