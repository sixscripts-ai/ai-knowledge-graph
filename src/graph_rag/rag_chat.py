#!/usr/bin/env python3
"""
VEX — ICT GraphRAG Chat (MLX-Powered)
=======================================
Combines:
  - ICTGraphStore for knowledge graph retrieval (the facts)
  - Fine-tuned Llama 3.2 3B via MLX LoRA adapter (the voice)

The graph supplies accurate ICT definitions, relationships, and causal chains.
The fine-tuned model provides coherent, trading-aware generation.

Usage:
    # Interactive chat
    python -m src.graph_rag.rag_chat

    # Single question
    python -m src.graph_rag.rag_chat -q "What is a Fair Value Gap?"

    # Verbose (show graph context + token stats)
    python -m src.graph_rag.rag_chat -q "Explain the Silver Bullet" -v
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph_rag.graph_store import ICTGraphStore, _normalize
from src.graph_rag.logic_engine import TradeReasoner, TradeDecision

# ── Defaults ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = str(_PROJECT_ROOT / "vex-3b-ict-v7")  # fused v7 model (no adapter needed)
DEFAULT_ADAPTER = None  # set to adapter path if using base + LoRA
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMP = 0.7

# ICT terms for concept extraction (supplements graph node matching)
_ICT_ALIASES = {
    "fvg": "fair_value_gap",
    "ob": "order_block",
    "bos": "break_of_structure",
    "choch": "change_of_character",
    "mss": "market_structure_shift",
    "ote": "optimal_trade_entry",
    "htf": "higher_time_frame",
    "ltf": "lower_time_frame",
    "smt": "smt_divergence",
    "bsl": "buy_side_liquidity",
    "ssl": "sell_side_liquidity",
    "amd": "power_of_three",
    "mmsm": "market_maker_sell_model",
    "mmbm": "market_maker_buy_model",
    "pd array": "pd_array_matrix",
    "cbdr": "cbdr",
    "ict": "ict",
    "killzone": "killzone",
    "killzones": "killzone",
    # Gap-filler aliases (v4 concepts)
    "ifvg": "inversion_fair_value_gap",
    "bpr": "balanced_price_range",
    "vi": "volume_imbalance",
    "lv": "liquidity_void",
    "nwog": "new_week_opening_gap",
    "ndog": "new_day_opening_gap",
    "ce": "consequent_encroachment",
    "sibi": "sibi",
    "bisi": "bisi",
    "irb": "immediate_rebalance",
    "sd": "standard_deviation",
    "erl": "external_range_liquidity",
    "irl": "internal_range_liquidity",
    "dol": "draw_on_liquidity",
    "eqh": "equal_highs",
    "eql": "equal_lows",
    "lrlr": "low_resistance_liquidity_run",
    "hrlz": "high_resistance_liquidity_zone",
    "sfp": "swing_failure_pattern",
    "osok": "one_shot_one_kill",
    "ipda": "ipda",
    "smr": "smart_money_reversal",
    "cisd": "change_in_state_of_delivery",
    "adr": "average_daily_range",
}


class RAGChat:
    """Graph-augmented ICT chat using a fine-tuned MLX model."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        adapter_path: Optional[str] = DEFAULT_ADAPTER,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temp: float = DEFAULT_TEMP,
        max_context_triples: int = 40,
        memory_window: int = 6,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temp = temp
        self.max_context_triples = max_context_triples
        self.memory_window = memory_window  # how many past turns to keep

        self._model = None
        self._tokenizer = None
        self.store: Optional[ICTGraphStore] = None
        self.reasoner: Optional[TradeReasoner] = None
        self._history: list[dict] = []  # conversation memory

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self) -> "RAGChat":
        """Load the knowledge graph, logic engine, and MLX model."""
        # Graph
        print("Loading ICT Knowledge Graph...")
        self.store = ICTGraphStore()
        self.store.load_all()
        stats = self.store.stats()
        print(f"  {stats['nodes']} nodes, {stats['edges']} edges from {len(stats['sources'])} sources")

        # Logic Engine
        print("Loading Trade Reasoning Engine...")
        self.reasoner = TradeReasoner(self.store)
        print("  Ready.")

        # Model
        print(f"\nLoading model: {self.model_path}")
        if self.adapter_path:
            print(f"  Adapter: {Path(self.adapter_path).name}")
        from mlx_lm import load
        self._model, self._tokenizer = load(
            self.model_path,
            adapter_path=self.adapter_path,
        )
        print("  Ready.\n")
        return self

    # ── Concept Extraction ─────────────────────────────────────────────────

    def extract_concepts(self, text: str) -> list[str]:
        """Extract ICT concepts from text by matching graph nodes + aliases.

        Returns normalized concept names, ordered by relevance (best match first).
        """
        text_lower = text.lower()
        scored: dict[str, int] = defaultdict(int)

        # 1. Check aliases first (common abbreviations → canonical names)
        for alias, canonical in _ICT_ALIASES.items():
            if alias in text_lower:
                scored[canonical] += 20

        # 2. Graph search (fuzzy — matches against names + definitions)
        for result in self.store.search(text, top_k=15):
            scored[result["node"]] += result["score"]

        # 3. Direct node matching (exact — matches "fair_value_gap" or "fair value gap")
        for node in self.store.G.nodes():
            if len(node) < 3:
                continue
            # Exact match in text
            if node in text_lower or node.replace("_", " ") in text_lower:
                scored[node] += 15
            # Partial word matching for multi-word concepts
            parts = node.split("_")
            if len(parts) >= 2 and all(p in text_lower for p in parts if len(p) > 2):
                scored[node] += 5

        # Sort by score descending, filter noise
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        # Only keep concepts with meaningful scores
        return [c for c, s in ranked if s >= 10]

    # ── Graph Context Building ─────────────────────────────────────────────

    def build_graph_context(self, concepts: list[str]) -> str:
        """Build a rich, readable context string from graph data.

        Groups information by concept with definitions, key relationships,
        and model connections. Filters out noisy/duplicate triples.
        """
        # Relations that produce noise — skip these
        SKIP_RELATIONS = {
            "mentioned", "discussed", "located", "in", "is_a", "belongs_to",
            "related_to", "refers_to", "configuration", "source",
            "contributes_to", "described", "look_like",
        }

        sections = []
        seen_triples = set()
        triple_count = 0

        for concept in concepts[:6]:  # Cap concepts to keep prompt tight
            if triple_count >= self.max_context_triples:
                break

            display = concept.replace("_", " ").title()
            lines = []

            # Definition (highest value — always include)
            defn = self.store.get_concept_definition(concept)
            if defn and len(defn) > 5:
                lines.append(f"Definition: {defn}")
                triple_count += 1

            # Neighbors grouped by relation type
            neighbors = self.store.get_neighbors(concept)
            by_relation: dict[str, list[str]] = defaultdict(list)
            for n in neighbors:
                rel = n["relation"]
                # Skip noisy relations
                if rel in SKIP_RELATIONS:
                    continue
                # Skip long node names (usually sentence fragments, not concepts)
                if len(n["node"]) > 60:
                    continue

                triple_key = (concept, rel, n["node"])
                rev_key = (n["node"], rel, concept)
                if triple_key in seen_triples or rev_key in seen_triples:
                    continue
                seen_triples.add(triple_key)

                node_display = n["node"].replace("_", " ")
                if n["direction"] == "out":
                    by_relation[rel].append(node_display)
                else:
                    by_relation[f"is {rel} by"].append(node_display)

            # Format grouped relations (limit per-relation items)
            for rel, targets in by_relation.items():
                if triple_count >= self.max_context_triples:
                    break
                rel_display = rel.replace("_", " ")
                unique_targets = list(dict.fromkeys(targets))[:4]
                lines.append(f"  {rel_display}: {', '.join(unique_targets)}")
                triple_count += len(unique_targets)

            # Models that use this concept
            models = self.store.get_models_for_pattern(concept)
            if models:
                model_names = list({m["model"].replace("_", " ") for m in models})[:3]
                lines.append(f"  Used in models: {', '.join(model_names)}")
                triple_count += 1

            if lines:
                sections.append(f"**{display}**\n" + "\n".join(lines))

        return "\n\n".join(sections) if sections else "No specific graph context found."

    # ── Prompt Construction ────────────────────────────────────────────────

    def build_prompt(self, question: str, graph_context: str) -> str:
        """Build the Llama 3.2 Instruct chat prompt with RAG context and memory."""
        system = (
            "You are VEX, an expert ICT (Inner Circle Trader) trading assistant.\n"
            "Use the knowledge graph facts below to answer accurately.\n"
            "Synthesize the information into a clear, practical explanation.\n"
            "Do NOT list raw graph data — explain concepts in your own words.\n"
            "If a user asks you to evaluate a trade setup, analyze the confluences "
            "and give a clear go/no-go recommendation with reasoning.\n"
            "\n"
            "### ICT Facts\n"
            f"{graph_context}"
        )

        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
        )

        # Add conversation memory (last N turns)
        for turn in self._history[-self.memory_window:]:
            prompt += (
                f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n"
                f"{turn['content']}<|eot_id|>"
            )

        # Current question
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    # ── Trade Evaluation ───────────────────────────────────────────────────

    def evaluate_setup(self, text: str) -> dict:
        """Parse a natural-language trade setup description and evaluate it.

        Extracts patterns, session, bias from the text and runs the logic engine.
        Returns the TradeDecision plus a natural-language summary.
        """
        text_lower = text.lower()

        # Extract patterns from text
        pattern_keywords = {
            "fvg": ["fvg", "fair value gap"],
            "order_block": ["order block", "ob"],
            "displacement": ["displacement", "displaced"],
            "liquidity_sweep": ["liquidity sweep", "swept liquidity", "sweep",
                                "stop hunt", "took out", "grabbed liquidity"],
            "breaker_block": ["breaker block", "breaker"],
            "smt_divergence": ["smt", "divergence"],
            "turtle_soup": ["turtle soup"],
        }
        detected_patterns = []
        for pattern, keywords in pattern_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_patterns.append(pattern)

        # Extract session
        session = "ny_am"  # default
        session_map = {
            "london": ["london", "ldn"],
            "ny_am": ["ny am", "new york am", "ny morning", "morning session"],
            "ny_pm": ["ny pm", "afternoon", "ny afternoon"],
            "asian": ["asian", "asia"],
        }
        for sess, keywords in session_map.items():
            if any(kw in text_lower for kw in keywords):
                session = sess
                break

        # Extract bias
        htf_bias = "neutral"
        if any(w in text_lower for w in ["bullish", "bull", "long", "buy"]):
            htf_bias = "bullish"
        elif any(w in text_lower for w in ["bearish", "bear", "short", "sell"]):
            htf_bias = "bearish"

        # Extract additional boolean signals
        signals = {
            "patterns": detected_patterns,
            "htf_bias": htf_bias,
            "session": session,
            "displacement": "displacement" in text_lower or "displaced" in text_lower,
            "in_killzone": any(kz in text_lower for kz in
                              ["killzone", "kill zone", "london", "ny am", "ny pm"]),
            "smt_divergence": "smt" in text_lower or "divergence" in text_lower,
            "at_ote": "ote" in text_lower or "optimal trade entry" in text_lower,
            "liquidity_swept": any(kw in text_lower for kw in
                                   ["swept", "sweep", "stop hunt", "grabbed"]),
            "htf_aligned": htf_bias != "neutral",
        }

        decision = self.reasoner.evaluate(signals)

        return {
            "decision": decision,
            "signals_parsed": signals,
            "summary": decision.summary(),
        }

    # ── Generation ─────────────────────────────────────────────────────────

    def ask(self, question: str, verbose: bool = False) -> dict:
        """Ask a question with graph-augmented context and conversation memory.

        Returns:
            {
                "answer": str,
                "concepts": list[str],
                "graph_context": str,
                "elapsed": float,
                "prompt_tokens": int,
                "trade_eval": dict | None,  # present if trade evaluation detected
            }
        """
        t0 = time.time()

        # Check if this is a trade evaluation question
        trade_eval = None
        eval_keywords = [
            "evaluate", "is this a valid", "should i take", "rate this",
            "go or no-go", "confluence", "score this", "is this setup",
            "would you trade", "assess this",
        ]
        q_lower = question.lower()
        is_eval = any(kw in q_lower for kw in eval_keywords)

        if is_eval and self.reasoner:
            trade_eval = self.evaluate_setup(question)

        # Step 1: Extract concepts from the question (+ conversation context)
        context_text = question
        if self._history:
            # Include last user message for context continuity
            last_user = [t for t in self._history if t["role"] == "user"]
            if last_user:
                context_text = last_user[-1]["content"] + " " + question
        concepts = self.extract_concepts(context_text)

        # Step 2: Build graph context from those concepts
        graph_context = self.build_graph_context(concepts)

        # If we have a trade evaluation, prepend the logic engine results
        if trade_eval:
            eval_summary = trade_eval["summary"]
            graph_context = (
                f"### Trade Evaluation (Logic Engine)\n{eval_summary}\n\n"
                f"### Knowledge Graph Facts\n{graph_context}"
            )

        # Step 3: Build the full prompt (with memory)
        prompt = self.build_prompt(question, graph_context)

        # Step 4: Generate with MLX
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(self.temp)
        answer = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=verbose,
        )

        elapsed = time.time() - t0
        answer_text = answer.strip()

        # Store in conversation memory
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer_text})

        # Trim memory if beyond window
        max_entries = self.memory_window * 2  # pairs of user/assistant
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

        # Estimate prompt tokens (rough: ~4 chars per token)
        prompt_tokens = len(prompt) // 4

        return {
            "answer": answer_text,
            "concepts": concepts,
            "graph_context": graph_context,
            "elapsed": elapsed,
            "prompt_tokens": prompt_tokens,
            "trade_eval": trade_eval,
        }

    def clear_memory(self):
        """Clear conversation history."""
        self._history.clear()

    # ── Interactive Chat ───────────────────────────────────────────────────

    def chat(self):
        """Interactive chat loop with graph-augmented RAG."""
        print("=" * 60)
        print("  VEX — ICT GraphRAG Chat")
        print("  Knowledge graph + logic engine + fine-tuned ICT voice")
        print("=" * 60)
        print()
        print("Commands:")
        print("  /evaluate <setup>   — Evaluate a trade setup (go/no-go)")
        print("  /graph <concept>    — Browse graph data for a concept")
        print("  /concepts <text>    — Show extracted concepts")
        print("  /context <question> — Show graph context (no generation)")
        print("  /verbose            — Toggle verbose mode")
        print("  /stats              — Show graph statistics")
        print("  /memory             — Show conversation history")
        print("  /clear              — Clear conversation memory")
        print("  /model <name>       — Explain an ICT model's requirements")
        print("  exit                — Quit")
        print()

        verbose = False

        while True:
            try:
                q = input("You: ").strip()
                if not q:
                    continue

                if q.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break

                # ── Commands ──

                if q == "/verbose":
                    verbose = not verbose
                    print(f"  Verbose mode: {'ON' if verbose else 'OFF'}")
                    continue

                if q == "/clear":
                    self.clear_memory()
                    print("  Conversation memory cleared.")
                    continue

                if q == "/memory":
                    if not self._history:
                        print("  No conversation history.")
                    else:
                        print(f"  {len(self._history)} messages in memory:")
                        for i, msg in enumerate(self._history):
                            role = msg["role"].upper()
                            content = msg["content"][:80]
                            print(f"    [{i+1}] {role}: {content}...")
                    continue

                if q == "/stats":
                    stats = self.store.stats()
                    print(f"  Nodes: {stats['nodes']}")
                    print(f"  Edges: {stats['edges']}")
                    print(f"  Sources: {', '.join(stats['sources'])}")
                    top_rels = list(stats["relation_types"].items())[:10]
                    print(f"  Top relations: {', '.join(f'{r}({c})' for r, c in top_rels)}")
                    continue

                if q.startswith("/graph "):
                    concept = _normalize(q[7:].strip())
                    defn = self.store.get_concept_definition(concept)
                    neighbors = self.store.get_neighbors(concept)
                    print(f"\n  {concept.replace('_', ' ').title()}")
                    if defn:
                        print(f"  Definition: {defn}")
                    else:
                        print("  No definition found.")
                    print(f"  Connections: {len(neighbors)}")
                    for n in neighbors[:20]:
                        arrow = "→" if n["direction"] == "out" else "←"
                        print(f"    {arrow} {n['relation']}: {n['node']}")
                    print()
                    continue

                if q.startswith("/concepts "):
                    text = q[10:].strip()
                    concepts = self.extract_concepts(text)
                    print(f"  Found {len(concepts)} concepts: {', '.join(concepts[:20])}")
                    continue

                if q.startswith("/context "):
                    text = q[9:].strip()
                    concepts = self.extract_concepts(text)
                    context = self.build_graph_context(concepts)
                    print(f"\n  Concepts: {', '.join(concepts[:10])}")
                    print(f"\n  Graph Context:\n{context}\n")
                    continue

                if q.startswith("/evaluate "):
                    setup_text = q[10:].strip()
                    if not setup_text:
                        print("  Usage: /evaluate <describe your setup>")
                        continue
                    print("\n  Evaluating setup...")
                    eval_result = self.evaluate_setup(setup_text)
                    print(f"\n{eval_result['summary']}")
                    print(f"\n  Parsed signals: {eval_result['signals_parsed']['patterns']}")
                    print(f"  Session: {eval_result['signals_parsed']['session']}")
                    print(f"  HTF Bias: {eval_result['signals_parsed']['htf_bias']}")
                    print()
                    continue

                if q.startswith("/model "):
                    model_name = q[7:].strip()
                    info = self.reasoner.explain_model(model_name)
                    if "error" in info:
                        print(f"  {info['error']}")
                        # List available models
                        models = list(self.reasoner._model_blueprints.keys())
                        if models:
                            print(f"  Available models: {', '.join(models)}")
                    else:
                        print(f"\n  Model: {info['model']}")
                        if info['description']:
                            print(f"  Description: {info['description']}")
                        print(f"  Requirements:")
                        for req in info['required']:
                            print(f"    - {req}")
                        if info['time_windows']:
                            print(f"  Time windows:")
                            for tw in info['time_windows']:
                                print(f"    {tw.get('start','')} - {tw.get('end','')}")
                    print()
                    continue

                # ── RAG question ──
                result = self.ask(q, verbose=verbose)

                print(f"\nVEX: {result['answer']}")

                # Show trade evaluation if detected
                if result.get("trade_eval"):
                    d = result["trade_eval"]["decision"]
                    status = "✅ GO" if d.go_no_go else "❌ NO-GO"
                    print(f"\n  Logic Engine: {status} (score: {d.score:.1f})")
                    if d.recommendation:
                        print(f"  Recommended model: {d.recommendation}")
                    if d.red_flags:
                        print(f"  Red flags: {', '.join(d.red_flags)}")

                print(
                    f"\n  [{len(result['concepts'])} concepts | "
                    f"~{result['prompt_tokens']} prompt tokens | "
                    f"{result['elapsed']:.1f}s | "
                    f"{len(self._history)//2} turns in memory]"
                )
                if result["concepts"]:
                    print(f"  Concepts: {', '.join(result['concepts'][:8])}")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VEX — ICT GraphRAG Chat (MLX-Powered)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="MLX model path or HF repo"
    )
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter path"
    )
    parser.add_argument(
        "--no-adapter", action="store_true", help="Run without adapter (base model only)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max generation tokens"
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "-q", "--question", help="Single question (non-interactive mode)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show token stats and graph context"
    )
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter

    chat = RAGChat(
        model_path=args.model,
        adapter_path=adapter,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
    chat.load()

    if args.question:
        result = chat.ask(args.question, verbose=args.verbose)
        print(f"\nVEX: {result['answer']}")
        print(f"\n  Concepts: {', '.join(result['concepts'][:10])}")
        if args.verbose:
            print(f"\n  Graph Context:\n{result['graph_context']}")
    else:
        chat.chat()


if __name__ == "__main__":
    main()
