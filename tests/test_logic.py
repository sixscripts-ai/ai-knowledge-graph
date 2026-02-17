#!/usr/bin/env python3
"""Quick smoke test for logic engine after graph cleaning."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph_rag.graph_store import ICTGraphStore
from src.graph_rag.logic_engine import TradeReasoner

store = ICTGraphStore()
store.load_all()
r = TradeReasoner(store)

# Test 1: A+ setup with correct signal key names
d = r.evaluate({
    "displacement": True,
    "htf_aligned": True,
    "htf_bias": "bullish",
    "in_killzone": True,
    "liquidity_swept": True,
    "in_fvg": True,
    "at_order_block": True,
    "at_ote_level": True,
    "structure_break": True,
    "smt_divergence": True,
    "session": "ny_am",
})
go = "GO" if d.go_no_go else "NO-GO"
print(f"Test 1 — A+ setup: score={d.score}, {go}")
print(f"  Confluence: {d.confluence_factors}")
print(f"  Red flags: {d.red_flags}")
print(f"  Models: {d.model_scores}")
print()

# Test 2: No displacement
d2 = r.evaluate({
    "displacement": False,
    "session": "asian",
})
go2 = "GO" if d2.go_no_go else "NO-GO"
print(f"Test 2 — No displacement: score={d2.score}, {go2}")
print(f"  Red flags: {d2.red_flags}")
print()

# Test 3: Minimal valid setup
d3 = r.evaluate({
    "displacement": True,
    "htf_bias": "bearish",
    "in_killzone": True,
    "in_fvg": True,
    "session": "london",
})
go3 = "GO" if d3.go_no_go else "NO-GO"
print(f"Test 3 — Minimal valid: score={d3.score}, {go3}")
print(f"  Confluence: {d3.confluence_factors}")
