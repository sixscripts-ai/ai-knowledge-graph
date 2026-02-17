#!/usr/bin/env python3
"""Full health check after cleanup — verify nothing important was deleted."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

print("=" * 60)
print("HEALTH CHECK: ai-knowledge-graph")
print("=" * 60)

# 1. Graph data file
print("\n--- Graph Data ---")
try:
    with open("ict_graph_final.json") as f:
        data = json.load(f)
    triples = data if isinstance(data, list) else data.get("triples", data.get("results", []))
    check("ict_graph_final.json loads", True, f"{len(triples)} triples")
    check("Triples have subject/predicate/object", all(
        "subject" in t and "predicate" in t and "object" in t for t in triples
    ))
    empty = sum(1 for t in triples if not t.get("subject") or not t.get("object"))
    check("No empty subjects/objects", empty == 0, f"{empty} empty")
except Exception as e:
    check("ict_graph_final.json loads", False, str(e))

# 2. Training data
print("\n--- Training Data ---")
for name in ["finetune_data/train.jsonl", "finetune_data/valid.jsonl", "finetune_data/test.jsonl"]:
    try:
        count = 0
        bad = 0
        with open(name) as f:
            for line in f:
                count += 1
                d = json.loads(line)
                msgs = d.get("messages", [])
                roles = [m["role"] for m in msgs]
                if "user" not in roles or "assistant" not in roles:
                    bad += 1
        check(f"{name}", bad == 0, f"{count} examples, {bad} bad")
    except Exception as e:
        check(f"{name}", False, str(e))

# 3. Source module imports
print("\n--- Source Modules ---")
modules = [
    ("src.graph_rag.graph_store", "ICTGraphStore"),
    ("src.graph_rag.logic_engine", "TradeReasoner"),
    ("src.graph_rag.graph_retriever", "GraphRAGRetriever"),
    ("src.graph_rag.training_generator", "TrainingDataGenerator"),
    ("src.graph_rag.training_generator_v3", "TrainingDataGeneratorV3"),
    ("src.graph_rag.run_pipeline", None),
    ("src.knowledge_graph.config", "load_config"),
    ("src.knowledge_graph.main", "process_text_in_chunks"),
    ("src.knowledge_graph.visualization", "visualize_knowledge_graph"),
    ("src.knowledge_graph.llm", "call_llm"),
    ("src.knowledge_graph.text_utils", "chunk_text"),
    ("src.knowledge_graph.entity_standardization", "standardize_entities"),
]
for mod_path, cls_name in modules:
    try:
        mod = __import__(mod_path, fromlist=[cls_name or ""])
        if cls_name:
            getattr(mod, cls_name)
        check(f"import {mod_path}", True)
    except Exception as e:
        check(f"import {mod_path}", False, str(e))

# 4. Graph store loads all sources
print("\n--- Graph Store Integration ---")
try:
    from src.graph_rag.graph_store import ICTGraphStore
    store = ICTGraphStore()
    store.load_all()
    n = store.G.number_of_nodes()
    e = store.G.number_of_edges()
    check("GraphStore.load_all()", True, f"{n} nodes, {e} edges")
    check("Node count > 10000", n > 10000, str(n))
    check("Edge count > 20000", e > 20000, str(e))
except Exception as ex:
    check("GraphStore.load_all()", False, str(ex))

# 5. Logic engine
print("\n--- Logic Engine ---")
try:
    from src.graph_rag.logic_engine import TradeReasoner
    r = TradeReasoner(store)
    
    # A+ setup
    result = r.evaluate({
        "displacement": True, "htf_aligned": True, "htf_bias": "bullish",
        "liquidity_swept": True, "in_fvg": True, "at_order_block": True,
        "at_ote_level": True, "structure_break": True, "smt_divergence": True,
        "in_killzone": True
    })
    score = getattr(result, 'score', 0)
    go = getattr(result, 'go_no_go', False)
    check("A+ setup = GO", go is True, f"score={score}")
    
    # No displacement
    result2 = r.evaluate({
        "displacement": False, "htf_aligned": False, "in_killzone": False,
    })
    score2 = getattr(result2, 'score', 0)
    go2 = getattr(result2, 'go_no_go', True)
    check("No displacement = NO-GO", go2 is False, f"score={score2}")
except Exception as e:
    check("Logic engine", False, str(e))

# 6. Config files
print("\n--- Config Files ---")
for f in ["config.toml", "adapter_config_v2.json", "lora_v3_config.yaml", "pyproject.toml", "requirements.txt"]:
    check(f"{f} exists", os.path.isfile(f))

# 7. Scripts directory
print("\n--- Scripts ---")
expected_scripts = [
    "scripts/clean_graph.py",
    "scripts/run_training_gen.py",
    "scripts/train_with_early_stopping.py",
    "scripts/prepare_data.py",
    "scripts/finalize_graph.py",
    "scripts/preview_graph.py",
    "scripts/json_to_html.py",
    "scripts/setup_env.sh",
    "scripts/test_phase2.py",
]
for s in expected_scripts:
    check(f"{s} exists", os.path.isfile(s))

# 8. Tests directory
print("\n--- Tests ---")
check("tests/test_logic.py exists", os.path.isfile("tests/test_logic.py"))

# 9. Verify deleted files are truly gone
print("\n--- Deleted Files (should NOT exist) ---")
should_not_exist = [
    "adapters/", "adapters-v1/", "adapters_v3/",
    "vex-ict-fused/", "vex-ict.gguf", "vex-ict-v2.gguf",
    "ict_graph_checkpoint.json", "ict_graph_final.backup.json",
    "ict_graph_preview.json", "ict_graph_preview.html",
    "ict_knowledge_brain.html", "train_v2.log", "uv.lock",
    "training_output/", "training_output_v3/",
    ".DS_Store", "src/.DS_Store", "src/knowledge_graph/.DS_Store",
]
for f in should_not_exist:
    exists = os.path.exists(f)
    check(f"{f} removed", not exists, "STILL EXISTS!" if exists else "gone")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, p in results if p)
failed = sum(1 for _, p in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} checks")
if failed:
    print("\nFAILED CHECKS:")
    for name, p in results:
        if not p:
            print(f"  x {name}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
