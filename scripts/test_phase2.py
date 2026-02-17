#!/usr/bin/env python3
"""
Phase 2 Test: GraphRAG Retriever
=================================
Tests: ingest → vector search → graph expansion → full RAG query
Uses mxbai-embed-large for embeddings, llama3.2 for generation.
"""

import sys
import time
import shutil
from pathlib import Path

# Use the project source
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph_rag.graph_store import ICTGraphStore
from graph_rag.graph_retriever import GraphRAGRetriever

# Use a temp DB path so we don't pollute anything
TEST_DB = Path(__file__).parent / "_test_rag_db"


def cleanup():
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


def test_graph_query_no_llm():
    """Test graph-only query (no LLM or embeddings needed)."""
    print("\n" + "=" * 60)
    print("TEST 1: Graph-only query (no LLM)")
    print("=" * 60)

    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(store, db_path=TEST_DB)

    for concept in ["fair_value_gap", "silver_bullet", "liquidity_sweep", "order_block"]:
        result = retriever.graph_query(concept)
        print(f"\n--- {concept} ---")
        print(f"  Definition: {str(result['definition'])[:100]}...")
        print(f"  Requires: {result['requires'][:5]}")
        print(f"  Enhanced by: {result['enhanced_by'][:5]}")
        print(f"  Models: {result['models'][:5]}")
        print(f"  2-hop related: {len(result['2_hop_related'])} concepts")

    print("\n✅ Graph-only queries work!")
    return True


def test_ingest():
    """Test knowledge base ingestion into ChromaDB."""
    print("\n" + "=" * 60)
    print("TEST 2: Ingest knowledge base into ChromaDB")
    print("=" * 60)

    cleanup()  # Fresh start

    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(
        store,
        embedding_model="mxbai-embed-large",
        db_path=TEST_DB,
    )

    t0 = time.time()
    num_chunks = retriever.ingest_knowledge_base()
    elapsed = time.time() - t0

    print(f"\n  Chunks ingested: {num_chunks}")
    print(f"  Time: {elapsed:.1f}s")
    assert num_chunks > 0, "No chunks ingested!"
    print("✅ Ingestion works!")
    return num_chunks


def test_retrieve(num_chunks: int):
    """Test hybrid retrieval (vector + graph)."""
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid retrieval (vector search + graph expansion)")
    print("=" * 60)

    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(
        store,
        embedding_model="mxbai-embed-large",
        db_path=TEST_DB,
    )

    queries = [
        "What makes a valid Silver Bullet setup?",
        "How does liquidity sweep relate to displacement?",
        "What are the requirements for a Unicorn model?",
    ]

    for q in queries:
        print(f"\n--- Query: {q} ---")
        t0 = time.time()
        result = retriever.retrieve(q)
        elapsed = time.time() - t0

        print(f"  Chunks retrieved: {len(result['chunks'])}")
        print(f"  Graph entities found: {len(result['graph_entities'])}")
        print(f"    Sample entities: {result['graph_entities'][:8]}")
        print(f"  Graph triples: {len(result['graph_triples'])}")
        print(f"  Time: {elapsed:.1f}s")

        # Show first chunk
        if result["chunks"]:
            c = result["chunks"][0]
            print(f"  Top chunk (score={c['score']:.3f}): {c['source']}")
            print(f"    Preview: {c['text'][:120]}...")

    print("\n✅ Hybrid retrieval works!")
    return True


def test_full_rag_query():
    """Test full RAG pipeline: retrieve + LLM generation."""
    print("\n" + "=" * 60)
    print("TEST 4: Full RAG query (retrieve + LLM answer)")
    print("=" * 60)

    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(
        store,
        embedding_model="mxbai-embed-large",
        llm_model="llama3.2",
        db_path=TEST_DB,
    )

    question = "What is the difference between a Silver Bullet and a Unicorn model, and when should I use each?"

    print(f"\n  Question: {question}")
    print("  Generating answer...", end="", flush=True)

    t0 = time.time()
    result = retriever.query(question)
    elapsed = time.time() - t0

    print(f" done ({elapsed:.1f}s)\n")
    print(f"  Answer:\n{result['answer']}\n")
    print(f"  Sources: {result['sources']}")
    print(f"  Graph entities: {result['graph_entities'][:10]}")
    print(f"\n✅ Full RAG pipeline works!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 2 TEST: GraphRAG Retriever")
    print("  Embedding model: mxbai-embed-large")
    print("  LLM model: llama3.2")
    print("=" * 60)

    try:
        # Test 1: Graph-only (no external deps)
        test_graph_query_no_llm()

        # Test 2: Ingest (needs Ollama embeddings)
        num_chunks = test_ingest()

        # Test 3: Retrieve (needs ChromaDB populated)
        test_retrieve(num_chunks)

        # Test 4: Full RAG (needs Ollama LLM)
        test_full_rag_query()

        print("\n" + "=" * 60)
        print("  ALL PHASE 2 TESTS PASSED ✅")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
