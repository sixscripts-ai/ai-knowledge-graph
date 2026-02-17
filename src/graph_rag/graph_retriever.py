"""
Phase 2: Graph-Enhanced RAG Retriever
=======================================
Hybrid retrieval: ChromaDB vector search + graph neighborhood expansion.

Instead of just returning the top-K similar chunks, this retriever:
1. Embeds the query → vector search → top-K chunks
2. Extracts entities from those chunks
3. Traverses the graph 1-2 hops from those entities
4. Pulls in related triples + definitions as additional context
5. Feeds the combined (chunks + graph context) to the LLM

Usage:
    from graph_rag import ICTGraphStore, GraphRAGRetriever

    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(store)

    # One-shot: ingest + query
    retriever.ingest_knowledge_base()
    result = retriever.query("What makes a valid Silver Bullet setup?")
    print(result["answer"])
    print(result["graph_context"])
    print(result["sources"])
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .graph_store import ICTGraphStore, _normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
_TRAIN_ICT = Path(
    os.environ.get("TRAIN_ICT_ROOT", Path.home() / "Documents" / "train-ict")
)

KNOWLEDGE_BASE_DIR = _TRAIN_ICT / "knowledge_base"
DATA_DIR = _TRAIN_ICT / "data"
RAG_DB_PATH = _TRAIN_ICT / "ai_rag" / "rag_db"


class GraphRAGRetriever:
    """Hybrid vector + graph retrieval for ICT knowledge."""

    def __init__(
        self,
        graph_store: ICTGraphStore,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "vex-ict",
        db_path: Optional[Path] = None,
        graph_hops: int = 2,
        vector_top_k: int = 5,
        graph_top_k: int = 15,
    ):
        self.store = graph_store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.db_path = str(db_path or RAG_DB_PATH)
        self.graph_hops = graph_hops
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k

        self._vectordb = None
        self._embeddings = None
        self._llm = None

    # ── Lazy initialization ────────────────────────────────────────────────

    def _init_embeddings(self):
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(model=self.embedding_model)
        return self._embeddings

    def _init_vectordb(self):
        if self._vectordb is None:
            from langchain_community.vectorstores import Chroma

            self._vectordb = Chroma(
                persist_directory=self.db_path,
                embedding_function=self._init_embeddings(),
            )
        return self._vectordb

    def _init_llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(
                model=self.llm_model,
                num_predict=512,  # Cap output length (vex-ict can be verbose)
                temperature=0.7,
            )
        return self._llm

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_knowledge_base(self, source_dirs: Optional[list[str]] = None) -> int:
        """Ingest markdown, JSON, YAML files from the knowledge base into ChromaDB.

        This replaces the old flat rag_ingest.py with graph-aware chunking:
        each chunk gets metadata about which graph concepts it mentions.
        """
        from langchain_community.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        source_dirs = source_dirs or [
            str(KNOWLEDGE_BASE_DIR),
            str(DATA_DIR / "schemas"),
            str(DATA_DIR / "training"),
            str(DATA_DIR / "playbooks"),
            str(DATA_DIR / "learning"),
        ]

        documents = []
        for dir_path in source_dirs:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                continue

            for ext in ("*.md", "*.yaml", "*.yml", "*.json", "*.txt"):
                for fpath in dir_path.rglob(ext):
                    if fpath.stat().st_size > 2 * 1024 * 1024:  # Skip >2MB
                        continue
                    try:
                        text = fpath.read_text(encoding="utf-8")
                    except Exception:
                        continue

                    # Extract graph concepts mentioned in this file
                    mentioned_concepts = self._extract_concepts(text)

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(fpath),
                                "filename": fpath.name,
                                "source_dir": fpath.parent.name,
                                "graph_concepts": ",".join(mentioned_concepts[:20]),
                            },
                        )
                    )

        if not documents:
            print("[graph_retriever] No documents found to ingest")
            return 0

        # Split with overlap — keep chunks short enough for embedding models
        # (mxbai-embed-large has 512 token limit ≈ ~2000 chars)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)

        # Enrich chunk metadata with graph concepts
        for chunk in chunks:
            concepts = self._extract_concepts(chunk.page_content)
            chunk.metadata["graph_concepts"] = ",".join(concepts[:10])

        # Filter out any chunks that are still too long (safety net)
        max_chars = 1800  # ~450 tokens, safe for 512-token models
        chunks = [c for c in chunks if len(c.page_content) <= max_chars]

        print(
            f"[graph_retriever] Ingesting {len(chunks)} chunks from {len(documents)} documents..."
        )
        embeddings = self._init_embeddings()

        # Batch ingestion to avoid single oversized request
        batch_size = 50
        from langchain_community.vectorstores import Chroma as _Chroma

        self._vectordb = None
        ingested = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                if self._vectordb is None:
                    self._vectordb = _Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=self.db_path,
                    )
                else:
                    self._vectordb.add_documents(batch)
                ingested += len(batch)
            except Exception as e:
                print(f"[graph_retriever] Warning: batch {i // batch_size} failed: {e}")
                # Try individual docs in the failed batch
                for doc in batch:
                    try:
                        if self._vectordb is None:
                            self._vectordb = _Chroma.from_documents(
                                documents=[doc],
                                embedding=embeddings,
                                persist_directory=self.db_path,
                            )
                        else:
                            self._vectordb.add_documents([doc])
                        ingested += 1
                    except Exception:
                        pass  # Skip this chunk

        print(
            f"[graph_retriever] Ingestion complete → {ingested}/{len(chunks)} chunks in ChromaDB"
        )
        return ingested

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> dict:
        """Hybrid retrieval: vector search + graph expansion.

        Returns:
            {
                "chunks": [...],           # Retrieved text chunks
                "graph_context": str,      # Graph-derived context
                "graph_entities": [...],   # Entities found in chunks
                "graph_triples": [...],    # Related triples from graph
                "combined_context": str,   # Full context for LLM
            }
        """
        db = self._init_vectordb()

        # Step 1: Vector search
        results = db.similarity_search_with_score(query, k=self.vector_top_k)

        chunks = []
        all_concepts = set()

        for doc, score in results:
            chunks.append(
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("filename", "unknown"),
                    "score": float(score),
                }
            )
            # Extract concepts from the chunk
            concepts = doc.metadata.get("graph_concepts", "").split(",")
            all_concepts.update(c for c in concepts if c)
            # Also extract from the text itself
            all_concepts.update(self._extract_concepts(doc.page_content))

        # Also extract concepts from the query itself
        query_concepts = self._extract_concepts(query)
        all_concepts.update(query_concepts)

        # Prioritize: query concepts first, then chunk concepts, cap total
        max_expand = 20  # Only expand the top-N most relevant concepts
        prioritized = list(query_concepts) + [
            c for c in all_concepts if c not in query_concepts
        ]
        expand_concepts = prioritized[:max_expand]

        # Step 2: Graph expansion (capped to avoid context explosion)
        graph_triples = []
        seen_triples = set()
        max_total_triples = 50  # Hard cap on total triples sent to LLM

        for concept in expand_concepts:
            if len(graph_triples) >= max_total_triples:
                break

            # Get neighborhood
            neighbors = self.store.get_neighbors(concept)
            for n in neighbors[: self.graph_top_k]:
                if len(graph_triples) >= max_total_triples:
                    break
                triple_key = (concept, n["relation"], n["node"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    graph_triples.append(
                        {
                            "subject": concept
                            if n["direction"] == "out"
                            else n["node"],
                            "predicate": n["relation"],
                            "object": n["node"] if n["direction"] == "out" else concept,
                        }
                    )

            # Get definition
            if len(graph_triples) < max_total_triples:
                defn = self.store.get_concept_definition(concept)
                if defn:
                    graph_triples.append(
                        {
                            "subject": concept,
                            "predicate": "defined_as",
                            "object": defn,
                        }
                    )

        # Step 3: Build graph context string
        graph_context = self._format_graph_context(graph_triples, all_concepts)

        # Step 4: Combine (cap total context to ~6000 chars for 3B model)
        chunk_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )
        combined = f"""## Retrieved Knowledge Base Content
{chunk_text}

## Graph-Derived Relationships
{graph_context}"""

        max_context_chars = 6000
        if len(combined) > max_context_chars:
            combined = combined[:max_context_chars] + "\n\n[context truncated]"

        return {
            "chunks": chunks,
            "graph_context": graph_context,
            "graph_entities": list(all_concepts),
            "graph_triples": graph_triples,
            "combined_context": combined,
        }

    def query(self, question: str, system_prompt: Optional[str] = None) -> dict:
        """Full RAG pipeline: retrieve + generate answer.

        Returns:
            {
                "answer": str,
                "sources": [...],
                "graph_entities": [...],
                "graph_context": str,
            }
        """
        from langchain.prompts import ChatPromptTemplate

        # Retrieve
        retrieval = self.retrieve(question)

        # Generate
        prompt_template = (
            system_prompt
            or """You are VEX, an expert ICT (Inner Circle Trader) trading assistant.
You have access to a knowledge graph of ICT concepts and their relationships.

Use the following context to answer the question. The context includes:
1. Retrieved documents from the ICT knowledge base
2. Graph-derived relationships showing how concepts connect

When answering:
- Reference specific ICT concepts and their relationships
- Explain causal chains (what must happen before what)
- Note confluences that strengthen or weaken setups
- Cite specific models, time windows, or rules when applicable
- If the graph shows a concept "requires" something, mention that prerequisite

Context:
{context}

---

Question: {question}
"""
        )

        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = self._init_llm()
        chain = prompt | llm

        response = chain.invoke(
            {
                "context": retrieval["combined_context"],
                "question": question,
            }
        )

        return {
            "answer": response.content,
            "sources": [c["source"] for c in retrieval["chunks"]],
            "graph_entities": retrieval["graph_entities"],
            "graph_context": retrieval["graph_context"],
        }

    # ── Graph-only queries (no vector search needed) ───────────────────────

    def graph_query(self, concept: str) -> dict:
        """Query the graph directly for everything about a concept.

        Useful for the Logic Engine — no LLM needed, just graph traversal.
        """
        concept = _normalize(concept)

        definition = self.store.get_concept_definition(concept)
        neighbors = self.store.get_neighbors(concept)
        related = self.store.get_related_concepts(concept, max_hops=2)
        models = self.store.get_models_for_pattern(concept)

        # Group neighbors by relation type
        by_relation = defaultdict(list)
        for n in neighbors:
            by_relation[n["relation"]].append(n["node"])

        return {
            "concept": concept,
            "definition": definition,
            "requires": by_relation.get("requires", []),
            "required_by": [
                n["node"]
                for n in neighbors
                if n["relation"] == "requires" and n["direction"] == "in"
            ],
            "enhances": by_relation.get("enhances", []),
            "enhanced_by": [
                n["node"]
                for n in neighbors
                if n["relation"] == "enhances" and n["direction"] == "in"
            ],
            "invalidated_by": by_relation.get("invalidates", []),
            "precedes": by_relation.get("precedes", []),
            "preceded_by": [
                n["node"]
                for n in neighbors
                if n["relation"] == "precedes" and n["direction"] == "in"
            ],
            "models": [m["model"] for m in models],
            "all_relations": dict(by_relation),
            "2_hop_related": related,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract ICT concept names from text using graph node matching."""
        text_lower = text.lower()
        found = []

        # Check all graph nodes against the text
        for node in self.store.G.nodes():
            # Skip very short names (likely false positives)
            if len(node) < 3:
                continue
            # Check both underscore and space forms
            if node in text_lower or node.replace("_", " ") in text_lower:
                found.append(node)

        return list(set(found))

    def _format_graph_context(self, triples: list[dict], entities: set[str]) -> str:
        """Format graph triples into a readable context string."""
        if not triples:
            return "No graph relationships found."

        lines = []
        # Group by subject
        by_subject = defaultdict(list)
        for t in triples:
            by_subject[t["subject"]].append(t)

        for subject in sorted(by_subject.keys()):
            subject_triples = by_subject[subject]
            lines.append(f"\n### {subject.replace('_', ' ').title()}")
            for t in subject_triples:
                pred = t["predicate"].replace("_", " ")
                obj = t["object"]
                if len(obj) > 200:  # Truncate long definitions
                    obj = obj[:200] + "..."
                lines.append(f"  - {pred}: {obj}")

        return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    """CLI: interactive GraphRAG chat."""
    store = ICTGraphStore().load_all()
    retriever = GraphRAGRetriever(store)

    print("\n=== ICT GraphRAG Chat ===")
    print("Type a question about ICT trading concepts.")
    print("Commands: /graph <concept> | /ingest | /stats | exit\n")

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break

            if q.startswith("/graph "):
                concept = q[7:].strip()
                import pprint

                pprint.pprint(retriever.graph_query(concept))
                continue

            if q == "/ingest":
                retriever.ingest_knowledge_base()
                continue

            if q == "/stats":
                import pprint

                pprint.pprint(store.stats())
                continue

            # Full RAG query
            print("Thinking...", end="", flush=True)
            result = retriever.query(q)
            print(f"\r{'':20}\r", end="")

            print(f"\nVEX: {result['answer']}")
            print(f"\nSources: {', '.join(result['sources'])}")
            print(f"Graph entities: {', '.join(result['graph_entities'][:10])}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
