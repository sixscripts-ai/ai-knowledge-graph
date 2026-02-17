"""
Phase 1: ICT Knowledge Graph Store
====================================
Loads 17,824 subject-predicate-object triples from ict_graph_final.json
AND structured relationships from concept_relationships.yaml + ict_ontology.yaml
into a unified NetworkX MultiDiGraph.

Usage:
    store = ICTGraphStore()
    store.load_all()
    
    # Query
    neighbors = store.get_neighbors("fair_value_gap", edge_type="requires")
    path = store.find_path("liquidity_sweep", "silver_bullet")
    subgraph = store.get_neighborhood("order_block", hops=2)
    related = store.get_related_concepts("displacement")
    models = store.get_models_for_pattern("fvg")
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    import yaml
except ImportError:
    yaml = None  # Graceful fallback — YAML sources are optional


# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # ai-knowledge-graph/
_TRAIN_ICT = Path(os.environ.get(
    "TRAIN_ICT_ROOT",
    Path.home() / "Documents" / "train-ict"
))

# Primary graph data
TRIPLES_PATH = _PROJECT_ROOT / "ict_graph_final.json"
GRAPH_DATA_PATH = _PROJECT_ROOT.parent / "graph_data.json"

# Structured schema files from train-ict
CONCEPT_RELS_PATH = _TRAIN_ICT / "data" / "schemas" / "concept_relationships.yaml"
ONTOLOGY_PATH = _TRAIN_ICT / "data" / "schemas" / "ict_ontology.yaml"
CONCEPT_GRAPH_PATH = _TRAIN_ICT / "knowledge_base" / "graphs" / "concept_graph.json"


def _normalize(name) -> str:
    """Normalize a node name for consistent lookup."""
    if not isinstance(name, str):
        if isinstance(name, list):
            name = ", ".join(str(x) for x in name)
        else:
            return ""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class ICTGraphStore:
    """Unified ICT knowledge graph backed by NetworkX."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self._node_types: dict[str, str] = {}  # node_id → type
        self._edge_counts: dict[str, int] = defaultdict(int)
        self._loaded_sources: list[str] = []

    # ── Loading ────────────────────────────────────────────────────────────

    def load_all(self) -> "ICTGraphStore":
        """Load every available data source into the graph."""
        self.load_triples()
        self.load_graph_data()
        self.load_concept_relationships()
        self.load_ontology()
        self.load_concept_graph()
        return self

    def load_triples(self, path: Optional[Path] = None) -> int:
        """Load subject-predicate-object triples from ict_graph_final.json."""
        path = path or TRIPLES_PATH
        if not path.exists():
            print(f"[graph_store] Skipping triples — {path} not found")
            return 0

        with open(path, "r") as f:
            triples = json.load(f)

        count = 0
        for t in triples:
            # Skip triples with null fields
            if not t.get("subject") or not t.get("predicate") or not t.get("object"):
                continue

            subj = _normalize(t["subject"])
            pred = t["predicate"].strip().lower()
            obj = _normalize(t["object"])
            chunk = t.get("chunk", 0)

            # Skip degenerate triples
            if not subj or not obj or obj == "none":
                continue

            self.G.add_node(subj, type="concept")
            self.G.add_node(obj, type="concept")
            self.G.add_edge(subj, obj, relation=pred, source="triples", chunk=chunk)
            self._edge_counts[pred] += 1
            count += 1

        self._loaded_sources.append(f"triples ({count} edges)")
        print(f"[graph_store] Loaded {count} triples → {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return count

    def load_graph_data(self, path: Optional[Path] = None) -> int:
        """Load the ForceGraph-format graph_data.json (85 nodes, 97 links)."""
        path = path or GRAPH_DATA_PATH
        if not path.exists():
            print(f"[graph_store] Skipping graph_data — {path} not found")
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        count = 0
        for node in data.get("nodes", []):
            nid = _normalize(node["id"])
            self.G.add_node(nid,
                            type=node.get("group", "concept"),
                            label=node.get("label", node.get("name", node["id"])),
                            **{k: v for k, v in node.items()
                               if k not in ("id", "group", "name", "label", "type")})
            self._node_types[nid] = node.get("group", "concept")

        for link in data.get("links", []):
            src = _normalize(link["source"] if isinstance(link["source"], str)
                             else link["source"]["id"])
            tgt = _normalize(link["target"] if isinstance(link["target"], str)
                             else link["target"]["id"])
            rel = link.get("label", link.get("relation", "related_to")).strip().lower()

            self.G.add_edge(src, tgt, relation=rel, source="graph_data")
            self._edge_counts[rel] += 1
            count += 1

        self._loaded_sources.append(f"graph_data ({count} edges)")
        print(f"[graph_store] Loaded graph_data → total {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return count

    def load_concept_relationships(self, path: Optional[Path] = None) -> int:
        """Load typed ICT concept relationships from concept_relationships.yaml.
        
        This is the richest source — it has causal chains, requirements,
        confluence weights, model blueprints, anti-patterns, and time rules.
        """
        path = path or CONCEPT_RELS_PATH
        if not path.exists() or yaml is None:
            print(f"[graph_store] Skipping concept_relationships — {'no yaml' if yaml is None else 'not found'}")
            return 0

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        count = 0

        # ── Causal chains ──
        for chain_name, chain in (data.get("causal_chains") or {}).items():
            steps = chain.get("steps", {})
            sorted_steps = sorted(steps.items(), key=lambda x: int(x[0]))
            for i in range(len(sorted_steps) - 1):
                curr = _normalize(sorted_steps[i][1].get("concept",
                                  sorted_steps[i][1].get("phase", f"step_{i}")))
                nxt = _normalize(sorted_steps[i + 1][1].get("concept",
                                 sorted_steps[i + 1][1].get("phase", f"step_{i+1}")))
                self.G.add_node(curr, type="concept")
                self.G.add_node(nxt, type="concept")
                self.G.add_edge(curr, nxt, relation="precedes",
                                source="concept_rels", chain=chain_name)
                count += 1

        # ── Concept requirements ──
        for concept, reqs in (data.get("concept_requirements") or {}).items():
            cnode = _normalize(concept)
            self.G.add_node(cnode, type="concept")

            for req in (reqs.get("requires") or []):
                rnode = _normalize(req["concept"])
                self.G.add_node(rnode, type="concept")
                self.G.add_edge(cnode, rnode, relation="requires",
                                source="concept_rels", reason=req.get("why", ""))
                count += 1

            for enh in (reqs.get("enhanced_by") or []):
                enode = _normalize(enh["concept"])
                self.G.add_node(enode, type="concept")
                self.G.add_edge(enode, cnode, relation="enhances",
                                source="concept_rels",
                                bonus=enh.get("bonus", 0),
                                reason=enh.get("why", ""))
                count += 1

            for inv in (reqs.get("invalidated_by") or []):
                cond = _normalize(inv.get("condition", inv.get("concept", "unknown")))
                self.G.add_node(cond, type="condition")
                self.G.add_edge(cond, cnode, relation="invalidates",
                                source="concept_rels", reason=inv.get("why", ""))
                count += 1

            for tgt in (reqs.get("targets") or []):
                tnode = _normalize(tgt["concept"])
                self.G.add_node(tnode, type="concept")
                self.G.add_edge(cnode, tnode, relation="targets",
                                source="concept_rels", reason=tgt.get("why", ""))
                count += 1

        # ── Model blueprints ──
        for model_name, model in (data.get("models") or {}).items():
            mnode = _normalize(model_name)
            self.G.add_node(mnode, type="model")

            for req_str in (model.get("required") or []):
                # Extract concept names from requirement strings
                for concept in self._extract_concepts_from_text(req_str):
                    self.G.add_edge(mnode, concept, relation="requires",
                                    source="concept_rels",
                                    requirement_text=req_str)
                    count += 1

            for tw in (model.get("time_windows") or []):
                tw_node = _normalize(tw.get("name", f"{model_name}_window"))
                self.G.add_node(tw_node, type="time_window",
                                start=tw.get("start"), end=tw.get("end"),
                                timezone=tw.get("timezone"))
                self.G.add_edge(mnode, tw_node, relation="active_during",
                                source="concept_rels")
                count += 1

        # ── Confluence weights ──
        weights = data.get("confluence_weights", {})
        for tier, items in weights.items():
            if tier in ("thresholds",):
                continue
            if isinstance(items, dict):
                for factor, weight in items.items():
                    fnode = _normalize(factor)
                    self.G.add_node(fnode, type="confluence_factor")
                    if not self.G.has_node("confluence_scoring"):
                        self.G.add_node("confluence_scoring", type="system")
                    self.G.add_edge(fnode, "confluence_scoring",
                                    relation="contributes_to",
                                    weight=weight, tier=tier,
                                    source="concept_rels")
                    count += 1

        # ── Anti-patterns ──
        for ap_name, ap in (data.get("anti_patterns") or {}).items():
            anode = _normalize(ap_name)
            self.G.add_node(anode, type="anti_pattern",
                            description=ap.get("description", ""),
                            why_fails=ap.get("why_fails", ""),
                            fix=ap.get("fix", ""))
            # Link anti-patterns to the concepts they violate
            for concept in self._extract_concepts_from_text(
                    f"{ap.get('description', '')} {ap.get('fix', '')}"):
                self.G.add_edge(anode, concept, relation="violates",
                                source="concept_rels")
                count += 1

        # ── Killzones / time rules ──
        for kz_name, kz in (data.get("time_rules", {}).get("killzones") or {}).items():
            kznode = _normalize(f"killzone_{kz_name}")
            self.G.add_node(kznode, type="killzone",
                            time=kz.get("time", ""),
                            behavior=kz.get("behavior", ""))
            for setup in (kz.get("best_setups") or []):
                snode = _normalize(setup)
                self.G.add_edge(kznode, snode, relation="best_for",
                                source="concept_rels")
                count += 1

        self._loaded_sources.append(f"concept_relationships ({count} edges)")
        print(f"[graph_store] Loaded concept_relationships → total {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return count

    def load_ontology(self, path: Optional[Path] = None) -> int:
        """Load ICT ontology definitions from ict_ontology.yaml."""
        path = path or ONTOLOGY_PATH
        if not path.exists() or yaml is None:
            print(f"[graph_store] Skipping ontology — {'no yaml' if yaml is None else 'not found'}")
            return 0

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        count = 0

        # Walk all top-level categories and extract definitions
        for category, items in data.items():
            if not isinstance(items, dict):
                continue
            cat_node = _normalize(category)
            self.G.add_node(cat_node, type="category")

            for item_name, item_data in items.items():
                inode = _normalize(item_name)
                if isinstance(item_data, dict):
                    self.G.add_node(inode, type="concept",
                                    definition=item_data.get("definition",
                                               item_data.get("description", "")),
                                    **{k: v for k, v in item_data.items()
                                       if isinstance(v, (str, int, float, bool))
                                       and k not in ("definition", "description")})
                    self.G.add_edge(inode, cat_node, relation="belongs_to",
                                    source="ontology")
                    count += 1

                    # Sub-items (e.g. bos, choch under structures)
                    for sub_name, sub_data in item_data.items():
                        if isinstance(sub_data, dict) and any(
                                k in sub_data for k in ("definition", "meaning",
                                                        "description", "full_name")):
                            snode = _normalize(sub_name)
                            self.G.add_node(snode, type="concept",
                                            definition=sub_data.get("definition",
                                                       sub_data.get("meaning", "")))
                            self.G.add_edge(snode, inode, relation="is_type_of",
                                            source="ontology")
                            count += 1
                elif isinstance(item_data, str):
                    self.G.add_node(inode, type="concept", definition=item_data)
                    self.G.add_edge(inode, cat_node, relation="belongs_to",
                                    source="ontology")
                    count += 1

        self._loaded_sources.append(f"ontology ({count} edges)")
        print(f"[graph_store] Loaded ontology → total {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return count

    def load_concept_graph(self, path: Optional[Path] = None) -> int:
        """Load concept_graph.json from train-ict knowledge_base."""
        path = path or CONCEPT_GRAPH_PATH
        if not path.exists():
            print(f"[graph_store] Skipping concept_graph — {path} not found")
            return 0

        with open(path, "r") as f:
            data = json.load(f)

        count = 0
        # Handle both {nodes, edges} and flat-list formats
        if isinstance(data, dict):
            nodes_data = data.get("nodes", [])
            # nodes can be a dict keyed by id or a list of dicts
            if isinstance(nodes_data, dict):
                for key, node in nodes_data.items():
                    if isinstance(node, dict):
                        nid = _normalize(node.get("id", key))
                    else:
                        nid = _normalize(key)
                    if nid:
                        attrs = node if isinstance(node, dict) else {}
                        self.G.add_node(nid,
                                        type=attrs.get("category", attrs.get("type", "concept")),
                                        label=attrs.get("name", key),
                                        level=attrs.get("level", 0),
                                        description=attrs.get("description", ""))
            else:
                for node in nodes_data:
                    nid = _normalize(node.get("id", node.get("name", "")))
                    if nid:
                        self.G.add_node(nid, type=node.get("type", "concept"))

            for edge in data.get("edges", data.get("links", [])):
                src = _normalize(edge.get("source", edge.get("from", "")))
                tgt = _normalize(edge.get("target", edge.get("to", "")))
                rel = edge.get("relation", edge.get("relationship", edge.get("label", "related_to")))
                rel = rel.lower() if isinstance(rel, str) else "related_to"
                if src and tgt:
                    desc = edge.get("description", "")
                    self.G.add_edge(src, tgt, relation=rel, source="concept_graph",
                                    description=desc)
                    count += 1

        self._loaded_sources.append(f"concept_graph ({count} edges)")
        print(f"[graph_store] Loaded concept_graph → total {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return count

    # ── Querying ───────────────────────────────────────────────────────────

    def get_neighbors(self, node: str, edge_type: Optional[str] = None,
                      direction: str = "both") -> list[dict]:
        """Get neighbors of a node, optionally filtered by edge type.
        
        Args:
            node: Node name (will be normalized)
            edge_type: Filter to specific relation type (e.g. "requires", "enhances")
            direction: "out", "in", or "both"
        
        Returns:
            List of dicts with keys: node, relation, direction, attrs
        """
        node = _normalize(node)
        if node not in self.G:
            return []

        results = []

        if direction in ("out", "both"):
            for _, tgt, data in self.G.out_edges(node, data=True):
                if edge_type and data.get("relation") != edge_type:
                    continue
                results.append({
                    "node": tgt,
                    "relation": data.get("relation", "related"),
                    "direction": "out",
                    "attrs": {k: v for k, v in data.items()
                              if k not in ("relation", "source")}
                })

        if direction in ("in", "both"):
            for src, _, data in self.G.in_edges(node, data=True):
                if edge_type and data.get("relation") != edge_type:
                    continue
                results.append({
                    "node": src,
                    "relation": data.get("relation", "related"),
                    "direction": "in",
                    "attrs": {k: v for k, v in data.items()
                              if k not in ("relation", "source")}
                })

        return results

    def get_neighborhood(self, node: str, hops: int = 2) -> nx.MultiDiGraph:
        """Get the N-hop neighborhood subgraph around a node."""
        node = _normalize(node)
        if node not in self.G:
            return nx.MultiDiGraph()

        # BFS to find all nodes within N hops (treating as undirected)
        undirected = self.G.to_undirected(as_view=True)
        neighborhood = set()
        frontier = {node}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                for neighbor in undirected.neighbors(n):
                    if neighbor not in neighborhood and neighbor not in frontier:
                        next_frontier.add(neighbor)
            neighborhood.update(frontier)
            frontier = next_frontier
        neighborhood.update(frontier)

        return self.G.subgraph(neighborhood).copy()

    def find_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two concepts."""
        source, target = _normalize(source), _normalize(target)
        try:
            return nx.shortest_path(self.G, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Try undirected
            try:
                return nx.shortest_path(self.G.to_undirected(), source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

    def find_path_with_relations(self, source: str, target: str) -> list[dict]:
        """Find shortest path with edge relation details."""
        path = self.find_path(source, target)
        if len(path) < 2:
            return []

        result = []
        for i in range(len(path) - 1):
            edges = self.G.get_edge_data(path[i], path[i + 1])
            if edges:
                # Get the first edge's relation
                edge_data = list(edges.values())[0]
                result.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "relation": edge_data.get("relation", "related"),
                })
            else:
                # Check reverse direction
                edges = self.G.get_edge_data(path[i + 1], path[i])
                if edges:
                    edge_data = list(edges.values())[0]
                    result.append({
                        "from": path[i + 1],
                        "to": path[i],
                        "relation": edge_data.get("relation", "related"),
                        "reversed": True,
                    })
        return result

    def get_models_for_pattern(self, pattern: str) -> list[dict]:
        """Find all trading models that require or use a given pattern."""
        pattern = _normalize(pattern)
        results = []

        # Direct edges from models to this pattern
        for src, _, data in self.G.in_edges(pattern, data=True):
            if self.G.nodes[src].get("type") == "model":
                results.append({
                    "model": src,
                    "relation": data.get("relation", "uses"),
                    "attrs": {k: v for k, v in data.items()
                              if k not in ("relation", "source")}
                })

        # Also check outgoing from pattern to models
        for _, tgt, data in self.G.out_edges(pattern, data=True):
            if self.G.nodes[tgt].get("type") == "model":
                results.append({
                    "model": tgt,
                    "relation": data.get("relation", "used_by"),
                    "attrs": {k: v for k, v in data.items()
                              if k not in ("relation", "source")}
                })

        return results

    def get_related_concepts(self, concept: str, max_hops: int = 2) -> dict[str, list[str]]:
        """Get concepts related to the given one, grouped by relationship type."""
        concept = _normalize(concept)
        neighbors = self.get_neighbors(concept)

        by_relation: dict[str, list[str]] = defaultdict(list)
        for n in neighbors:
            by_relation[n["relation"]].append(n["node"])

        # Also get 2-hop connections
        if max_hops >= 2:
            for n in neighbors:
                for n2 in self.get_neighbors(n["node"]):
                    if n2["node"] != concept:
                        key = f"{n['relation']}→{n2['relation']}"
                        if n2["node"] not in by_relation[key]:
                            by_relation[key].append(n2["node"])

        return dict(by_relation)

    def get_concept_definition(self, concept: str) -> Optional[str]:
        """Get the definition of a concept from the graph node attributes."""
        concept = _normalize(concept)
        if concept not in self.G:
            return None
        attrs = self.G.nodes[concept]
        return attrs.get("definition", attrs.get("description", None))

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        """Get all nodes of a given type."""
        return [n for n, d in self.G.nodes(data=True)
                if d.get("type") == node_type]

    def get_all_relation_types(self) -> dict[str, int]:
        """Get counts of all edge relation types in the graph."""
        counts: dict[str, int] = defaultdict(int)
        for _, _, data in self.G.edges(data=True):
            counts[data.get("relation", "unknown")] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Simple text search across node names and definitions."""
        query_lower = query.lower()
        results = []

        for node, data in self.G.nodes(data=True):
            score = 0
            if query_lower in node:
                score += 10
            if query_lower == node:
                score += 50
            definition = data.get("definition", "")
            if isinstance(definition, str) and query_lower in definition.lower():
                score += 5

            if score > 0:
                results.append({
                    "node": node,
                    "type": data.get("type", "unknown"),
                    "definition": definition,
                    "score": score,
                    "degree": self.G.degree(node),
                })

        results.sort(key=lambda x: (-x["score"], -x["degree"]))
        return results[:top_k]

    # ── Export / Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return graph statistics."""
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "sources": self._loaded_sources,
            "node_types": dict(defaultdict(int, {
                d.get("type", "unknown"): sum(1 for _, d2 in self.G.nodes(data=True)
                                              if d2.get("type") == d.get("type"))
                for _, d in self.G.nodes(data=True)
            })),
            "relation_types": self.get_all_relation_types(),
            "connected_components": nx.number_weakly_connected_components(self.G),
            "density": nx.density(self.G),
        }

    def to_triples(self) -> list[dict]:
        """Export the entire graph as a list of triples."""
        return [
            {"subject": src, "predicate": data.get("relation", "related"),
             "object": tgt}
            for src, tgt, data in self.G.edges(data=True)
        ]

    def export_for_neo4j(self, output_dir: Path) -> None:
        """Export nodes and edges as CSVs for Neo4j import."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nodes
        with open(output_dir / "nodes.csv", "w") as f:
            f.write("id:ID,type,definition\n")
            for node, data in self.G.nodes(data=True):
                defn = str(data.get("definition", "")).replace('"', '\\"')
                f.write(f'"{node}","{data.get("type", "")}","{defn}"\n')

        # Edges
        with open(output_dir / "edges.csv", "w") as f:
            f.write(":START_ID,:END_ID,:TYPE,source\n")
            for src, tgt, data in self.G.edges(data=True):
                rel = data.get("relation", "RELATED")
                source = data.get("source", "")
                f.write(f'"{src}","{tgt}","{rel.upper()}","{source}"\n')

        print(f"[graph_store] Exported to {output_dir}/")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """Best-effort extraction of ICT concept names from requirement text."""
        # Known ICT concepts that might appear in requirement strings
        known = [
            "fvg", "fair_value_gap", "order_block", "ob", "liquidity",
            "displacement", "bos", "break_of_structure", "choch",
            "market_structure_shift", "mss", "ote", "optimal_trade_entry",
            "htf_bias", "htf", "ltf", "smt", "smt_divergence",
            "accumulation", "manipulation", "distribution",
            "killzone", "asian_range", "cbdr", "judas_swing",
            "swing_high", "swing_low", "equal_highs", "equal_lows",
            "buy_side_liquidity", "sell_side_liquidity", "bsl", "ssl",
            "premium", "discount", "equilibrium",
            "silver_bullet", "turtle_soup", "unicorn",
            "pd_array", "imbalance", "mitigation",
        ]
        text_lower = text.lower()
        found = []
        for concept in known:
            if concept in text_lower or concept.replace("_", " ") in text_lower:
                found.append(_normalize(concept))
        return found if found else [_normalize(text[:60])]  # Fallback: use text itself


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point — load graph and print stats."""
    import pprint
    store = ICTGraphStore()
    store.load_all()

    print("\n=== Graph Statistics ===")
    pprint.pprint(store.stats())

    print("\n=== Sample: FVG neighbors ===")
    for n in store.get_neighbors("fair_value_gap")[:10]:
        print(f"  {n['direction']:>3} | {n['relation']:<20} | {n['node']}")

    print("\n=== Sample: Path from liquidity_sweep to silver_bullet ===")
    path = store.find_path_with_relations("liquidity_sweep", "silver_bullet")
    for step in path:
        print(f"  {step['from']} --[{step['relation']}]--> {step['to']}")

    print("\n=== Models for FVG ===")
    for m in store.get_models_for_pattern("fvg"):
        print(f"  {m['model']} ({m['relation']})")


if __name__ == "__main__":
    main()
