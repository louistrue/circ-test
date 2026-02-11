"""Command-line interface for enhanced IFC product matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_match(args: argparse.Namespace) -> None:
    """Run a matching query against a database."""
    from ifc_matching.engines.embedding import EmbeddingMatcher
    from ifc_matching.engines.reranker import LLMReranker
    from ifc_matching.engines.hybrid import HybridMatcher

    # Load candidates
    if args.database_file:
        path = Path(args.database_file)
        if path.suffix == ".json":
            candidates = json.loads(path.read_text())
            if isinstance(candidates, dict):
                candidates = candidates.get("materials", candidates.get("names", []))
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                candidates = [c.get("name", str(c)) for c in candidates]
        else:
            candidates = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip()
            ]
    else:
        candidates = args.candidates

    if not candidates:
        print("Error: no candidates provided. Use --candidates or --database-file.", file=sys.stderr)
        sys.exit(1)

    # Build engine
    if args.mode == "embedding":
        backend = "openai" if "embedding" in args.model and "text-" in args.model else "sentence-transformers"
        engine = EmbeddingMatcher(
            model_name=args.model,
            backend=backend,
        )
    elif args.mode == "rerank":
        engine = LLMReranker(model_name=args.model)
    else:  # hybrid
        retriever = EmbeddingMatcher(
            model_name=args.retriever_model or "all-MiniLM-L6-v2",
            backend="openai" if args.retriever_model and "text-" in args.retriever_model else "sentence-transformers",
        )
        reranker = LLMReranker(model_name=args.reranker_model or "gpt-4o")
        engine = HybridMatcher(
            retriever=retriever,
            reranker=reranker,
            retrieval_k=args.retrieval_k,
        )

    results = engine.match(
        args.query,
        candidates,
        top_k=args.top_k,
        element_type=args.element_type,
    )

    # Output
    if args.json:
        output = [
            {"candidate": r.candidate, "score": r.score, "method": r.method, "metadata": r.metadata}
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        print(f"\nMatching results for: {args.query!r}")
        if args.element_type:
            print(f"Element type context: {args.element_type}")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.candidate:<40s}  score={r.score:.4f}  [{r.method}]")
        print()


def _cmd_proximity(args: argparse.Namespace) -> None:
    """Analyse geometric proximity in an IFC file."""
    from ifc_matching.geometry.proximity import GeometricProximityAnalyzer

    analyzer = GeometricProximityAnalyzer(
        touch_tolerance=args.tolerance,
        near_threshold=args.near_threshold,
    )

    print(f"Analysing geometric proximity in: {args.ifc_file}")
    graph = analyzer.build_adjacency_graph(args.ifc_file)

    total_edges = sum(len(edges) for edges in graph.values()) // 2  # undirected
    print(f"Found {len(graph)} elements with {total_edges} adjacency edges")

    if args.json:
        output = {}
        for eid, edges in graph.items():
            output[eid] = [
                {
                    "element_a": e.element_a,
                    "element_b": e.element_b,
                    "distance": e.distance,
                    "overlap_volume": e.overlap_volume,
                    "contact_type": e.contact_type,
                }
                for e in edges
            ]
        print(json.dumps(output, indent=2))
    else:
        for eid, edges in sorted(graph.items()):
            print(f"\n  {eid}:")
            for e in edges[:5]:
                other = e.element_b if e.element_a == eid else e.element_a
                print(f"    -> {other}  [{e.contact_type}]  dist={e.distance:.4f}")
            if len(edges) > 5:
                print(f"    ... and {len(edges) - 5} more")


def _cmd_bci(args: argparse.Namespace) -> None:
    """Calculate or simulate BCI via Neo4j."""
    from ifc_matching.graph.neo4j_store import Neo4jCircularityStore
    from ifc_matching.databases.archetype import ConnectionType

    store = Neo4jCircularityStore(
        uri=args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
        database=args.neo4j_database,
    )

    if args.simulate:
        # Format: connection_id:new_type (e.g. "conn-1:BOLTED")
        parts = args.simulate.split(":")
        conn_id = parts[0]
        new_type = ConnectionType[parts[1].upper()]
        projected = store.simulate_connection_change(conn_id, new_type)
        current = store.calculate_bci()
        print(f"Current BCI:   {current:.4f}")
        print(f"Projected BCI: {projected:.4f} (if {conn_id} -> {new_type.name})")
        print(f"Delta:         {projected - current:+.4f}")
    else:
        bci = store.calculate_bci()
        print(f"Building Circularity Indicator: {bci:.4f}")

    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ifc-match",
        description="Enhanced IFC Product Matching CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- match --
    p_match = subparsers.add_parser("match", help="Match an IFC product to database entries")
    p_match.add_argument("query", help="IFC product/material name to match")
    p_match.add_argument("--candidates", nargs="+", help="Candidate database entries")
    p_match.add_argument("--database-file", help="File with candidate entries (JSON/txt, one per line)")
    p_match.add_argument("--mode", choices=["embedding", "rerank", "hybrid"], default="embedding")
    p_match.add_argument("--model", default="all-MiniLM-L6-v2", help="Model for embedding or rerank mode")
    p_match.add_argument("--retriever-model", help="Embedding model for hybrid mode stage 1")
    p_match.add_argument("--reranker-model", help="LLM model for hybrid mode stage 2")
    p_match.add_argument("--retrieval-k", type=int, default=10, help="Candidates to pass to reranker")
    p_match.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    p_match.add_argument("--element-type", help="IFC element type for context (e.g. IfcWall)")
    p_match.add_argument("--json", action="store_true", help="Output as JSON")
    p_match.set_defaults(func=_cmd_match)

    # -- proximity --
    p_prox = subparsers.add_parser("proximity", help="Analyse geometric proximity in IFC file")
    p_prox.add_argument("ifc_file", help="Path to IFC file")
    p_prox.add_argument("--tolerance", type=float, default=0.01, help="Touch tolerance (m)")
    p_prox.add_argument("--near-threshold", type=float, default=0.5, help="Near threshold (m)")
    p_prox.add_argument("--json", action="store_true", help="Output as JSON")
    p_prox.set_defaults(func=_cmd_proximity)

    # -- bci --
    p_bci = subparsers.add_parser("bci", help="Calculate Building Circularity Indicator")
    p_bci.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p_bci.add_argument("--neo4j-user", default="neo4j")
    p_bci.add_argument("--neo4j-password", default="password")
    p_bci.add_argument("--neo4j-database", default="neo4j")
    p_bci.add_argument("--simulate", help="Simulate change: CONNECTION_ID:NEW_TYPE (e.g. conn-1:BOLTED)")
    p_bci.set_defaults(func=_cmd_bci)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
