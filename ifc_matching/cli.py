"""Command-line interface for enhanced IFC product matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _infer_llm_backend(model_name: str) -> str:
    """Infer 'openai', 'openai-responses', or 'anthropic' backend from the model name."""
    if "claude" in model_name.lower():
        return "anthropic"
    if model_name.startswith("gpt-5"):
        return "openai-responses"
    return "openai"


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
        rerank_backend = _infer_llm_backend(args.model)
        engine = LLMReranker(model_name=args.model, backend=rerank_backend)
    else:  # hybrid
        from ifc_matching.engines.embedding import DEFAULT_MODEL as _DEFAULT_EMB
        retriever = EmbeddingMatcher(
            model_name=args.retriever_model or _DEFAULT_EMB,
            backend="openai" if args.retriever_model and "text-" in args.retriever_model else "sentence-transformers",
        )
        reranker_model = args.reranker_model or "gpt-4o"
        rerank_backend = _infer_llm_backend(reranker_model)
        reranker = LLMReranker(model_name=reranker_model, backend=rerank_backend)
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


def _cmd_ifc_import(args: argparse.Namespace) -> None:
    """Extract materials and elements from an IFC file."""
    from ifc_matching.ifc_extractor import IfcMaterialExtractor

    extractor = IfcMaterialExtractor(args.ifc_file)

    if args.summary:
        summary = extractor.summary()
        print(f"\nIFC Summary: {args.ifc_file}")
        print(f"  Total elements:        {summary['total_elements']}")
        print(f"  With materials:        {summary['with_materials']}")
        print(f"  Total material layers: {summary['total_material_layers']}")
        print(f"\n  Element types:")
        for etype, count in sorted(summary["element_types"].items(), key=lambda x: -x[1]):
            print(f"    {etype:<30s} {count}")
        return

    elements = extractor.extract_all()

    if args.json:
        import dataclasses
        output = []
        for el in elements:
            d = dataclasses.asdict(el)
            d["match_query"] = el.match_query
            d["primary_material"] = el.primary_material
            output.append(d)
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"\nExtracted {len(elements)} elements from: {args.ifc_file}")
        print("-" * 70)
        for el in elements:
            mats = ", ".join(el.material_names) if el.materials else "(no materials)"
            print(f"  {el.element_type:<25s} {el.name:<30s} -> {mats}")
            if args.verbose:
                print(f"    GlobalId: {el.global_id}")
                print(f"    Type: {el.type_name}  Predefined: {el.predefined_type}")
                print(f"    Match query: {el.match_query!r}")
                if el.properties:
                    print(f"    Properties: {el.properties}")
                if el.quantities:
                    print(f"    Quantities: {el.quantities}")
                for mat in el.materials:
                    thick = f" ({mat.thickness*1000:.0f}mm)" if mat.thickness else ""
                    print(f"      Layer: {mat.name}{thick}")
        print()


def _cmd_match_ifc(args: argparse.Namespace) -> None:
    """Match all materials from an IFC file against a database (e.g. KBOB, ÖKOBAUDAT)."""
    from ifc_matching.ifc_extractor import IfcMaterialExtractor
    from ifc_matching.engines.embedding import EmbeddingMatcher, DEFAULT_MODEL
    from ifc_matching.engines.reranker import LLMReranker
    from ifc_matching.engines.hybrid import HybridMatcher
    from ifc_matching.preprocessing import preprocess_query

    # Determine data source and load candidates (or prepare per-query search)
    source = "kbob"  # kbob | oekobaudat | file
    candidates: list[str] = []
    oeko_client = None
    page_size = 50

    if args.oekobaudat:
        source = "oekobaudat"
        from ifc_matching.databases.oekobaudat_client import (
            OekobaudatClient,
            DEFAULT_DATASTOCK_UUID,
            COMPLIANCE_EN15804_A2,
        )
        print("Using ÖKOBAUDAT search API (EN 15804+A2)...")
        oeko_client = OekobaudatClient(
            datastock_uuid=args.oekobau_datastock or DEFAULT_DATASTOCK_UUID,
            compliance_uuid=args.oekobau_compliance or COMPLIANCE_EN15804_A2,
            cache_path=args.oekobau_cache,
        )
        page_size = args.oekobau_page_size
    elif args.kbob:
        from ifc_matching.databases.kbob_client import KBOBClient
        print("Fetching KBOB materials from lcadata.ch...")
        client = KBOBClient(
            api_key=args.kbob_key,
            cache_path=args.kbob_cache or ".kbob_cache.json",
        )
        candidates = client.get_material_names()
        print(f"  Loaded {len(candidates)} KBOB materials")
    elif args.database_file:
        source = "file"
        path = Path(args.database_file)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                candidates = [
                    item.get("name", item.get("nameDE", str(item)))
                    if isinstance(item, dict) else str(item)
                    for item in data
                ]
            elif isinstance(data, dict):
                items = data.get("materials", data.get("data", []))
                candidates = [
                    item.get("name", item.get("nameDE", str(item)))
                    if isinstance(item, dict) else str(item)
                    for item in items
                ]
        else:
            candidates = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    if source != "oekobaudat" and not candidates:
        print("Error: no candidates. Use --kbob, --oekobaudat, or --database-file.", file=sys.stderr)
        sys.exit(1)

    # Extract elements from IFC
    print(f"Extracting materials from: {args.ifc_file}")
    extractor = IfcMaterialExtractor(args.ifc_file)
    match_data = extractor.extract_for_matching()
    print(f"  Found {len(match_data)} material queries to match")

    # Build matcher
    if args.mode == "hybrid":
        reranker_model = args.reranker_model or "claude-opus-4-6"
        rerank_backend = _infer_llm_backend(reranker_model)
        oeko_search_fn = None
        if source == "oekobaudat" and oeko_client:
            def _oeko_search(query: str, element_type: str | None = None, class_hint: str | None = None, limit: int = 30):
                results = oeko_client.search_processes(name=query, class_id=class_hint, limit=limit)
                return [r.display_name for r in results]
            oeko_search_fn = _oeko_search
        engine = HybridMatcher(
            retriever=EmbeddingMatcher(
                model_name=args.retriever_model or DEFAULT_MODEL,
                backend="sentence-transformers",
            ),
            reranker=LLMReranker(
                model_name=reranker_model,
                backend=rerank_backend,
                temperature=0.0,
                oekobaudat_search_fn=oeko_search_fn,
            ),
            retrieval_k=10,
        )
    else:
        engine = EmbeddingMatcher(
            model_name=args.retriever_model or DEFAULT_MODEL,
            backend="sentence-transformers",
        )

    # Match each extracted material
    db_label = "ÖKOBAUDAT search" if source == "oekobaudat" else f"{len(candidates)} database entries"
    print(f"\nMatching against {db_label} ({args.mode} mode)...")
    results = []
    for i, md in enumerate(match_data, 1):
        if source == "oekobaudat" and oeko_client:
            # Per-query search: extract clean German keywords for ÖKOBAUDAT API
            from ifc_matching.preprocessing import (
                extract_oekobaudat_search_terms,
                filter_candidates_by_element_type,
            )

            search_terms = extract_oekobaudat_search_terms(md["query"])
            # Search each term and merge results (dedup by display name)
            all_results: dict[str, object] = {}
            for term in search_terms:
                hits = oeko_client.search_processes(name=term, limit=page_size)
                for r in hits:
                    if r.display_name not in all_results:
                        all_results[r.display_name] = r
            search_results = list(all_results.values())
            candidates = [r.display_name for r in search_results]
            if candidates and md.get("element_type"):
                candidate_ids = {r.display_name: r.classific_id for r in search_results}
                candidates = filter_candidates_by_element_type(
                    candidates, candidate_ids, md["element_type"], source="oekobaudat"
                )
            if not candidates:
                candidates = ["(no ÖKOBAUDAT results)"]

        match_results = engine.match(
            md["query"],
            candidates,
            top_k=3,
            element_type=md["element_type"],
        )
        top = match_results[0] if match_results else None
        result = {
            **md,
            "matched_to": top.candidate if top else "N/A",
            "score": round(top.score, 4) if top else 0,
            "method": top.method if top else "",
        }
        results.append(result)

        if not args.json:
            tag = f"[{top.score:.2f}]" if top else "[N/A]"
            print(f"  {i:3d}. {md['query']:<35s} -> {result['matched_to']:<35s} {tag} ({md['element_type']})")

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    # Save results
    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.output}")

    print(f"\nMatched {len(results)} materials from IFC to database")


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
    p_match.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Model for embedding or rerank mode")
    p_match.add_argument("--retriever-model", help="Embedding model for hybrid mode stage 1")
    p_match.add_argument("--reranker-model", help="LLM model for hybrid mode stage 2")
    p_match.add_argument("--retrieval-k", type=int, default=10, help="Candidates to pass to reranker")
    p_match.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    p_match.add_argument("--element-type", help="IFC element type for context (e.g. IfcWall)")
    p_match.add_argument("--json", action="store_true", help="Output as JSON")
    p_match.set_defaults(func=_cmd_match)

    # -- ifc-import --
    p_ifc = subparsers.add_parser("ifc-import", help="Extract materials and elements from an IFC file")
    p_ifc.add_argument("ifc_file", help="Path to IFC file")
    p_ifc.add_argument("--summary", action="store_true", help="Show summary only")
    p_ifc.add_argument("--verbose", "-v", action="store_true", help="Show detailed info per element")
    p_ifc.add_argument("--json", action="store_true", help="Output as JSON")
    p_ifc.set_defaults(func=_cmd_ifc_import)

    # -- match-ifc --
    p_mifc = subparsers.add_parser("match-ifc", help="Match IFC materials against a database (e.g. KBOB)")
    p_mifc.add_argument("ifc_file", help="Path to IFC file")
    p_mifc.add_argument("--kbob", action="store_true", help="Use KBOB database via lcadata.ch API")
    p_mifc.add_argument("--kbob-key", help="lcadata.ch API key (or set LCADATA_API_KEY env var)")
    p_mifc.add_argument("--kbob-cache", help="Path to cache KBOB data locally (default: .kbob_cache.json)")
    p_mifc.add_argument("--oekobaudat", action="store_true", help="Use ÖKOBAUDAT database via search API")
    p_mifc.add_argument("--oekobau-datastock", help="ÖKOBAUDAT datastock UUID (default: 2021-II)")
    p_mifc.add_argument("--oekobau-compliance", help="Compliance UUID (default: EN 15804+A2)")
    p_mifc.add_argument("--oekobau-page-size", type=int, default=50, help="Max EPDs per search (default: 50)")
    p_mifc.add_argument("--oekobau-cache", help="Path to cache ÖKOBAUDAT search results")
    p_mifc.add_argument("--database-file", help="Alternative: path to JSON/txt file with candidate names")
    p_mifc.add_argument("--mode", choices=["embedding", "hybrid"], default="hybrid")
    p_mifc.add_argument("--retriever-model", help="Embedding model for stage 1")
    p_mifc.add_argument("--reranker-model", help="LLM model for stage 2 (hybrid mode)")
    p_mifc.add_argument("--output", "-o", help="Save results to JSON file")
    p_mifc.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    p_mifc.set_defaults(func=_cmd_match_ifc)

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
