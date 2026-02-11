"""Demonstration of enhanced IFC product matching.

This updated example shows three matching modes:
1. Original embedding-based matching (preserves legacy behaviour).
2. LLM reranking for contextual accuracy.
3. Hybrid retrieve-and-rerank pipeline (recommended).
"""

from ifc_matching.engines.embedding import EmbeddingMatcher
from ifc_matching.engines.base import MatchResult


def demo_embedding_matching() -> None:
    """Original approach: cosine similarity with sentence-transformers."""
    print("=" * 60)
    print("Stage 1: Embedding-based matching (original approach)")
    print("=" * 60)

    query = "concrete"
    candidates = ["cement", "brick", "rigid insulation", "mineral wool", "gypsum board"]

    matcher = EmbeddingMatcher(
        model_name="all-MiniLM-L6-v2",
        backend="sentence-transformers",
        use_token_matching=True,
    )

    results = matcher.match(query, candidates, top_k=5)
    _print_results(query, results)


def demo_openai_embedding() -> None:
    """Upgraded embeddings: text-embedding-3-large via OpenAI API."""
    print("=" * 60)
    print("Stage 1 (upgraded): OpenAI text-embedding-3-large")
    print("=" * 60)
    print("  Requires OPENAI_API_KEY environment variable.")
    print("  Usage:")
    print("    matcher = EmbeddingMatcher(")
    print("        model_name='text-embedding-3-large',")
    print("        backend='openai',")
    print("    )")
    print()


def demo_hybrid_matching() -> None:
    """Full hybrid pipeline: embeddings + LLM reranking."""
    print("=" * 60)
    print("Stage 2: Hybrid retrieve-and-rerank pipeline")
    print("=" * 60)
    print("  Requires API keys for embedding + reranker models.")
    print("  Usage:")
    print("    from ifc_matching.engines.hybrid import HybridMatcher")
    print("    from ifc_matching.engines.embedding import EmbeddingMatcher")
    print("    from ifc_matching.engines.reranker import LLMReranker")
    print()
    print("    matcher = HybridMatcher(")
    print("        retriever=EmbeddingMatcher('text-embedding-3-large', 'openai'),")
    print("        reranker=LLMReranker('gpt-4o'),")
    print("        retrieval_k=10,")
    print("    )")
    print("    results = matcher.match(")
    print("        'rigid insulation',")
    print("        candidates,")
    print("        element_type='IfcWall',  # physical context for reranker")
    print("    )")
    print()


def demo_database_layer() -> None:
    """Multi-database support: TOTEM, OKOBAUDAT, KBOB."""
    print("=" * 60)
    print("Database Layer: Multi-standard support")
    print("=" * 60)
    print("  from ifc_matching.databases.archetype import get_database")
    print()
    print("  db = get_database('totem')     # Belgian")
    print("  db = get_database('okobaudat') # German")
    print("  db = get_database('kbob')      # Swiss")
    print("  db.load('path/to/data.json')")
    print("  candidates = db.get_material_names()")
    print()


def demo_circularity() -> None:
    """BCI calculation and write-back."""
    from ifc_matching.databases.archetype import ConnectionType
    from ifc_matching.graph.circularity import (
        MaterialNode,
        ConnectionEdge,
        ElementAssembly,
        building_circularity_indicator,
    )

    print("=" * 60)
    print("Circularity Assessment: BCI Calculation")
    print("=" * 60)

    # Build a sample wall assembly
    wall = ElementAssembly(
        element_id="wall-001",
        element_type="IfcWall",
        name="Exterior Wall",
        materials=[
            MaterialNode(id="m1", name="Concrete", mass=500, recyclability=0.6, reuse_potential=0.2),
            MaterialNode(id="m2", name="Insulation XPS", mass=20, recyclability=0.3, reuse_potential=0.1),
            MaterialNode(id="m3", name="Gypsum Board", mass=30, recyclability=0.7, reuse_potential=0.5),
        ],
        connections=[
            ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.CHEMICAL_BOND),
            ConnectionEdge(id="c2", source_id="m2", target_id="m3", connection_type=ConnectionType.SCREWED),
        ],
    )

    slab = ElementAssembly(
        element_id="slab-001",
        element_type="IfcSlab",
        name="Floor Slab",
        materials=[
            MaterialNode(id="m4", name="Reinforced Concrete", mass=800, recyclability=0.5, reuse_potential=0.1),
            MaterialNode(id="m5", name="Floor Finish", mass=15, recyclability=0.4, reuse_potential=0.6),
        ],
        connections=[
            ConnectionEdge(id="c3", source_id="m4", target_id="m5", connection_type=ConnectionType.CHEMICAL_BOND),
        ],
    )

    elements = [wall, slab]

    for e in elements:
        print(f"\n  {e.name} ({e.element_type}):")
        print(f"    MCI = {e.material_circularity_index():.4f}")
        print(f"    DI  = {e.disassembly_index():.4f}")
        print(f"    ECI = {e.element_circularity_indicator():.4f}")

    bci = building_circularity_indicator(elements)
    print(f"\n  Building Circularity Indicator (BCI) = {bci:.4f}")

    # Simulate "what-if": upgrade wall insulation connection to bolted
    print("\n  What-if: Change wall insulation from Chemical Bond to Bolted")
    wall.connections[0] = ConnectionEdge(
        id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.BOLTED
    )
    new_bci = building_circularity_indicator(elements)
    print(f"  New BCI = {new_bci:.4f} (delta: {new_bci - bci:+.4f})")
    print()


def _print_results(query: str, results: list[MatchResult]) -> None:
    print(f"\n  Query: {query!r}")
    print(f"  {'Rank':<6}{'Candidate':<30}{'Score':<10}{'Method'}")
    print(f"  {'-'*6}{'-'*30}{'-'*10}{'-'*20}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<6}{r.candidate:<30}{r.score:<10.4f}{r.method}")
    print()


def main():
    print("\n  Enhanced IFC Product Matching v2.0")
    print("  Based on Forth (2024) with hybrid retrieve-rerank pipeline\n")

    demo_embedding_matching()
    demo_openai_embedding()
    demo_hybrid_matching()
    demo_database_layer()
    demo_circularity()


if __name__ == "__main__":
    main()
