"""Real-world benchmark: IFC materials from M-BIN-TW-51.ifc vs full KBOB database.

Unlike benchmark_full.py which used synthetic English candidates (10 per test),
this benchmark uses the REAL 322-entry KBOB database from lcadata.ch and
REAL IFC material names extracted from an actual Revit model.

This exposes the true difficulty of the matching problem:
- Revit naming conventions (_XXX_wg, underscores, abbreviations)
- German → German matching (no language bridge)
- 322 candidates instead of 10
- No "easy" distractors -- many genuinely similar entries
"""

import os
import time
import json
from pathlib import Path

# Load API keys from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from ifc_matching.engines.embedding import (
    EmbeddingMatcher,
    DEFAULT_MODEL,
    MULTILINGUAL_MINI_MODEL,
    LEGACY_MODEL,
)
from ifc_matching.engines.reranker import LLMReranker
from ifc_matching.engines.hybrid import HybridMatcher
from ifc_matching.databases.kbob_client import KBOBClient

# ---------------------------------------------------------------------------
# Real-world test cases: IFC material queries from M-BIN-TW-51.ifc
# with manually verified ground truth against ACTUAL KBOB entries
# ---------------------------------------------------------------------------

REAL_TEST_CASES = [
    {
        "query": "_Beton_C30-37_wg",
        "element_type": "IfcSlab",
        "expected": "Hochbaubeton (ohne Bewehrung)",
        "acceptable_ids": ["01.002"],  # any 01.002.xx variant is correct
        "note": "C30/37 is standard structural concrete. KBOB uses 'Hochbaubeton' not strength classes.",
    },
    {
        "query": "_Beton_C30-37_wg",
        "element_type": "IfcWall",
        "expected": "Hochbaubeton (ohne Bewehrung)",
        "acceptable_ids": ["01.002"],
        "note": "Same concrete, different element type -- should still match Hochbaubeton.",
    },
    {
        "query": "_Beton_C30-37_Erdebebenwände_wg",
        "element_type": "IfcWall",
        "expected": "Hochbaubeton (ohne Bewehrung)",
        "acceptable_ids": ["01.002"],
        "note": "Seismic walls use same concrete. 'Erdebebenwände' is usage context, not material.",
    },
    {
        "query": "_Beton_C50-60_vorfabrizierte_Elemente_wg",
        "element_type": "IfcSlab",
        "expected": "Betonfertigteil, hochfester Beton, ab Werk",
        "acceptable_ids": ["01.041", "01.042"],  # hochfest or normal prefab
        "note": "C50/60 + 'vorfabrizierte Elemente' = prefab high-strength concrete.",
    },
    {
        "query": "_Beton_C50-60_vorfabrizierte_Elemente_wg",
        "element_type": "IfcWall",
        "expected": "Betonfertigteil, hochfester Beton, ab Werk",
        "acceptable_ids": ["01.041", "01.042"],
        "note": "Prefab concrete wall element.",
    },
    {
        "query": "_Mauerwerk - Kalksandstein_wg",
        "element_type": "IfcWall",
        "expected": "Kalksandstein",
        "acceptable_ids": ["02.002"],  # any 02.002.xx variant
        "note": "Direct material name match. KBOB has Kalksandstein + supplier variants.",
    },
    {
        "query": "_Wärmedaemmung_druckfest_wg",
        "element_type": "IfcSlab",
        "expected": "Polystyrol extrudiert (XPS)",
        "acceptable_ids": ["10.004", "10.005", "10.006", "10.007"],
        # EPS, XPS, PUR/PIR, Schaumglas -- all valid rigid/druckfest insulation
        "note": "Pressure-resistant insulation under slabs. XPS most common but EPS/PUR also valid.",
    },
    {
        "query": "_Wärmedaemmung_druckfest_wg",
        "element_type": "IfcWall",
        "expected": "Polystyrol expandiert (EPS)",
        "acceptable_ids": ["10.004", "10.005", "10.006"],
        # EPS, XPS, PUR/PIR -- wall insulation
        "note": "Pressure-resistant wall insulation. EPS is most common for facades.",
    },
]


def load_kbob_candidates():
    """Load real KBOB database and build candidate list + ID lookup."""
    client = KBOBClient(cache_path=".kbob_cache.json")
    materials = client.get_all_materials()
    candidates = [m.name_de for m in materials]
    id_lookup = {m.name_de: m.id for m in materials}
    return candidates, id_lookup


def check_match(result_name, id_lookup, test_case):
    """Check if a match is correct (strict or acceptable)."""
    if not result_name:
        return False, False

    matched_id = id_lookup.get(result_name, "")

    # Strict: exact expected name
    strict = result_name == test_case["expected"]

    # Acceptable: ID prefix matches any acceptable prefix
    acceptable = any(
        matched_id.startswith(prefix) for prefix in test_case["acceptable_ids"]
    )

    return strict, acceptable


def run_approach(name, matcher, test_cases, candidates, id_lookup):
    """Run all test cases with a given matcher against real KBOB candidates."""
    results = []
    correct_strict = 0
    correct_acceptable = 0
    total_time = 0.0

    for i, tc in enumerate(test_cases, 1):
        t0 = time.time()
        match_results = matcher.match(
            tc["query"],
            candidates,
            top_k=5,
            element_type=tc["element_type"],
        )
        elapsed = time.time() - t0
        total_time += elapsed

        top = match_results[0] if match_results else None
        top_name = top.candidate if top else None
        top_score = top.score if top else 0.0
        top_id = id_lookup.get(top_name, "???") if top_name else "???"

        strict, acceptable = check_match(top_name, id_lookup, tc)
        if strict:
            correct_strict += 1
        if acceptable:
            correct_acceptable += 1

        status = "STRICT" if strict else ("OK" if acceptable else "FAIL")
        results.append({
            "test": i,
            "query": tc["query"],
            "element_type": tc["element_type"],
            "expected": tc["expected"],
            "got": top_name,
            "got_id": top_id,
            "score": top_score,
            "strict": strict,
            "acceptable": acceptable,
            "status": status,
            "top5": [
                {"name": r.candidate, "id": id_lookup.get(r.candidate, "???"), "score": r.score}
                for r in match_results[:5]
            ],
            "time": elapsed,
        })

    n = len(test_cases)
    return {
        "name": name,
        "results": results,
        "strict_accuracy": correct_strict / n if n else 0,
        "acceptable_accuracy": correct_acceptable / n if n else 0,
        "correct_strict": correct_strict,
        "correct_acceptable": correct_acceptable,
        "total": n,
        "total_time": total_time,
    }


def print_results(approach):
    """Print detailed results for one approach."""
    print(f"\n{'='*90}")
    print(f"  {approach['name']}")
    print(f"  Strict: {approach['correct_strict']}/{approach['total']} "
          f"({approach['strict_accuracy']:.0%})  |  "
          f"Acceptable: {approach['correct_acceptable']}/{approach['total']} "
          f"({approach['acceptable_accuracy']:.0%})  |  "
          f"Time: {approach['total_time']:.1f}s")
    print(f"{'='*90}")

    for r in approach["results"]:
        icon = "✓" if r["strict"] else ("~" if r["acceptable"] else "✗")
        print(f"\n  {icon} Test {r['test']}: {r['query']!r} ({r['element_type']})")
        print(f"    Expected: {r['expected']}")
        print(f"    Got:      {r['got']} [{r['got_id']}] (score={r['score']:.4f})")
        if not r["acceptable"]:
            print(f"    Top 5:")
            for j, t in enumerate(r["top5"], 1):
                marker = " <<" if any(t["id"].startswith(p) for p in
                    REAL_TEST_CASES[r["test"]-1]["acceptable_ids"]) else ""
                print(f"      {j}. [{t['id']:<10s}] {t['name']:<55s} {t['score']:.4f}{marker}")


def main():
    print("REAL-WORLD BENCHMARK: IFC materials vs full KBOB database (322 entries)")
    print("="*90)

    # Load real KBOB data
    print("\nLoading KBOB database from lcadata.ch...")
    candidates, id_lookup = load_kbob_candidates()
    print(f"  {len(candidates)} real KBOB candidates loaded")

    all_approaches = []

    # -----------------------------------------------------------------------
    # 1. Embedding-only: Legacy MiniLM-L6 (English)
    # -----------------------------------------------------------------------
    print(f"\n--- Loading {LEGACY_MODEL} ---")
    emb_legacy = EmbeddingMatcher(
        model_name=LEGACY_MODEL,
        backend="sentence-transformers",
        use_token_matching=True,
    )
    emb_legacy.match("warmup", candidates[:5], top_k=1)
    result = run_approach("Embedding: MiniLM-L6 (legacy, English)", emb_legacy, REAL_TEST_CASES, candidates, id_lookup)
    print_results(result)
    all_approaches.append(result)

    # -----------------------------------------------------------------------
    # 2. Embedding-only: Multilingual MiniLM
    # -----------------------------------------------------------------------
    print(f"\n--- Loading {MULTILINGUAL_MINI_MODEL} ---")
    emb_multi = EmbeddingMatcher(
        model_name=MULTILINGUAL_MINI_MODEL,
        backend="sentence-transformers",
        use_token_matching=True,
    )
    emb_multi.match("warmup", candidates[:5], top_k=1)
    result = run_approach("Embedding: Multilingual-MiniLM", emb_multi, REAL_TEST_CASES, candidates, id_lookup)
    print_results(result)
    all_approaches.append(result)

    # -----------------------------------------------------------------------
    # 3. Embedding-only: BGE-M3
    # -----------------------------------------------------------------------
    print(f"\n--- Loading {DEFAULT_MODEL} ---")
    emb_bge = EmbeddingMatcher(
        model_name=DEFAULT_MODEL,
        backend="sentence-transformers",
        use_token_matching=True,
    )
    emb_bge.match("warmup", candidates[:5], top_k=1)
    result = run_approach("Embedding: BGE-M3", emb_bge, REAL_TEST_CASES, candidates, id_lookup)
    print_results(result)
    all_approaches.append(result)

    # -----------------------------------------------------------------------
    # 4. Hybrid: BGE-M3 + Claude
    # -----------------------------------------------------------------------
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n--- Hybrid: BGE-M3 + Claude claude-opus-4-6 ---")
        hybrid_claude = HybridMatcher(
            retriever=EmbeddingMatcher(
                model_name=DEFAULT_MODEL,
                backend="sentence-transformers",
                use_token_matching=True,
            ),
            reranker=LLMReranker(
                model_name="claude-opus-4-6",
                backend="anthropic",
                temperature=0.0,
            ),
            retrieval_k=15,
        )
        result = run_approach("Hybrid: BGE-M3 + Claude claude-opus-4-6", hybrid_claude, REAL_TEST_CASES, candidates, id_lookup)
        print_results(result)
        all_approaches.append(result)
    else:
        print("\n  [SKIP] Hybrid Claude -- no ANTHROPIC_API_KEY")

    # -----------------------------------------------------------------------
    # 5. Hybrid: BGE-M3 + GPT-5.2
    # -----------------------------------------------------------------------
    if os.environ.get("OPENAI_API_KEY"):
        print("\n--- Hybrid: BGE-M3 + GPT-5.2 ---")
        hybrid_gpt = HybridMatcher(
            retriever=EmbeddingMatcher(
                model_name=DEFAULT_MODEL,
                backend="sentence-transformers",
                use_token_matching=True,
            ),
            reranker=LLMReranker(
                model_name="gpt-5.2",
                backend="openai-responses",
                temperature=0.0,
            ),
            retrieval_k=15,
        )
        result = run_approach("Hybrid: BGE-M3 + GPT-5.2", hybrid_gpt, REAL_TEST_CASES, candidates, id_lookup)
        print_results(result)
        all_approaches.append(result)
    else:
        print("\n  [SKIP] Hybrid GPT-5.2 -- no OPENAI_API_KEY")

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY: REAL-WORLD BENCHMARK ({len(candidates)} KBOB candidates)")
    print(f"{'='*90}")
    print(f"\n  {'Approach':<45s} {'Strict':>8s} {'Accept':>8s} {'Time':>7s}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*7}")
    for a in all_approaches:
        print(f"  {a['name']:<45s} {a['strict_accuracy']:>7.0%} {a['acceptable_accuracy']:>7.0%} {a['total_time']:>6.1f}s")

    # Per-test comparison
    print(f"\n  Per-test breakdown:")
    print(f"  {'Test':<50s}", end="")
    for a in all_approaches:
        short = a["name"].split(":")[0][:8] if ":" in a["name"] else a["name"][:8]
        print(f" {short:>8s}", end="")
    print()

    for i, tc in enumerate(REAL_TEST_CASES):
        label = f"{tc['query'][:35]} ({tc['element_type'][-6:]})"
        print(f"  {label:<50s}", end="")
        for a in all_approaches:
            r = a["results"][i]
            icon = "STRICT" if r["strict"] else ("OK" if r["acceptable"] else "FAIL")
            print(f" {icon:>8s}", end="")
        print()

    # Comparison with synthetic benchmark
    print(f"\n\n{'='*90}")
    print(f"  SYNTHETIC vs REAL-WORLD COMPARISON")
    print(f"{'='*90}")
    print(f"""
  Our synthetic benchmark (benchmark_full.py) reported 100% accuracy.
  This real-world benchmark exposes the truth.

  Key differences:
  ┌────────────────────────┬──────────────────────────────┬──────────────────────────────┐
  │                        │ Synthetic benchmark          │ Real-world benchmark         │
  ├────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Candidates             │ 10 hand-picked (English)     │ 322 real KBOB (German)       │
  │ Queries                │ Clean English/German         │ Revit conventions (_xxx_wg)  │
  │ Language               │ German → English             │ German → German              │
  │ Distractors            │ Obviously wrong ones         │ Many genuinely similar       │
  │ Ground truth source    │ Invented                     │ Author-assigned KBOB IDs     │
  └────────────────────────┴──────────────────────────────┴──────────────────────────────┘

  The synthetic benchmark was useful for comparing RELATIVE model performance,
  but its ABSOLUTE accuracy numbers were misleading.
""")

    # Save results
    output = {
        "benchmark": "real-world",
        "kbob_candidates": len(candidates),
        "test_cases": len(REAL_TEST_CASES),
        "ifc_source": "M-BIN-TW-51.ifc",
        "approaches": [
            {
                "name": a["name"],
                "strict_accuracy": a["strict_accuracy"],
                "acceptable_accuracy": a["acceptable_accuracy"],
                "total_time": a["total_time"],
                "results": a["results"],
            }
            for a in all_approaches
        ],
    }
    out_path = Path("benchmark_real_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
