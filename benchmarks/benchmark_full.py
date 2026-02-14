"""Full benchmark: 5-way comparison across embedding models and rerankers.

Compares:
1. Embedding-only (legacy all-MiniLM-L6-v2, English-only, 33M params)
2. Embedding-only (paraphrase-multilingual-MiniLM-L12-v2, 118M params)
3. Embedding-only (BAAI/bge-m3, 568M params, 100+ languages)
4. Hybrid: BGE-M3 embeddings + Claude claude-opus-4-6 reranker
5. Hybrid: BGE-M3 embeddings + GPT-5.2 reranker
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


# ---------------------------------------------------------------------------
# Test cases with corrected ground truth (see FINDINGS.md)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "query": "rigid insulation",
        "element_type": "IfcWall",
        "candidates": [
            "XPS board", "mineral wool", "glass wool", "EPS foam",
            "PIR board", "rock wool", "cellulose insulation", "cork board",
            "hemp fiber insulation", "aerogel blanket",
        ],
        "expected": "XPS board",
        "acceptable": ["XPS board", "EPS foam", "PIR board"],
        "note": "Ambiguous insulation subtype for walls; XPS/EPS/PIR all valid rigid options",
    },
    {
        "query": "Stahlbeton",
        "element_type": "IfcSlab",
        "candidates": [
            "Reinforced concrete C30/37", "Steel S235", "Precast concrete",
            "Lightweight concrete", "Concrete block", "Timber CLT",
            "Fiber cement board", "Cast iron", "Aluminum profile", "Brick masonry",
        ],
        "expected": "Reinforced concrete C30/37",
        "acceptable": ["Reinforced concrete C30/37", "Precast concrete"],
        "note": "Multilingual: German 'Stahlbeton' = reinforced concrete",
    },
    {
        "query": "Gipskartonplatte",
        "element_type": "IfcWall",
        "candidates": [
            "Gypsum plasterboard", "Cement board", "MDF panel",
            "OSB board", "Calcium silicate board", "Plywood sheet",
            "Fiber cement board", "Plaster coat", "Acoustic panel", "Cork panel",
        ],
        "expected": "Gypsum plasterboard",
        "acceptable": ["Gypsum plasterboard"],
        "note": "Multilingual: German 'Gipskartonplatte' = gypsum plasterboard",
    },
    {
        "query": "Dampfsperre",
        "element_type": "IfcWall",
        "candidates": [
            "Vapour barrier PE", "Waterproof membrane", "Bitumen sheet",
            "Breathable membrane", "Aluminium foil", "EPDM membrane",
            "PVC roofing membrane", "Geotextile", "Glass fibre mesh", "Radon barrier",
        ],
        "expected": "Vapour barrier PE",
        "acceptable": ["Vapour barrier PE"],
        "note": "Multilingual: German 'Dampfsperre' = vapour barrier",
    },
    {
        "query": "mineral wool 120mm",
        "element_type": "IfcRoof",
        "candidates": [
            "Rock wool insulation 120mm", "Glass wool roll 100mm",
            "Mineral fibre board", "EPS 100mm", "PUR foam 80mm",
            "Cellulose blown 120mm", "Wood fibre board 100mm",
            "Sheep wool insulation", "Perlite fill", "Vacuum insulation panel",
        ],
        "expected": "Rock wool insulation 120mm",
        "acceptable": ["Rock wool insulation 120mm", "Mineral fibre board"],
        "note": "Context: roof application + thickness matching",
    },
    {
        "query": "Fensterglas",
        "element_type": "IfcWindow",
        "candidates": [
            "Float glass 6mm", "Laminated safety glass", "Triple glazing unit",
            "Tempered glass", "Low-E coated glass", "Glass wool insulation",
            "Polycarbonate sheet", "Acrylic glass", "Mirror glass", "Glass block",
        ],
        "expected": "Float glass 6mm",
        "acceptable": ["Float glass 6mm", "Triple glazing unit", "Low-E coated glass"],
        "note": "Multilingual + context: 'Fensterglas' in an IfcWindow = window glass",
    },
    {
        "query": "concrete",
        "element_type": "IfcColumn",
        "candidates": [
            "Reinforced concrete C30/37", "Precast concrete column",
            "Steel HEB profile", "Timber glulam", "Brick pillar",
            "Concrete block", "Fiber reinforced concrete", "Lightweight concrete",
            "Cast stone", "Natural stone",
        ],
        "expected": "Precast concrete column",
        "acceptable": ["Precast concrete column", "Reinforced concrete C30/37"],
        "note": "Corrected: precast is dominant for columns; see FINDINGS.md",
    },
    {
        "query": "Holzstaender",
        "element_type": "IfcWall",
        "candidates": [
            "Timber stud C24", "Timber CLT panel", "Glulam beam",
            "OSB board", "Plywood sheet", "Steel stud",
            "MDF panel", "Hardwood decking", "Bamboo panel", "LVL beam",
        ],
        "expected": "Timber stud C24",
        "acceptable": ["Timber stud C24"],
        "note": "Multilingual: 'Holzstaender' = timber stud/frame",
    },
]

# ---------------------------------------------------------------------------
# Embedding model configurations (ordered small -> large)
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = {
    "MiniLM-L6 (legacy)": {
        "model_name": LEGACY_MODEL,
        "short": "legacy",
    },
    "Multilingual-MiniLM": {
        "model_name": MULTILINGUAL_MINI_MODEL,
        "short": "multi-mini",
    },
    "BGE-M3": {
        "model_name": DEFAULT_MODEL,
        "short": "bge-m3",
    },
}

# ---------------------------------------------------------------------------
# Reranker configurations
# ---------------------------------------------------------------------------

RERANKERS = {
    "Claude claude-opus-4-6": {
        "model_name": "claude-opus-4-6",
        "backend": "anthropic",
    },
    "GPT-5.2": {
        "model_name": "gpt-5.2",
        "backend": "openai-responses",
    },
}


def run_approach(name, matcher, test_cases):
    """Run all test cases with a given matcher, return results."""
    results = []
    correct_strict = 0
    correct_acceptable = 0

    for i, tc in enumerate(test_cases, 1):
        t0 = time.time()
        match_results = matcher.match(
            tc["query"], tc["candidates"], top_k=5, element_type=tc["element_type"],
        )
        elapsed = time.time() - t0

        top = str(match_results[0].candidate) if match_results else "N/A"
        is_strict = top == tc["expected"]
        is_acceptable = top in tc["acceptable"]
        if is_strict:
            correct_strict += 1
        if is_acceptable:
            correct_acceptable += 1

        results.append({
            "test": i,
            "query": tc["query"],
            "element_type": tc["element_type"],
            "expected": tc["expected"],
            "got": top,
            "strict_correct": is_strict,
            "acceptable_correct": is_acceptable,
            "top3": [
                {"candidate": str(r.candidate), "score": round(r.score, 4), "method": r.method}
                for r in match_results[:3]
            ],
            "time_s": round(elapsed, 2),
        })

    total = len(test_cases)
    return {
        "approach": name,
        "strict_accuracy": f"{correct_strict}/{total} ({100*correct_strict/total:.0f}%)",
        "acceptable_accuracy": f"{correct_acceptable}/{total} ({100*correct_acceptable/total:.0f}%)",
        "strict_n": correct_strict,
        "acceptable_n": correct_acceptable,
        "total": total,
        "results": results,
    }


def print_result_row(tc_idx, tc, *approach_results):
    """Print one row of comparison across all approaches."""
    query_str = f"'{tc['query']}' ({tc['element_type']})"
    print(f"\n--- Test {tc_idx}: {query_str} ---")
    print(f"    Expected: {tc['expected']}  |  Acceptable: {tc['acceptable']}")

    for r in approach_results:
        tag = "CORRECT" if r["strict_correct"] else ("~OK~" if r["acceptable_correct"] else "WRONG")
        label = r.get("_label", "???")
        print(f"    {label:<40s}: {r['got']:<35s} {tag:<8s} ({r['time_s']:.2f}s)")
        for t in r["top3"]:
            print(f"      {t['candidate']:<35s} score={t['score']:.4f}  [{t['method']}]")


def main():
    print("=" * 80)
    print("  FULL BENCHMARK: 3 Embedding Models x 2 Frontier Rerankers")
    print("=" * 80)

    # ---- Load embedding models ----
    emb_matchers = {}
    step = 0
    total_steps = len(EMBEDDING_MODELS) + len(RERANKERS)

    for label, cfg in EMBEDDING_MODELS.items():
        step += 1
        print(f"\n[{step}/{total_steps}] Loading {label} ({cfg['model_name']})...")
        matcher = EmbeddingMatcher(
            model_name=cfg["model_name"],
            backend="sentence-transformers",
            use_token_matching=True,
        )
        # Warmup to trigger model download/load
        matcher.match("warmup", ["a", "b"], top_k=1)
        emb_matchers[label] = matcher
        print(f"  -> {label} loaded OK")

    # ---- Build hybrid matchers (BGE-M3 + each reranker) ----
    hybrid_matchers = {}
    best_emb_label = "BGE-M3"
    best_emb = emb_matchers[best_emb_label]

    for label, cfg in RERANKERS.items():
        step += 1
        print(f"[{step}/{total_steps}] Setting up Hybrid ({best_emb_label}) + {label}...")
        hybrid_matchers[label] = HybridMatcher(
            retriever=EmbeddingMatcher(
                model_name=DEFAULT_MODEL,
                backend="sentence-transformers",
                use_token_matching=True,
            ),
            reranker=LLMReranker(
                model_name=cfg["model_name"],
                backend=cfg["backend"],
                temperature=0.0,
            ),
            retrieval_k=10,
        )

    # ---- Run all approaches ----
    all_summaries = {}
    ordered_keys = []

    # Embedding-only approaches
    for label, matcher in emb_matchers.items():
        key = f"emb_{EMBEDDING_MODELS[label]['short']}"
        ordered_keys.append(key)
        print(f"\nRunning embedding-only: {label}...")
        all_summaries[key] = run_approach(f"Emb: {label}", matcher, TEST_CASES)

    # Hybrid approaches
    for label, matcher in hybrid_matchers.items():
        key = f"hybrid_{label}"
        ordered_keys.append(key)
        print(f"Running Hybrid ({best_emb_label}) + {label}...")
        all_summaries[key] = run_approach(f"Hybrid({best_emb_label})+{label}", matcher, TEST_CASES)

    # ---- Friendly labels ----
    friendly = {}
    for label, cfg in EMBEDDING_MODELS.items():
        friendly[f"emb_{cfg['short']}"] = f"Emb: {label}"
    for label in RERANKERS:
        friendly[f"hybrid_{label}"] = f"Hybrid(BGE-M3) + {label}"

    # ---- Detailed results ----
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS")
    print("=" * 80)

    for i, tc in enumerate(TEST_CASES, 1):
        rows = []
        for key in ordered_keys:
            row = all_summaries[key]["results"][i - 1]
            row["_label"] = friendly[key]
            rows.append(row)
        print_result_row(i, tc, *rows)

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    header = f"  {'Approach':<45s} {'Strict':<12s} {'Acceptable':<12s}"
    print(header)
    print("  " + "-" * 67)

    for key in ordered_keys:
        s = all_summaries[key]
        print(f"  {friendly[key]:<45s} {s['strict_accuracy']:<12s} {s['acceptable_accuracy']:<12s}")

    print(f"\n  Paper baseline: <70% accuracy (embedding-only)")

    # ---- Embedding model comparison ----
    emb_keys = [k for k in ordered_keys if k.startswith("emb_")]
    print(f"\n  === Embedding Model Comparison (standalone) ===")
    for k in emb_keys:
        s = all_summaries[k]
        print(f"    {friendly[k]:<40s} strict={s['strict_n']}/8  acceptable={s['acceptable_n']}/8")

    # ---- BGE-M3 vs legacy ----
    leg = all_summaries["emb_legacy"]
    bge = all_summaries["emb_bge-m3"]
    print(f"\n  === BGE-M3 vs Legacy (embedding-only) ===")
    print(f"    Strict:     {bge['strict_n'] - leg['strict_n']:+d} ({leg['strict_n']} -> {bge['strict_n']})")
    print(f"    Acceptable: {bge['acceptable_n'] - leg['acceptable_n']:+d} ({leg['acceptable_n']} -> {bge['acceptable_n']})")

    # ---- Hybrid over BGE-M3 embedding-only ----
    print(f"\n  === Hybrid (BGE-M3) vs Embedding-only (BGE-M3) ===")
    for label in RERANKERS:
        s = all_summaries[f"hybrid_{label}"]
        ds = s["strict_n"] - bge["strict_n"]
        da = s["acceptable_n"] - bge["acceptable_n"]
        print(f"    + {label}: {ds:+d} strict, {da:+d} acceptable")

    # ---- Full pipeline vs paper baseline ----
    print(f"\n  === Full pipeline improvement over legacy baseline ===")
    for label in RERANKERS:
        s = all_summaries[f"hybrid_{label}"]
        ds = s["strict_n"] - leg["strict_n"]
        da = s["acceptable_n"] - leg["acceptable_n"]
        print(f"    Hybrid(BGE-M3)+{label}: {ds:+d} strict, {da:+d} acceptable")

    # ---- Head-to-head rerankers ----
    labels = list(RERANKERS.keys())
    if len(labels) == 2:
        a, b = labels
        sa = all_summaries[f"hybrid_{a}"]
        sb = all_summaries[f"hybrid_{b}"]
        print(f"\n  === Head-to-head: {a} vs {b} (strict) ===")
        a_wins = b_wins = ties = 0
        for i in range(len(TEST_CASES)):
            ra = sa["results"][i]
            rb = sb["results"][i]
            if ra["strict_correct"] and not rb["strict_correct"]:
                a_wins += 1
            elif rb["strict_correct"] and not ra["strict_correct"]:
                b_wins += 1
            else:
                ties += 1
        print(f"    {a} wins: {a_wins}")
        print(f"    {b} wins: {b_wins}")
        print(f"    Ties: {ties}")

    print("\n" + "=" * 80)

    # ---- Save ----
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "embedding_models": {k: v["model_name"] for k, v in EMBEDDING_MODELS.items()},
            "rerankers": {k: v["model_name"] for k, v in RERANKERS.items()},
        },
        "approaches": {key: all_summaries[key] for key in ordered_keys},
    }
    outpath = "benchmark_full_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {outpath}")


if __name__ == "__main__":
    main()
