"""ÖKOBAUDAT benchmark: same 56 IFC queries matched against ÖKOBAUDAT search API.

Full independent pipeline:
1. Preprocess each IFC query
2. Search ÖKOBAUDAT per-query (simulating real usage with millions of EPDs)
3. Embedding-match search results with BGE-M3
4. Score against ÖKOBAUDAT-specific ground truth (classific_id prefixes)
5. Report accuracy by category and language

This validates the same matching approach from FINDINGS.md against a second,
much larger data source.
"""

import os
import time
import json
from pathlib import Path

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from ifc_matching.engines.embedding import EmbeddingMatcher, DEFAULT_MODEL
from ifc_matching.databases.oekobaudat_client import OekobaudatClient
from ifc_matching.preprocessing import (
    preprocess_query,
    extract_oekobaudat_search_terms,
    filter_candidates_by_element_type,
)

# ---------------------------------------------------------------------------
# ÖKOBAUDAT category mapping (from OEKOBAU.DAT_Categories.xml)
#
#   1   Mineralische Baustoffe
#     1.1  Bindemittel (Zement, Kalk, Gips, Lehm)
#     1.2  Zuschläge (Sand, Naturstein, Blähton, ...)
#     1.3  Steine und Elemente (Kalksandstein, Ziegel, Porenbeton,
#          Betonfertigteile, Fliesen, Gipsplatten, Faserzement, ...)
#     1.4  Mörtel und Beton (Beton, Mauermörtel, Estrich, Putz)
#   2   Dämmstoffe
#   3   Holz
#   4   Metalle
#     4.1  Stahl, 4.2  Edelstahl, 4.3  Aluminium, 4.5  Zink
#   5   Beschichtungen
#   6   Kunststoffe (incl. Abdichtungsbahnen, Folien)
#   7   Komponenten von Fenstern und Vorhangfassaden
#   9   Sonstige
#  10   Komposite
# ---------------------------------------------------------------------------

# Ground truth: same 56 IFC queries, mapped to ACCEPTABLE ÖKOBAUDAT classific_id
# prefixes.  A match is "acceptable" if the top result's classific_id starts
# with any listed prefix.

GROUND_TRUTH = [
    # === CONCRETE ===
    {"query": "Ortbeton - bewehrt", "element_type": "IfcSlab",
     "acceptable_ids": ["1.4.01", "1.3.05"], "category": "concrete", "lang": "de"},
    {"query": "Ortbeton - bewehrt Verputzt", "element_type": "IfcWall",
     "acceptable_ids": ["1.4.01", "1.3.05"], "category": "concrete", "lang": "de"},
    {"query": "Concrete", "element_type": "IfcSlab",
     "acceptable_ids": ["1.4.01", "1.3.05"], "category": "concrete", "lang": "en"},
    {"query": "Concrete - Cast In Situ", "element_type": "IfcSlab",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "en"},
    {"query": "Beton, tragend 268571148", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "Stahlbeton 65690", "element_type": "IfcSlab",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "Stahlbeton 2816491304", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "_Beton_C30-37_wg", "element_type": "IfcWall",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "_Beton_C30-37_Erdebebenwände_wg", "element_type": "IfcWall",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "_Beton_C50-60_vorfabrizierte_Elemente_wg", "element_type": "IfcSlab",
     "acceptable_ids": ["1.3.05", "1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "f2_beton ihwg_C35/45", "element_type": "IfcSlab",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "nl"},
    {"query": "f2_beton ihwg_C30/37", "element_type": "IfcFooting",
     "acceptable_ids": ["1.4.01"], "category": "concrete", "lang": "nl"},
    {"query": "f2_beton prefab_C45/55", "element_type": "IfcColumn",
     "acceptable_ids": ["1.3.05", "1.4.01"], "category": "concrete", "lang": "nl"},
    {"query": "DD prefabbeton gevel", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.05", "1.4.01"], "category": "concrete", "lang": "nl"},
    {"query": "Leichtbeton - Mauerwerk", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.03", "1.3.04", "1.4.01"], "category": "concrete", "lang": "de"},
    {"query": "Leichtbeton 102890359", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.3.03", "1.3.04", "1.4.01"], "category": "concrete", "lang": "de"},
    # === MASONRY ===
    {"query": "_Mauerwerk - Kalksandstein_wg", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.01"], "category": "masonry", "lang": "de"},
    {"query": "Kalksandstein 268899148", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.3.01"], "category": "masonry", "lang": "de"},
    {"query": "Masonry - Brick", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.3.02", "1.3.01"], "category": "masonry", "lang": "en"},
    {"query": "Masonry - Concrete Block", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.04", "1.3.05", "1.3.03", "1.4.01"], "category": "masonry", "lang": "en"},
    {"query": "Mauerwerk - Naturstein", "element_type": "IfcSlab",
     "acceptable_ids": ["1.3.08", "1.2.02"], "category": "masonry", "lang": "de"},
    # === INSULATION ===
    {"query": "_Wärmedaemmung_druckfest_wg", "element_type": "IfcSlab",
     "acceptable_ids": ["2.2", "2.3", "2.4"], "category": "insulation", "lang": "de"},
    {"query": "Dämmung - hart", "element_type": "IfcSlab",
     "acceptable_ids": ["2.2", "2.3", "2.4", "2.6"], "category": "insulation", "lang": "de"},
    {"query": "Dämmung - weich", "element_type": "IfcWall",
     "acceptable_ids": ["2.1", "2.10", "2.11", "2.12", "2.13"], "category": "insulation", "lang": "de"},
    {"query": "Isolierung, hart 275646950", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["2.2", "2.3", "2.4"], "category": "insulation", "lang": "de"},
    {"query": "Steinwolle (Rockwool Fixrock033 VS)", "element_type": "IfcWall",
     "acceptable_ids": ["2.1.03", "2.1.01", "8.1.02", "2.1"], "category": "insulation", "lang": "de"},
    {"query": "AT_Isolatie_140mm", "element_type": "IfcSlab",
     "acceptable_ids": ["2."], "category": "insulation", "lang": "nl"},
    {"query": "Insulation / Thermal Barriers - Rigid insulation", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["2.2", "2.3", "2.4"], "category": "insulation", "lang": "en"},
    {"query": "Insulation / Thermal Barriers - Semi-rigid insulation", "element_type": "IfcSlab",
     "acceptable_ids": ["2.1", "2.10", "2.11"], "category": "insulation", "lang": "en"},
    # === WOOD ===
    {"query": "Holz", "element_type": "IfcWall",
     "acceptable_ids": ["3."], "category": "wood", "lang": "de"},
    {"query": "Holz - generisch 140-100-70", "element_type": "IfcDoor",
     "acceptable_ids": ["3."], "category": "wood", "lang": "de"},
    {"query": "Wood - Dimensional Lumber", "element_type": "IfcSlab",
     "acceptable_ids": ["3.1", "3.2"], "category": "wood", "lang": "en"},
    {"query": "Wood - Sheathing - plywood", "element_type": "IfcSlab",
     "acceptable_ids": ["3.2.02", "3.2"], "category": "wood", "lang": "en"},
    {"query": "Wood - Flooring", "element_type": "IfcSlab",
     "acceptable_ids": ["3.3", "3."], "category": "wood", "lang": "en"},
    # === STEEL ===
    {"query": "Stahl", "element_type": "IfcRailing",
     "acceptable_ids": ["4.1"], "category": "steel", "lang": "de"},
    {"query": "Metal - Steel - 345 MPa", "element_type": "IfcBeam",
     "acceptable_ids": ["4.1.03", "4.1.04"], "category": "steel", "lang": "en"},
    {"query": "h2_staal_ S235", "element_type": "IfcBeam",
     "acceptable_ids": ["4.1.03", "4.1.04"], "category": "steel", "lang": "nl"},
    {"query": "h2_staal_ S355", "element_type": "IfcBeam",
     "acceptable_ids": ["4.1.03", "4.1.04"], "category": "steel", "lang": "nl"},
    {"query": "DD staal Jordahl", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["4.1"], "category": "steel", "lang": "nl"},
    {"query": "Metall - Edelstahl gebürstet", "element_type": "IfcDoor",
     "acceptable_ids": ["4.2", "8.1.03"], "category": "steel", "lang": "de"},
    {"query": "Metall - Zink", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["4.5"], "category": "steel", "lang": "de"},
    # === ALUMINIUM ===
    {"query": "Aluminium 587913947", "element_type": "IfcSlab",
     "acceptable_ids": ["4.3"], "category": "aluminium", "lang": "de"},
    # === GYPSUM / DRYWALL ===
    {"query": "Trockenbau - Gipsplatte", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.13", "1.1.03"], "category": "gypsum", "lang": "de"},
    {"query": "Trockenbau - Rigips Die Blaue RFI", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.13", "1.1.03"], "category": "gypsum", "lang": "de"},
    {"query": "Trockenbau - Rigips Die Dicke RFI", "element_type": "IfcCovering",
     "acceptable_ids": ["1.3.13", "1.1.03"], "category": "gypsum", "lang": "de"},
    {"query": "Gips 275646950", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["1.3.13", "1.1.03"], "category": "gypsum", "lang": "de"},
    {"query": "Plasterboard", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.13", "1.1.03"], "category": "gypsum", "lang": "en"},
    # === GLASS ===
    {"query": "Glas", "element_type": "IfcWindow",
     "acceptable_ids": ["7."], "category": "glass", "lang": "de"},
    {"query": "Glas - Isolierverglasung klar", "element_type": "IfcWindow",
     "acceptable_ids": ["7."], "category": "glass", "lang": "de"},
    {"query": "Glas - klar", "element_type": "IfcPlate",
     "acceptable_ids": ["7.", "1.3.16"], "category": "glass", "lang": "de"},
    # === ROOFING / MEMBRANES ===
    {"query": "Dachdeckung - Bitumenbahn", "element_type": "IfcRoof",
     "acceptable_ids": ["6.", "5.3"], "category": "membrane", "lang": "de"},
    {"query": "Roofing - EPDM Membrane", "element_type": "IfcRoof",
     "acceptable_ids": ["6.", "7.3"], "category": "membrane", "lang": "en"},
    # === FACADE ===
    {"query": "Faserzementplatte (Equitone - natura)", "element_type": "IfcWall",
     "acceptable_ids": ["1.3.12", "10."], "category": "facade", "lang": "de"},
    # === TILE ===
    {"query": "Ceramic Tile", "element_type": "IfcSlab",
     "acceptable_ids": ["1.3.07", "1.3.06"], "category": "tile", "lang": "en"},
    # === NATURAL STONE ===
    {"query": "Naturstein - Granit grau", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["1.3.08", "1.2.02"], "category": "stone", "lang": "de"},
    {"query": "Dachdeckung - Kies", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["1.2.01", "1.2"], "category": "stone", "lang": "de"},
]


def main():
    n = len(GROUND_TRUTH)
    print(f"ÖKOBAUDAT benchmark: {n} test cases (same IFC queries as KBOB benchmark)")
    print(f"  Categories: {len(set(t['category'] for t in GROUND_TRUTH))}")
    print(f"  Languages:  {sorted(set(t['lang'] for t in GROUND_TRUTH))}")
    print(f"  Pipeline:   preprocess → ÖKOBAUDAT search → BGE-M3 embedding → score")

    # --- Init ---
    oeko = OekobaudatClient(cache_path=".oeko_bench_cache.json")
    print(f"\n--- Loading BGE-M3 model ---")
    emb = EmbeddingMatcher(
        model_name=DEFAULT_MODEL,
        backend="sentence-transformers",
        use_token_matching=True,
    )
    emb.match("warmup", ["warmup candidate"], top_k=1)
    print("  Model loaded.")

    def check(result_name, classific_id, tc):
        if not result_name or result_name.startswith("(no"):
            return False
        return any(classific_id.startswith(p) for p in tc["acceptable_ids"])

    def _search_multi(oeko_client, search_terms, limit_per_term=50):
        """Search ÖKOBAUDAT with multiple terms and merge results."""
        all_results = {}  # display_name → OekobaudatProcess (dedup)
        for term in search_terms:
            hits = oeko_client.search_processes(name=term, limit=limit_per_term)
            for r in hits:
                if r.display_name not in all_results:
                    all_results[r.display_name] = r
        return list(all_results.values())

    # === Approach runner ===
    def run_approach(name, *, use_keyword_extractor, use_preprocessing):
        results = []
        ok_a = 0
        t_total = 0.0
        no_results = 0
        for tc in GROUND_TRUTH:
            raw_query = tc["query"]
            t0 = time.time()

            # --- Determine search terms ---
            if use_keyword_extractor:
                search_terms = extract_oekobaudat_search_terms(raw_query)
            elif use_preprocessing:
                search_terms = [preprocess_query(raw_query).split(" | ")[0].strip()]
            else:
                search_terms = [raw_query]
            search_terms = [t for t in search_terms if t]

            # --- Search ÖKOBAUDAT (merge results from all terms) ---
            search_results = _search_multi(oeko, search_terms, limit_per_term=50)
            candidates = [r.display_name for r in search_results]
            cid_lookup = {r.display_name: r.classific_id for r in search_results}

            if not candidates:
                no_results += 1
                dt = time.time() - t0
                t_total += dt
                results.append({
                    **tc, "search_terms": search_terms, "n_candidates": 0,
                    "got": "(no results)", "got_id": "", "score": 0,
                    "acceptable": False, "time": dt, "top3": [],
                })
                continue

            # --- Optionally filter by element type ---
            if use_preprocessing or use_keyword_extractor:
                candidates = filter_candidates_by_element_type(
                    candidates, cid_lookup, tc["element_type"], source="oekobaudat"
                )

            # --- Embedding match ---
            query_for_match = preprocess_query(raw_query) if (use_preprocessing or use_keyword_extractor) else raw_query
            mr = emb.match(query_for_match, candidates, top_k=5, element_type=tc["element_type"])

            dt = time.time() - t0
            t_total += dt

            top = mr[0] if mr else None
            top_name = top.candidate if top else ""
            top_cid = cid_lookup.get(top_name, "")
            a = check(top_name, top_cid, tc)
            if a:
                ok_a += 1

            top3 = [
                (r.candidate, cid_lookup.get(r.candidate, "?"), r.score)
                for r in mr[:3]
            ]
            results.append({
                **tc, "search_terms": search_terms,
                "n_candidates": len(candidates),
                "got": top_name, "got_id": top_cid,
                "score": top.score if top else 0,
                "acceptable": a, "time": dt, "top3": top3,
            })

        return {
            "name": name, "results": results,
            "acceptable": ok_a / n, "ok_a": ok_a, "n": n,
            "time": t_total, "no_results": no_results,
        }

    def show(a):
        print(f"\n{'='*110}")
        print(f"  {a['name']}")
        print(f"  Acceptable: {a['ok_a']}/{a['n']} ({a['acceptable']:.0%})  |  "
              f"Time: {a['time']:.1f}s  |  No results: {a['no_results']}")
        print(f"{'='*110}")
        for r in a["results"]:
            icon = "~" if r["acceptable"] else "✗"
            got_short = r["got"][:50]
            terms_str = ", ".join(r.get("search_terms", []))
            n_cand = r.get("n_candidates", "?")
            print(f"  {icon} [{r['lang']:2s}] {r['query']!r:48s} search=[{terms_str[:30]:30s}] ({n_cand:>3} cand) → {got_short[:40]:40s} [{r['got_id']}]")
            if not r["acceptable"] and r["top3"]:
                for j, (nm, cid, sc) in enumerate(r["top3"], 1):
                    flag = " <<" if any(cid.startswith(p) for p in r["acceptable_ids"]) else ""
                    print(f"        {j}. [{cid:<10s}] {nm[:55]:55s} {sc:.4f}{flag}")

    approaches = []

    # 1. Raw query → ÖKOBAUDAT search → BGE-M3 (baseline)
    print(f"\n--- Approach 1: Raw search → BGE-M3 (baseline) ---")
    r = run_approach("Raw query → search → embedding",
                     use_keyword_extractor=False, use_preprocessing=False)
    show(r)
    approaches.append(r)

    # 2. Preprocessed query → ÖKOBAUDAT search → BGE-M3 + category filter
    print(f"\n--- Approach 2: Preprocess → search → BGE-M3 + filter ---")
    r = run_approach("Preprocess → search → embedding",
                     use_keyword_extractor=False, use_preprocessing=True)
    show(r)
    approaches.append(r)

    # 3. Keyword extraction → multi-term search → BGE-M3 + filter (full pipeline)
    print(f"\n--- Approach 3: Keyword extract → multi-search → BGE-M3 + filter ---")
    r = run_approach("Keywords → multi-search → embedding",
                     use_keyword_extractor=True, use_preprocessing=False)
    show(r)
    approaches.append(r)

    # === Summary ===
    print(f"\n\n{'='*100}")
    print(f"  ÖKOBAUDAT BENCHMARK SUMMARY ({n} tests, per-query search against ~1M EPDs)")
    print(f"  Data source: ÖKOBAUDAT (EN 15804+A2, datastock 2021-II)")
    print(f"  Scoring: top-1 classific_id must match acceptable category prefix")
    print(f"{'='*100}")
    print(f"\n  {'Approach':<50s} {'Accept':>8s} {'Time':>7s} {'NoRes':>6s}")
    print(f"  {'-'*50} {'-'*8} {'-'*7} {'-'*6}")
    for a in approaches:
        print(f"  {a['name']:<50s} {a['acceptable']:>7.0%} {a['time']:>6.1f}s {a['no_results']:>5d}")

    # By category
    print(f"\n  Accuracy by material category:")
    cats = sorted(set(t["category"] for t in GROUND_TRUTH))
    for cat in cats:
        cat_tests = [t for t in GROUND_TRUTH if t["category"] == cat]
        scores = []
        for a in approaches:
            cat_results = [r for r in a["results"] if r["category"] == cat]
            ok = sum(1 for r in cat_results if r["acceptable"])
            scores.append(f"{ok}/{len(cat_tests)}")
        print(f"    {cat.upper():<14s} " + "  ".join(f"{s:>6s}" for s in scores))

    # By language
    print(f"\n  Accuracy by language:")
    for lang in sorted(set(t["lang"] for t in GROUND_TRUTH)):
        lang_tests = [t for t in GROUND_TRUTH if t["lang"] == lang]
        scores = []
        for a in approaches:
            lang_results = [r for r in a["results"] if r["lang"] == lang]
            ok = sum(1 for r in lang_results if r["acceptable"])
            scores.append(f"{ok}/{len(lang_tests)}")
        print(f"    {lang.upper():<14s} " + "  ".join(f"{s:>6s}" for s in scores))

    # Approach legend for compact table
    print(f"\n  Legend:")
    for i, a in enumerate(approaches, 1):
        print(f"    Approach {i}: {a['name']}")

    # Save
    out = {
        "source": "ÖKOBAUDAT", "compliance": "EN 15804+A2",
        "datastock": "2021-II",
        "tests": n, "approaches": [
            {k: v for k, v in a.items() if k != "results"} |
            {"results": [{k: v for k, v in r.items() if k != "top3"} for r in a["results"]]}
            for a in approaches
        ],
    }
    Path("benchmark_oekobaudat_results.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False)
    )
    print(f"\n  Results saved to benchmark_oekobaudat_results.json")


if __name__ == "__main__":
    main()
