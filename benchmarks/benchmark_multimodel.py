"""Multi-model real-world benchmark: 75 materials from 7 IFC files vs 322 KBOB entries.

Sources:
- M-BIN-TW-51.ifc (our own Swiss Revit model)
- duplex_arc.ifc (IFC-Bench, buildingSMART, English)
- fzk_house_arc.ifc (IFC-Bench, KIT, German)
- smiley_west_arc.ifc (IFC-Bench, KIT, German)
- sixty5_str.ifc (IFC-Bench, buildingSMART, Dutch/English)
- sixty5_facade.ifc (IFC-Bench, buildingSMART, Dutch)
- digital_hub_arc.ifc (IFC-Bench, RWTH Aachen, German)

Author-assigned ground truth against KBOB category IDs (not independently verified).
"""

import os, time, json
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from ifc_matching.engines.embedding import EmbeddingMatcher, DEFAULT_MODEL
from ifc_matching.engines.reranker import LLMReranker
from ifc_matching.engines.hybrid import HybridMatcher
from ifc_matching.databases.kbob_client import KBOBClient
from ifc_matching.preprocessing import preprocess_query, filter_candidates_by_element_type

# ---------------------------------------------------------------------------
# Ground truth: IFC material query → acceptable KBOB category IDs
# A match is "acceptable" if the top result's KBOB ID starts with any prefix.
# "strict_name" is the ideal KBOB name (for strict scoring).
# "unmatchable" = True means KBOB has no corresponding entry.
# ---------------------------------------------------------------------------

GROUND_TRUTH = [
    # === CONCRETE (various conventions) ===
    {"query": "Ortbeton - bewehrt", "element_type": "IfcSlab",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "Ortbeton - bewehrt Verputzt", "element_type": "IfcWall",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "Concrete", "element_type": "IfcSlab",
     "acceptable_ids": ["01.002", "01.001"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "en"},
    {"query": "Concrete - Cast In Situ", "element_type": "IfcSlab",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "en"},
    {"query": "Beton, tragend 268571148", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "Stahlbeton 65690", "element_type": "IfcSlab",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "Stahlbeton 2816491304", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "_Beton_C30-37_wg", "element_type": "IfcWall",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "_Beton_C30-37_Erdebebenwände_wg", "element_type": "IfcWall",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "de"},
    {"query": "_Beton_C50-60_vorfabrizierte_Elemente_wg", "element_type": "IfcSlab",
     "acceptable_ids": ["01.041", "01.042"], "strict_name": "Betonfertigteil, hochfester Beton, ab Werk",
     "category": "concrete", "lang": "de"},
    {"query": "f2_beton ihwg_C35/45", "element_type": "IfcSlab",
     "acceptable_ids": ["01.002"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "nl"},
    {"query": "f2_beton ihwg_C30/37", "element_type": "IfcFooting",
     "acceptable_ids": ["01.002", "01.004"], "strict_name": "Hochbaubeton (ohne Bewehrung)",
     "category": "concrete", "lang": "nl"},
    {"query": "f2_beton prefab_C45/55", "element_type": "IfcColumn",
     "acceptable_ids": ["01.041", "01.042"], "strict_name": "Betonfertigteil, hochfester Beton, ab Werk",
     "category": "concrete", "lang": "nl"},
    {"query": "DD prefabbeton gevel", "element_type": "IfcWall",
     "acceptable_ids": ["01.041", "01.042"], "strict_name": "Betonfertigteil, Normalbeton, ab Werk",
     "category": "concrete", "lang": "nl"},
    {"query": "Leichtbeton - Mauerwerk", "element_type": "IfcWall",
     "acceptable_ids": ["02.006", "02.004", "02.005"], "strict_name": "Porenbetonstein",
     "category": "concrete", "lang": "de"},
    {"query": "Leichtbeton 102890359", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["02.006", "02.004", "02.005"], "strict_name": "Porenbetonstein",
     "category": "concrete", "lang": "de"},
    # === MASONRY ===
    {"query": "_Mauerwerk - Kalksandstein_wg", "element_type": "IfcWall",
     "acceptable_ids": ["02.002"], "strict_name": "Kalksandstein",
     "category": "masonry", "lang": "de"},
    {"query": "Kalksandstein 268899148", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["02.002"], "strict_name": "Kalksandstein",
     "category": "masonry", "lang": "de"},
    {"query": "Masonry - Brick", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["02.001"], "strict_name": "Backstein",
     "category": "masonry", "lang": "en"},
    {"query": "Masonry - Concrete Block", "element_type": "IfcWall",
     "acceptable_ids": ["02.007", "03.001"], "strict_name": "Zementstein",
     "category": "masonry", "lang": "en"},
    {"query": "Mauerwerk - Naturstein", "element_type": "IfcSlab",
     "acceptable_ids": ["03.018", "03.017", "03.019"], "strict_name": "Natursteinplatte",
     "category": "masonry", "lang": "de"},
    # === INSULATION ===
    {"query": "_Wärmedaemmung_druckfest_wg", "element_type": "IfcSlab",
     "acceptable_ids": ["10.004", "10.005", "10.006", "10.007"],
     "strict_name": "Polystyrol extrudiert (XPS)",
     "category": "insulation", "lang": "de"},
    {"query": "Dämmung - hart", "element_type": "IfcSlab",
     "acceptable_ids": ["10.004", "10.005", "10.006", "10.007"],
     "strict_name": "Polystyrol extrudiert (XPS)",
     "category": "insulation", "lang": "de"},
    {"query": "Dämmung - weich", "element_type": "IfcWall",
     "acceptable_ids": ["10.001", "10.008", "10.009", "10.010"],
     "strict_name": "Glaswolle",
     "category": "insulation", "lang": "de"},
    {"query": "Isolierung, hart 275646950", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["10.004", "10.005", "10.006"],
     "strict_name": "Polystyrol expandiert (EPS)",
     "category": "insulation", "lang": "de"},
    {"query": "Steinwolle (Rockwool Fixrock033 VS)", "element_type": "IfcWall",
     "acceptable_ids": ["10.008"], "strict_name": "Steinwolle",
     "category": "insulation", "lang": "de"},
    {"query": "AT_Isolatie_140mm", "element_type": "IfcSlab",
     "acceptable_ids": ["10.004", "10.005", "10.006", "10.008", "10.001"],
     "strict_name": "Polystyrol expandiert (EPS)",
     "category": "insulation", "lang": "nl"},
    {"query": "Insulation / Thermal Barriers - Rigid insulation", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["10.004", "10.005", "10.006"],
     "strict_name": "Polystyrol extrudiert (XPS)",
     "category": "insulation", "lang": "en"},
    {"query": "Insulation / Thermal Barriers - Semi-rigid insulation", "element_type": "IfcSlab",
     "acceptable_ids": ["10.001", "10.008"],
     "strict_name": "Glaswolle",
     "category": "insulation", "lang": "en"},
    # === WOOD ===
    {"query": "Holz", "element_type": "IfcWall",
     "acceptable_ids": ["07."], "strict_name": "3- und 5-Schicht Massivholzplatte",
     "category": "wood", "lang": "de"},
    {"query": "Holz - generisch 140-100-70", "element_type": "IfcDoor",
     "acceptable_ids": ["07."], "strict_name": "3- und 5-Schicht Massivholzplatte",
     "category": "wood", "lang": "de"},
    {"query": "Wood - Dimensional Lumber", "element_type": "IfcSlab",
     "acceptable_ids": ["07."], "strict_name": "Konstruktionsholz",
     "category": "wood", "lang": "en"},
    {"query": "Wood - Sheathing - plywood", "element_type": "IfcSlab",
     "acceptable_ids": ["07.012", "07."], "strict_name": "Sperrholzplatte",
     "category": "wood", "lang": "en"},
    {"query": "Wood - Flooring", "element_type": "IfcSlab",
     "acceptable_ids": ["07.", "11.013", "11.014"], "strict_name": "Parkett Mehrschicht",
     "category": "wood", "lang": "en"},
    # === STEEL ===
    {"query": "Stahl", "element_type": "IfcRailing",
     "acceptable_ids": ["06."], "strict_name": "Stahlprofil",
     "category": "steel", "lang": "de"},
    {"query": "Metal - Steel - 345 MPa", "element_type": "IfcBeam",
     "acceptable_ids": ["06.004", "06.005"], "strict_name": "Stahlprofil",
     "category": "steel", "lang": "en"},
    {"query": "h2_staal_ S235", "element_type": "IfcBeam",
     "acceptable_ids": ["06.004", "06.005"], "strict_name": "Stahlprofil",
     "category": "steel", "lang": "nl"},
    {"query": "h2_staal_ S355", "element_type": "IfcBeam",
     "acceptable_ids": ["06.004", "06.005"], "strict_name": "Stahlprofil",
     "category": "steel", "lang": "nl"},
    {"query": "DD staal Jordahl", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["06.004", "06.005", "06.003"], "strict_name": "Stahlprofil",
     "category": "steel", "lang": "nl"},
    {"query": "Metall - Edelstahl gebürstet", "element_type": "IfcDoor",
     "acceptable_ids": ["06."], "strict_name": "Edelstahl",
     "category": "steel", "lang": "de"},
    {"query": "Metall - Zink", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["06.010"], "strict_name": "Zinkblech",
     "category": "steel", "lang": "de"},
    # === ALUMINIUM ===
    {"query": "Aluminium 587913947", "element_type": "IfcSlab",
     "acceptable_ids": ["06.001", "06.002"], "strict_name": "Aluminiumblech, blank",
     "category": "aluminium", "lang": "de"},
    # === GYPSUM / DRYWALL ===
    {"query": "Trockenbau - Gipsplatte", "element_type": "IfcWall",
     "acceptable_ids": ["03.006", "03.007"], "strict_name": "Gipskartonplatte",
     "category": "gypsum", "lang": "de"},
    {"query": "Trockenbau - Rigips Die Blaue RFI", "element_type": "IfcWall",
     "acceptable_ids": ["03.006", "03.007"], "strict_name": "Gipskartonplatte",
     "category": "gypsum", "lang": "de"},
    {"query": "Trockenbau - Rigips Die Dicke RFI", "element_type": "IfcCovering",
     "acceptable_ids": ["03.006", "03.007"], "strict_name": "Gipskartonplatte",
     "category": "gypsum", "lang": "de"},
    {"query": "Gips 275646950", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["03.006", "03.007", "04.001"], "strict_name": "Gipskartonplatte",
     "category": "gypsum", "lang": "de"},
    {"query": "Plasterboard", "element_type": "IfcWall",
     "acceptable_ids": ["03.006", "03.007"], "strict_name": "Gipskartonplatte",
     "category": "gypsum", "lang": "en"},
    # === GLASS ===
    {"query": "Glas", "element_type": "IfcWindow",
     "acceptable_ids": ["05.001", "05.002", "05.003", "05.009", "05.010"],
     "strict_name": "Isolierverglasung 2-fach, Ug-Wert 1.1 W/m2K, Dicke 24 mm 3",
     "category": "glass", "lang": "de"},
    {"query": "Glas - Isolierverglasung klar", "element_type": "IfcWindow",
     "acceptable_ids": ["05.001", "05.002", "05.003", "05.009", "05.010", "05.011", "05.012"],
     "strict_name": "Isolierverglasung 2-fach, Ug-Wert 1.1 W/m2K, Dicke 24 mm 3",
     "category": "glass", "lang": "de"},
    {"query": "Glas - klar", "element_type": "IfcPlate",
     "acceptable_ids": ["05.001", "05.002", "05.003", "05.009", "05.010"],
     "strict_name": "Isolierverglasung 2-fach, Ug-Wert 1.1 W/m2K, Dicke 24 mm 3",
     "category": "glass", "lang": "de"},
    # === ROOFING / MEMBRANES ===
    {"query": "Dachdeckung - Bitumenbahn", "element_type": "IfcRoof",
     "acceptable_ids": ["09.003"], "strict_name": "Dichtungsbahn bituminös",
     "category": "membrane", "lang": "de"},
    {"query": "Roofing - EPDM Membrane", "element_type": "IfcRoof",
     "acceptable_ids": ["09.004", "09.005"], "strict_name": "Dichtungsbahn Kunststoff (FPO/TPO)",
     "category": "membrane", "lang": "en"},
    # === FACADE ===
    {"query": "Faserzementplatte (Equitone - natura)", "element_type": "IfcWall",
     "acceptable_ids": ["03.003", "05.019", "05.020"], "strict_name": "Faserzementplatte gross",
     "category": "facade", "lang": "de"},
    # === TILE ===
    {"query": "Ceramic Tile", "element_type": "IfcSlab",
     "acceptable_ids": ["11.008", "11.007"], "strict_name": "Keramikplatte glasiert, 8.5 mm",
     "category": "tile", "lang": "en"},
    # === NATURAL STONE ===
    {"query": "Naturstein - Granit grau", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["03.018", "03.019"], "strict_name": "Natursteinplatte",
     "category": "stone", "lang": "de"},
    {"query": "Dachdeckung - Kies", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["03.009", "03.010"], "strict_name": "Kies",
     "category": "stone", "lang": "de"},
]

def main():
    print(f"Multi-model benchmark: {len(GROUND_TRUTH)} test cases from IFC-Bench + own model")
    print(f"Categories: {len(set(t['category'] for t in GROUND_TRUTH))}")
    print(f"Languages: {set(t['lang'] for t in GROUND_TRUTH)}")

    # Load KBOB
    print("\nLoading KBOB database...")
    client = KBOBClient(cache_path=".kbob_cache.json")
    kbob_mats = client.get_all_materials()
    candidates = [m.name_de for m in kbob_mats]
    id_lookup = {m.name_de: m.id for m in kbob_mats}
    print(f"  {len(candidates)} KBOB candidates")

    def check(result_name, tc):
        if not result_name:
            return False, False
        mid = id_lookup.get(result_name, "")
        strict = result_name == tc["strict_name"]
        acceptable = any(mid.startswith(p) for p in tc["acceptable_ids"])
        return strict, acceptable

    def run(name, matcher, tests, cands, *, use_preprocessing=False):
        results = []
        ok_s, ok_a = 0, 0
        t_total = 0.0
        for tc in tests:
            query = tc["query"]
            c = cands
            if use_preprocessing:
                query = preprocess_query(query)
                c = filter_candidates_by_element_type(cands, id_lookup, tc["element_type"])
            t0 = time.time()
            mr = matcher.match(query, c, top_k=5, element_type=tc["element_type"])
            dt = time.time() - t0
            t_total += dt
            top = mr[0] if mr else None
            top_name = top.candidate if top else ""
            top_id = id_lookup.get(top_name, "???")
            s, a = check(top_name, tc)
            if s: ok_s += 1
            if a: ok_a += 1
            status = "STRICT" if s else ("OK" if a else "FAIL")
            results.append({**tc, "got": top_name, "got_id": top_id, "score": top.score if top else 0,
                            "strict": s, "acceptable": a, "status": status, "time": dt,
                            "top3": [(r.candidate, id_lookup.get(r.candidate, "?"), r.score) for r in mr[:3]],
                            "preprocessed_query": query if use_preprocessing else None})
        n = len(tests)
        return {"name": name, "results": results, "strict": ok_s/n, "acceptable": ok_a/n,
                "ok_s": ok_s, "ok_a": ok_a, "n": n, "time": t_total}

    def show(a):
        print(f"\n{'='*90}")
        print(f"  {a['name']}")
        print(f"  Strict: {a['ok_s']}/{a['n']} ({a['strict']:.0%})  |  "
              f"Acceptable: {a['ok_a']}/{a['n']} ({a['acceptable']:.0%})  |  Time: {a['time']:.1f}s")
        print(f"{'='*90}")
        for r in a["results"]:
            icon = "✓" if r["strict"] else ("~" if r["acceptable"] else "✗")
            print(f"  {icon} [{r['lang']:2s}] {r['query']!r:50s} → {r['got'][:45]:45s} [{r['got_id']}]")
            if not r["acceptable"]:
                for j, (n, i, s) in enumerate(r["top3"], 1):
                    flag = " <<" if any(i.startswith(p) for p in r["acceptable_ids"]) else ""
                    print(f"        {j}. [{i:<10s}] {n[:50]:50s} {s:.4f}{flag}")

    approaches = []

    # Shared embedding model instance (avoid loading BGE-M3 multiple times)
    print(f"\n--- Loading BGE-M3 model ---")
    emb = EmbeddingMatcher(model_name=DEFAULT_MODEL, backend="sentence-transformers", use_token_matching=True)
    emb.match("warmup", candidates[:5], top_k=1)
    print("  Model loaded.")

    # 1. Embedding BGE-M3 (baseline, no preprocessing)
    print(f"\n--- BGE-M3 Embedding (baseline) ---")
    r = run("Embedding: BGE-M3", emb, GROUND_TRUTH, candidates, use_preprocessing=False)
    show(r)
    approaches.append(r)

    # 2. Embedding BGE-M3 + preprocessing
    print(f"\n--- BGE-M3 + Preprocessing ---")
    r = run("Embedding + Preprocess", emb, GROUND_TRUTH, candidates, use_preprocessing=True)
    show(r)
    approaches.append(r)

    # 3. Hybrid BGE-M3 + Claude (baseline)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\n--- Hybrid: BGE-M3 + Claude ---")
        hybrid_c = HybridMatcher(
            retriever=emb,
            reranker=LLMReranker(model_name="claude-opus-4-6", backend="anthropic", temperature=0.0),
            retrieval_k=15,
        )
        r = run("Hybrid: BGE-M3 + Claude", hybrid_c, GROUND_TRUTH, candidates, use_preprocessing=False)
        show(r)
        approaches.append(r)

        # 4. Hybrid BGE-M3 + Claude + preprocessing
        print(f"\n--- Hybrid: BGE-M3 + Claude + Preprocess ---")
        r = run("Hybrid + Claude + Preprocess", hybrid_c, GROUND_TRUTH, candidates, use_preprocessing=True)
        show(r)
        approaches.append(r)

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  MULTI-MODEL BENCHMARK SUMMARY ({len(GROUND_TRUTH)} tests, {len(candidates)} KBOB candidates)")
    print(f"  Sources: 7 IFC files (IFC-Bench + own), 4 languages, {len(set(t['category'] for t in GROUND_TRUTH))} material categories")
    print(f"{'='*90}")
    print(f"\n  {'Approach':<42s} {'Strict':>8s} {'Accept':>8s} {'Time':>7s}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*7}")
    for a in approaches:
        print(f"  {a['name']:<42s} {a['strict']:>7.0%} {a['acceptable']:>7.0%} {a['time']:>6.1f}s")

    # By category
    print(f"\n  By category:")
    cats = sorted(set(t["category"] for t in GROUND_TRUTH))
    for cat in cats:
        cat_tests = [t for t in GROUND_TRUTH if t["category"] == cat]
        print(f"\n  {cat.upper()} ({len(cat_tests)} tests):")
        for a in approaches:
            cat_results = [r for r in a["results"] if r["category"] == cat]
            ok = sum(1 for r in cat_results if r["acceptable"])
            print(f"    {a['name']:<40s} {ok}/{len(cat_tests)} acceptable")

    # By language
    print(f"\n  By language:")
    for lang in sorted(set(t["lang"] for t in GROUND_TRUTH)):
        lang_tests = [t for t in GROUND_TRUTH if t["lang"] == lang]
        print(f"\n  {lang.upper()} ({len(lang_tests)} tests):")
        for a in approaches:
            lang_results = [r for r in a["results"] if r["lang"] == lang]
            ok = sum(1 for r in lang_results if r["acceptable"])
            print(f"    {a['name']:<40s} {ok}/{len(lang_tests)} acceptable")

    # Save
    out = {"tests": len(GROUND_TRUTH), "candidates": len(candidates),
           "approaches": [{k: v for k, v in a.items() if k != "results"} | 
                          {"results": [{k: v for k, v in r.items() if k != "top3"} for r in a["results"]]}
                          for a in approaches]}
    Path("benchmark_multimodel_v2_results.json").write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(f"\n  Results saved to benchmark_multimodel_v2_results.json")


if __name__ == "__main__":
    main()
