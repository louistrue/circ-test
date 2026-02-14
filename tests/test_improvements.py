"""Quick sample test: 8 diverse cases covering the worst failures.

Compares:
- BEFORE: raw query → embedding → match
- AFTER:  clean → expand → filter → embedding → match

Uses embedding-only (no LLM API calls) for speed.
"""

import os, time
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from ifc_matching.engines.embedding import EmbeddingMatcher, DEFAULT_MODEL
from ifc_matching.databases.kbob_client import KBOBClient
from ifc_matching.preprocessing import (
    clean_ifc_query,
    expand_query_with_synonyms,
    preprocess_query,
    filter_candidates_by_element_type,
)

# 8 representative test cases (each was a failure or borderline in the benchmark)
SAMPLE_TESTS = [
    # 1. Revit noise + insulation (TOTAL FAILURE before)
    {"query": "_Wärmedaemmung_druckfest_wg", "element_type": "IfcSlab",
     "acceptable_ids": ["10.004", "10.005", "10.006", "10.007"],
     "category": "insulation", "lang": "de"},
    # 2. Dutch steel (FAILURE before)
    {"query": "h2_staal_ S235", "element_type": "IfcBeam",
     "acceptable_ids": ["06.004", "06.005"],
     "category": "steel", "lang": "nl"},
    # 3. English insulation (FAILURE before)
    {"query": "Insulation / Thermal Barriers - Rigid insulation", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["10.004", "10.005", "10.006"],
     "category": "insulation", "lang": "en"},
    # 4. German wood (FAILURE before — matched doors)
    {"query": "Holz - generisch 140-100-70", "element_type": "IfcDoor",
     "acceptable_ids": ["07."],
     "category": "wood", "lang": "de"},
    # 5. English plasterboard (FAILURE before)
    {"query": "Plasterboard", "element_type": "IfcWall",
     "acceptable_ids": ["03.006", "03.007", "03.008"],
     "category": "gypsum", "lang": "en"},
    # 6. Dutch precast concrete (FAILURE before)
    {"query": "DD prefabbeton gevel", "element_type": "IfcWall",
     "acceptable_ids": ["01.041", "01.042"],
     "category": "concrete", "lang": "nl"},
    # 7. German steel with brand noise (FAILURE before)
    {"query": "Metall - Zink", "element_type": "IfcBuildingElementProxy",
     "acceptable_ids": ["06.010"],
     "category": "steel", "lang": "de"},
    # 8. ArchiCAD numeric ID noise (worked before, sanity check)
    {"query": "Kalksandstein 268899148", "element_type": "IfcWallStandardCase",
     "acceptable_ids": ["02.002"],
     "category": "masonry", "lang": "de"},
]

print("Loading KBOB database...")
client = KBOBClient(cache_path=".kbob_cache.json")
kbob_mats = client.get_all_materials()
all_candidates = [m.name_de for m in kbob_mats]
id_lookup = {m.name_de: m.id for m in kbob_mats}
print(f"  {len(all_candidates)} KBOB candidates loaded")

print("\n" + "="*90)
print("  PREPROCESSING DEMO")
print("="*90)
for tc in SAMPLE_TESTS:
    raw = tc["query"]
    cleaned = clean_ifc_query(raw)
    expanded = expand_query_with_synonyms(cleaned)
    print(f"\n  [{tc['lang']:2s}] {raw!r}")
    print(f"    → cleaned:  {cleaned!r}")
    print(f"    → expanded: {expanded!r}")
    
    filtered = filter_candidates_by_element_type(all_candidates, id_lookup, tc["element_type"])
    print(f"    → filtered: {len(filtered)}/{len(all_candidates)} candidates for {tc['element_type']}")


print("\n\nLoading BGE-M3 model...")
emb = EmbeddingMatcher(model_name=DEFAULT_MODEL, backend="sentence-transformers", use_token_matching=True)
emb.match("warmup", all_candidates[:5], top_k=1)  # warmup
print("  Model loaded.\n")


def check(result_name, tc):
    mid = id_lookup.get(result_name, "")
    return any(mid.startswith(p) for p in tc["acceptable_ids"])


def run_test(label, query, candidates, tc, top_k=5):
    """Run a single test and return results."""
    t0 = time.time()
    results = emb.match(query, candidates, top_k=top_k, element_type=tc["element_type"])
    dt = time.time() - t0
    top = results[0] if results else None
    top_name = top.candidate if top else ""
    top_id = id_lookup.get(top_name, "???")
    ok = check(top_name, tc)
    return {
        "label": label, "got": top_name, "got_id": top_id,
        "ok": ok, "time": dt, "top3": results[:3],
    }


# Run comparisons
print("="*90)
print("  BEFORE vs AFTER (embedding-only, no LLM)")
print("="*90)

before_ok = 0
after_ok = 0

for tc in SAMPLE_TESTS:
    raw = tc["query"]
    
    # BEFORE: raw query, all candidates
    before = run_test("BEFORE", raw, all_candidates, tc)
    
    # AFTER: preprocessed query, filtered candidates
    processed = preprocess_query(raw)
    filtered = filter_candidates_by_element_type(all_candidates, id_lookup, tc["element_type"])
    after = run_test("AFTER", processed, filtered, tc)
    
    if before["ok"]: before_ok += 1
    if after["ok"]: after_ok += 1
    
    icon_b = "✓" if before["ok"] else "✗"
    icon_a = "✓" if after["ok"] else "✗"
    changed = "IMPROVED" if (after["ok"] and not before["ok"]) else \
              "REGRESSED" if (before["ok"] and not after["ok"]) else \
              "same"
    
    print(f"\n  [{tc['lang']:2s}] {raw!r}")
    print(f"    BEFORE {icon_b}: {before['got'][:55]:55s} [{before['got_id']}] ({before['time']:.1f}s)")
    print(f"    AFTER  {icon_a}: {after['got'][:55]:55s} [{after['got_id']}] ({after['time']:.1f}s)")
    if changed == "IMPROVED":
        print(f"    >>> {changed} <<<")
    elif changed == "REGRESSED":
        print(f"    !!! {changed} !!!")
    
    # Show top-3 for AFTER if it failed
    if not after["ok"]:
        print(f"    AFTER top-3:")
        for i, r in enumerate(after["top3"], 1):
            rid = id_lookup.get(r.candidate, "?")
            flag = " <<" if any(rid.startswith(p) for p in tc["acceptable_ids"]) else ""
            print(f"      {i}. [{rid:<10s}] {r.candidate[:50]:50s} {r.score:.4f}{flag}")

n = len(SAMPLE_TESTS)
print(f"\n{'='*90}")
print(f"  SUMMARY: BEFORE {before_ok}/{n} → AFTER {after_ok}/{n}")
print(f"  Delta: {after_ok - before_ok:+d} ({(after_ok-before_ok)/n:+.0%})")
print(f"{'='*90}")
