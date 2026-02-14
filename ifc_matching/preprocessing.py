"""Query preprocessing and domain knowledge for IFC-to-LCA matching.

Four layers of improvement:
1. Query cleaning: Strip Revit/ArchiCAD/Tekla naming noise
2. Cross-lingual synonyms: Map EN/NL/FR construction terms → DE terms  
3. Element-type category filtering: Narrow candidates by IFC product type
4. ÖKOBAUDAT search-term extraction: Derive clean German API keywords from
   any IFC query (multilingual, noisy) for the ÖKOBAUDAT full-text search
"""

from __future__ import annotations

import re
from typing import Literal


# ---------------------------------------------------------------------------
# 1. Query Preprocessing
# ---------------------------------------------------------------------------

def clean_ifc_query(query: str) -> str:
    """Normalize IFC material names from various authoring tools.
    
    Handles:
    - Revit Swiss conventions: _prefix, _wg suffix, underscores
    - Numeric IDs from ArchiCAD/Revit: "Stahlbeton 65690", "Gips 275646950"
    - RGB color codes: "140-100-70", "51-79-234"
    - Tool prefixes: f2_, h2_, DD_, AT_
    - German umlaut ASCII encoding: ae→ä, oe→ö, ue→ü
    - Revit element info: "Floor:STB 25cm, Beton C30/37 Bodenplatte 2:2515405"
    """
    original = query
    
    # Strip leading/trailing underscores and whitespace
    query = query.strip().strip('_').strip()
    
    # Remove Revit _wg suffix (Werkgruppe)
    query = re.sub(r'[_ ]wg$', '', query, flags=re.IGNORECASE)
    
    # Remove tool-specific prefixes: f2_, h2_, DD_, AT_ (but keep the rest)
    query = re.sub(r'^([a-zA-Z]\d?_|DD[_ ]|AT_)', '', query)
    
    # Remove long numeric IDs (6+ digits, likely element IDs)
    query = re.sub(r'\b\d{6,}\b', '', query)
    
    # Remove RGB-style color codes (3 groups of 1-3 digits separated by dashes)
    query = re.sub(r'\b\d{1,3}-\d{1,3}-\d{1,3}\b', '', query)
    
    # Replace underscores with spaces
    query = query.replace('_', ' ')
    
    # Normalize German umlauts from ASCII encoding
    # Be careful: "ae" at word boundary is more likely an umlaut replacement
    # than in words like "aerosol". We use a conservative approach.
    query = _normalize_umlauts(query)
    
    # Extract material info from Revit composite names like
    # "Floor:STB 25cm, Beton C30/37 Bodenplatte 2:2515405"
    if ':' in query and any(kw in query.lower() for kw in ['beton', 'holz', 'stahl', 'gips', 'glas']):
        # Try to extract the material-relevant part
        parts = query.split(':')
        for part in parts:
            part = part.strip()
            if any(kw in part.lower() for kw in ['beton', 'holz', 'stahl', 'gips', 'glas', 'dämm', 'isolier']):
                query = part
                break
    
    # Clean up multiple spaces
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove trailing numeric debris
    query = re.sub(r'\s+\d+$', '', query).strip()
    
    return query if query else original


def _normalize_umlauts(text: str) -> str:
    """Carefully normalize ae→ä, oe→ö, ue→ü in German construction terms.
    
    Only applies when the pattern is likely a German umlaut replacement,
    not a legitimate letter combination (e.g., 'Israel', 'Aerosol').
    """
    # Known construction terms with umlauts
    UMLAUT_WORDS = {
        'waerme': 'wärme', 'daemm': 'dämm', 'daemmung': 'dämmung',
        'waermedaemmung': 'wärmedämmung',
        'puerstahl': 'pürstahl',
        'gruendung': 'gründung', 'uebergang': 'übergang',
        'auessere': 'äussere', 'aeussere': 'äussere',
        'oeffnung': 'öffnung',
        'fussboden': 'fussboden',  # don't change this one
    }
    
    lower = text.lower()
    for ascii_form, umlaut_form in UMLAUT_WORDS.items():
        if ascii_form in lower:
            # Replace preserving case of first char
            idx = lower.index(ascii_form)
            text = text[:idx] + umlaut_form + text[idx + len(ascii_form):]
            lower = text.lower()
    
    return text


# ---------------------------------------------------------------------------
# 2. Cross-Lingual Synonym Expansion
# ---------------------------------------------------------------------------

# Maps IFC material terms (any language) to German KBOB search terms.
# Returns a list of additional search terms to inject into the query
# or use as retrieval expansion.
SYNONYM_MAP: dict[str, list[str]] = {
    # --- Dutch → German ---
    "staal": ["Stahl", "Stahlprofil"],
    "beton": ["Beton", "Hochbaubeton"],
    "prefab": ["Fertigteil", "Betonfertigteil"],
    "prefabbeton": ["Betonfertigteil"],
    "isolatie": ["Dämmung", "Wärmedämmung"],
    "gevel": ["Fassade"],
    "randen": ["Rand", "Betonfertigteil"],
    "ihwg": ["Hochbaubeton"],  # Dutch project code for in-situ concrete
    
    # --- English → German ---
    "concrete": ["Beton", "Hochbaubeton"],
    "cast in situ": ["Ortbeton", "Hochbaubeton"],
    "steel": ["Stahl", "Stahlprofil"],
    "insulation": ["Dämmung", "Wärmedämmung"],
    "rigid insulation": ["Polystyrol extrudiert", "XPS", "EPS"],
    "semi-rigid insulation": ["Glaswolle", "Steinwolle"],
    "thermal barriers": ["Wärmedämmung"],
    "masonry": ["Mauerwerk"],
    "brick": ["Backstein", "Mauerwerk"],
    "concrete block": ["Zementstein", "Betonstein"],
    "plasterboard": ["Gipskartonplatte"],
    "plywood": ["Sperrholzplatte"],
    "dimensional lumber": ["Konstruktionsholz", "Brettschichtholz"],
    "wood flooring": ["Parkett"],
    "sheathing": ["Holzwerkstoff", "Sperrholzplatte"],
    "epdm membrane": ["Dichtungsbahn Gummi EPDM"],
    "epdm": ["Dichtungsbahn Gummi EPDM"],
    "roofing": ["Dachdeckung", "Dichtungsbahn"],
    "ceramic tile": ["Keramikplatte", "Steinzeugplatte"],
    "grout": ["Mörtel", "Zementmörtel"],
    "air space": ["Luftschicht"],
    "stud layer": ["Ständerwerk", "Metallständer"],
    
    # --- Generic German construction terms → specific KBOB ---
    "wärmedämmung druckfest": ["Polystyrol extrudiert XPS", "Polystyrol expandiert EPS"],
    "wärmedämmung": ["Glaswolle", "Steinwolle", "Polystyrol expandiert EPS"],
    "dämmung hart": ["Polystyrol extrudiert XPS", "Polystyrol expandiert EPS", "Schaumglas"],
    "dämmung weich": ["Glaswolle", "Steinwolle", "Weichfaserplatte"],
    "isolierung hart": ["Polystyrol extrudiert XPS", "Polystyrol expandiert EPS"],
    "ortbeton bewehrt": ["Hochbaubeton"],
    "ortbeton": ["Hochbaubeton"],
    "stahlbeton": ["Hochbaubeton"],
    "leichtbeton": ["Porenbetonstein", "Leichtzementstein"],
    "trockenbau": ["Gipskartonplatte", "Gipsfaserplatte"],
    "rigips": ["Gipskartonplatte"],
    "naturstein": ["Natursteinplatte"],
    "edelstahl": ["Edelstahl", "Chromstahl"],
    "zink": ["Zinkblech"],
    "aluminium": ["Aluminiumblech"],
    "vorfabriziert": ["Betonfertigteil"],
    "fertigbeton": ["Betonfertigteil"],
}


def expand_query_with_synonyms(query: str) -> str:
    """Expand a cleaned query with German KBOB synonyms.
    
    Matches synonym keys against the lowercased query and appends
    the German equivalents. This helps the embedding retriever
    find the correct KBOB entries across languages.
    """
    lower = query.lower()
    expansions = set()
    
    # Try longest matches first (more specific)
    for key in sorted(SYNONYM_MAP.keys(), key=len, reverse=True):
        if key in lower:
            for synonym in SYNONYM_MAP[key]:
                expansions.add(synonym)
    
    if expansions:
        return query + " | " + " ".join(sorted(expansions))
    return query


# ---------------------------------------------------------------------------
# 3. Element-Type → Category Filtering (KBOB and ÖKOBAUDAT)
# ---------------------------------------------------------------------------

# Maps IFC element types to likely KBOB category ID prefixes.
# Used to pre-filter candidates before embedding search.
ELEMENT_TYPE_CATEGORIES: dict[str, list[str]] = {
    # Structural concrete/masonry
    "IfcWall": ["01", "02", "03", "04", "07", "10"],
    "IfcWallStandardCase": ["01", "02", "03", "04", "07", "10"],
    "IfcSlab": ["01", "03", "04", "07", "09", "10", "11"],
    "IfcBeam": ["01", "06", "07"],
    "IfcColumn": ["01", "06", "07"],
    "IfcFooting": ["01", "00"],
    "IfcPile": ["01", "00"],
    "IfcRamp": ["01", "03", "11"],
    
    # Roof
    "IfcRoof": ["01", "03", "07", "09", "10"],
    
    # Openings
    "IfcWindow": ["05", "03"],
    "IfcDoor": ["12", "05"],
    "IfcCurtainWall": ["05", "06"],
    
    # Other
    "IfcCovering": ["03", "04", "07", "10", "11"],
    "IfcStairFlight": ["01", "03", "07"],
    "IfcStair": ["01", "03", "07"],
    "IfcRailing": ["06"],
    "IfcMember": ["06", "07"],
    "IfcPlate": ["03", "05", "06"],
    
    # Generic
    "IfcBuildingElementProxy": [],  # don't filter — could be anything
    "IfcBuildingElementPart": [],
}

# Maps IFC element types to ÖKOBAU.DAT class ID prefixes (e.g. "1", "1.4", "2.1").
# ÖKOBAUDAT uses hierarchical IDs: 1=Mineral, 2=Insulation, 3=Wood, 4=Metals,
# 5=Coatings, 6=Plastics, 7=Window/Facade, 8=Building tech, 9=Misc, 10=Composites.
ELEMENT_TYPE_CATEGORIES_OEKOBAUDAT: dict[str, list[str]] = {
    "IfcWall": ["1", "2", "3", "4"],
    "IfcWallStandardCase": ["1", "2", "3", "4"],
    "IfcSlab": ["1", "2", "3", "4"],
    "IfcBeam": ["1", "3", "4"],
    "IfcColumn": ["1", "3", "4"],
    "IfcFooting": ["1"],
    "IfcPile": ["1"],
    "IfcRamp": ["1", "3", "4"],
    "IfcRoof": ["1", "2", "3", "4"],
    "IfcWindow": ["5", "6", "7"],
    "IfcDoor": ["3", "7"],
    "IfcCurtainWall": ["5", "6", "7"],
    "IfcCovering": ["1", "2", "3", "4", "5"],
    "IfcStairFlight": ["1", "3", "4"],
    "IfcStair": ["1", "3", "4"],
    "IfcRailing": ["4"],
    "IfcMember": ["3", "4"],
    "IfcPlate": ["1", "3", "4", "5", "6", "7"],
    "IfcBuildingElementProxy": [],
    "IfcBuildingElementPart": [],
}


def filter_candidates_by_element_type(
    candidates: list[str],
    candidate_ids: dict[str, str],
    element_type: str | None,
    *,
    source: Literal["kbob", "oekobaudat"] = "kbob",
) -> list[str]:
    """Filter candidates to categories relevant to the IFC element type.

    Uses KBOB or ÖKOBAUDAT category mappings depending on source.
    candidate_ids maps display_name -> category_id (KBOB id or classific_id).
    Returns the filtered list, or the full list if element_type is unknown
    or has no category mapping.
    """
    if not element_type:
        return candidates

    if source == "oekobaudat":
        prefixes = ELEMENT_TYPE_CATEGORIES_OEKOBAUDAT.get(element_type, [])
    else:
        prefixes = ELEMENT_TYPE_CATEGORIES.get(element_type, [])

    if not prefixes:
        return candidates  # unknown type or too generic → don't filter

    filtered = [
        c for c in candidates
        if any(candidate_ids.get(c, "").startswith(p) for p in prefixes)
    ]

    # Safety: if filtering removed too many (< 10), fall back to full list
    # This prevents edge cases where the mapping is too restrictive
    if len(filtered) < 10:
        return candidates

    return filtered


# ---------------------------------------------------------------------------
# 4. ÖKOBAUDAT Search-Term Extraction
# ---------------------------------------------------------------------------

# Maps a recognised material keyword (any language, lowercased) to a list
# of German search terms suitable for the ÖKOBAUDAT full-text API.
# Ordered most-specific first; the benchmark / CLI tries them in sequence
# until results are found.
#
# IMPORTANT: The ÖKOBAUDAT name search does **AND** matching across words.
# Each returned term should therefore be a SINGLE keyword (or two very
# common collocated words like "Brettschichtholz") so the API can match.

MATERIAL_KEYWORD_TO_OEKO: dict[str, list[str]] = {
    # ---- German construction terms → ÖKOBAUDAT search keywords ----
    # Concrete family
    "ortbeton": ["Beton"],
    "stahlbeton": ["Beton"],
    "hochbaubeton": ["Beton"],
    "spannbeton": ["Spannbeton", "Beton"],
    "leichtbeton": ["Leichtbeton", "Beton"],
    "porenbeton": ["Porenbeton"],
    "beton": ["Beton"],
    "estrich": ["Estrich"],
    "mörtel": ["Mörtel"],
    "putz": ["Putz"],
    "zement": ["Zement"],
    # Masonry
    "kalksandstein": ["Kalksandstein"],
    "mauerwerk": ["Mauerwerk", "Kalksandstein"],
    "backstein": ["Ziegel", "Klinker"],
    "ziegel": ["Ziegel"],
    "klinker": ["Klinker"],
    # Insulation
    "wärmedämmung": ["Dämmstoff", "Dämmung"],
    "dämmung": ["Dämmstoff", "Dämmung"],
    "steinwolle": ["Steinwolle"],
    "glaswolle": ["Glaswolle"],
    "mineralwolle": ["Mineralwolle"],
    "polystyrol": ["Polystyrol"],
    "eps": ["EPS"],
    "xps": ["XPS"],
    "schaumglas": ["Schaumglas"],
    "isolierung": ["Dämmstoff", "Dämmung"],
    "weichfaser": ["Holzfaser"],
    "holzfaser": ["Holzfaser"],
    # Wood
    "holz": ["Holz"],
    "brettschichtholz": ["Brettschichtholz"],
    "sperrholz": ["Sperrholz"],
    "parkett": ["Parkett"],
    "laminat": ["Laminat"],
    "konstruktionsholz": ["Konstruktionsholz", "Schnittholz"],
    "schnittholz": ["Schnittholz"],
    # Metals
    "stahl": ["Stahl"],
    "edelstahl": ["Edelstahl"],
    "chromstahl": ["Edelstahl"],
    "zink": ["Zink"],
    "kupfer": ["Kupfer"],
    "aluminium": ["Aluminium"],
    "metall": ["Stahl"],
    # Gypsum
    "gipskarton": ["Gipskartonplatte", "Gips"],
    "gipsfaser": ["Gipsfaserplatte", "Gips"],
    "gipsplatte": ["Gipsplatte", "Gips"],
    "gips": ["Gips"],
    "rigips": ["Gipskartonplatte", "Gips"],
    "trockenbau": ["Gipskartonplatte", "Gips"],
    # Glass / facade
    "glas": ["Glas"],
    "isolierverglasung": ["Verglasung", "Glas"],
    "verglasung": ["Verglasung", "Glas"],
    "faserzement": ["Faserzement"],
    # Roofing / sealing
    "bitumen": ["Bitumen"],
    "dachdeckung": ["Dach"],
    "epdm": ["EPDM"],
    "abdichtung": ["Abdichtung"],
    "dichtungsbahn": ["Abdichtung"],
    # Floor / tile
    "fliese": ["Fliese"],
    "keramik": ["Keramik", "Fliese"],
    # Stone / aggregate
    "naturstein": ["Naturstein"],
    "granit": ["Granit", "Naturstein"],
    "kies": ["Kies"],
    "schotter": ["Kies"],

    # ---- English → German (ÖKOBAUDAT is German-only) ----
    "concrete": ["Beton"],
    "reinforced concrete": ["Beton"],
    "cast in situ": ["Beton"],
    "lightweight concrete": ["Leichtbeton", "Beton"],
    "concrete block": ["Betonstein", "Betonmauerstein", "Beton"],
    "precast": ["Betonfertigteil", "Beton"],
    "mortar": ["Mörtel"],
    "screed": ["Estrich"],
    "steel": ["Stahl"],
    "stainless steel": ["Edelstahl"],
    "aluminium": ["Aluminium"],
    "aluminum": ["Aluminium"],
    "zinc": ["Zink"],
    "copper": ["Kupfer"],
    "metal": ["Stahl"],
    "wood": ["Holz"],
    "timber": ["Holz"],
    "lumber": ["Schnittholz", "Holz"],
    "plywood": ["Sperrholz", "Holz"],
    "flooring": ["Bodenbelag", "Parkett"],
    "sheathing": ["Holzwerkstoff", "Sperrholz"],
    "insulation": ["Dämmstoff", "Dämmung"],
    "rigid insulation": ["Polystyrol", "XPS", "EPS", "Dämmstoff"],
    "semi-rigid insulation": ["Mineralwolle", "Steinwolle", "Glaswolle"],
    "semi-rigid": ["Mineralwolle", "Steinwolle", "Glaswolle"],
    "thermal barrier": ["Dämmstoff"],
    "glass wool": ["Glaswolle"],
    "rock wool": ["Steinwolle"],
    "stone wool": ["Steinwolle"],
    "mineral wool": ["Mineralwolle"],
    "foam glass": ["Schaumglas"],
    "plasterboard": ["Gipskartonplatte", "Gips"],
    "drywall": ["Gipskartonplatte"],
    "gypsum": ["Gips"],
    "glass": ["Glas"],
    "glazing": ["Verglasung", "Glas"],
    "masonry": ["Mauerwerk", "Kalksandstein"],
    "brick": ["Ziegel", "Klinker"],
    "natural stone": ["Naturstein"],
    "granite": ["Granit", "Naturstein"],
    "gravel": ["Kies"],
    "tile": ["Fliese"],
    "ceramic": ["Keramik", "Fliese"],
    "bitumen": ["Bitumen"],
    "roofing": ["Dach", "Bitumen"],
    "membrane": ["Abdichtung", "EPDM"],
    "epdm": ["EPDM", "Abdichtung"],
    "fibre cement": ["Faserzement"],
    "fiber cement": ["Faserzement"],

    # ---- Dutch → German ----
    "staal": ["Stahl"],
    "beton": ["Beton"],  # also covers "prefabbeton" via substring
    "prefabbeton": ["Betonfertigteil", "Beton"],
    "prefab": ["Betonfertigteil", "Beton"],
    "isolatie": ["Dämmstoff", "Dämmung"],
    "gevel": ["Fassade"],
    "hout": ["Holz"],
    "glas": ["Glas"],
    "steen": ["Naturstein"],
    "baksteen": ["Ziegel"],

    # ---- French → German ----
    "béton": ["Beton"],
    "acier": ["Stahl"],
    "bois": ["Holz"],
    "isolation": ["Dämmstoff"],
    "verre": ["Glas"],
}


def extract_oekobaudat_search_terms(query: str) -> list[str]:
    """Derive clean German search keywords for the ÖKOBAUDAT API.

    The ÖKOBAUDAT ``name`` search does **AND full-text matching**.  Multi-word
    queries with noise (tool IDs, strength classes, brand names) return zero
    results.  This function:

    1. Cleans the IFC query (remove tool junk)
    2. Scans for the longest matching material keyword
    3. Returns an ordered list of German search terms (specific → broad)

    Returns
    -------
    List of 1–3 German terms to try in order.  Never empty — always falls
    back to the single longest word in the cleaned query.
    """
    cleaned = clean_ifc_query(query).lower()

    # Collect ALL keyword matches with their position and length.
    matches: list[tuple[int, int, str, list[str]]] = []  # (pos, len, keyword, terms)
    for keyword in MATERIAL_KEYWORD_TO_OEKO:
        pos = cleaned.find(keyword)
        if pos != -1:
            matches.append((pos, len(keyword), keyword, MATERIAL_KEYWORD_TO_OEKO[keyword]))

    if matches:
        # Remove subsumed matches: if keyword A falls entirely within
        # keyword B's span, discard A (e.g. "stahl" inside "edelstahl").
        non_subsumed: list[tuple[int, int, str, list[str]]] = []
        for m in matches:
            mpos, mlen, mkw, mterms = m
            subsumed = False
            for other in matches:
                opos, olen, okw, _ = other
                if olen > mlen and opos <= mpos and opos + olen >= mpos + mlen:
                    subsumed = True
                    break
            if not subsumed:
                non_subsumed.append(m)

        # Sort remaining by keyword length descending (most specific first).
        non_subsumed.sort(key=lambda m: m[1], reverse=True)

        # Flatten all terms, de-duplicated, preserving specificity order.
        seen: set[str] = set()
        all_terms: list[str] = []
        for _, _, _, terms in non_subsumed:
            for t in terms:
                if t not in seen:
                    seen.add(t)
                    all_terms.append(t)
        return all_terms

    # Fallback: use the longest word (>= 3 chars) from the cleaned query
    words = [w for w in re.split(r"[\s\-_/,;()|]+", cleaned) if len(w) >= 3]
    if words:
        longest = max(words, key=len)
        return [longest.capitalize()]

    return [query.strip()[:30]]


# ---------------------------------------------------------------------------
# Combined preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """Full preprocessing: clean → expand with synonyms."""
    cleaned = clean_ifc_query(query)
    expanded = expand_query_with_synonyms(cleaned)
    return expanded
