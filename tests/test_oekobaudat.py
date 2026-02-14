"""Tests for ÖKOBAUDAT client and source-aware preprocessing."""

import pytest

from ifc_matching.databases.oekobaudat_client import (
    OekobaudatClient,
    OekobaudatProcess,
)
from ifc_matching.preprocessing import (
    filter_candidates_by_element_type,
    ELEMENT_TYPE_CATEGORIES_OEKOBAUDAT,
)


class TestOekobaudatProcess:
    def test_display_name(self):
        p = OekobaudatProcess(uuid="x", name="Beton C20/25", classific_id="1.4.01")
        assert p.display_name == "Beton C20/25"

    def test_parse_process(self):
        item = {
            "uuid": "abc-123",
            "name": "Test EPD",
            "classificId": "1.4.01",
            "classific": "Mineralische Baustoffe / Beton",
        }
        client = OekobaudatClient()
        p = client._parse_process(item)
        assert p.uuid == "abc-123"
        assert p.name == "Test EPD"
        assert p.classific_id == "1.4.01"


class TestOekobaudatClient:
    def test_search_processes_integration(self):
        """Integration test: real API call (skipped if offline)."""
        client = OekobaudatClient()  # no cache_path
        try:
            results = client.search_processes(name="Beton", limit=5)
        except Exception as e:
            pytest.skip(f"ÖKOBAUDAT API unavailable: {e}")
        assert len(results) <= 5
        for r in results:
            assert isinstance(r, OekobaudatProcess)
            assert r.uuid
            assert r.name

    def test_get_category_lookup(self):
        client = OekobaudatClient()
        try:
            lookup = client.get_category_lookup(name="Beton", limit=3)
        except Exception as e:
            pytest.skip(f"ÖKOBAUDAT API unavailable: {e}")
        assert isinstance(lookup, dict)
        for name, cid in lookup.items():
            assert isinstance(name, str)
            assert isinstance(cid, str)
            assert cid  # e.g. "1.4.01"


class TestFilterCandidatesOekobaudat:
    def test_filter_by_element_type_oekobaudat(self):
        # Need 10+ matching candidates so safety fallback doesn't return full list
        # IfcBeam allows 1, 3, 4 (concrete, wood, metals)
        candidates = (
            ["Beton C20/25", "Beton C30/37", "Hochbaubeton", "Ortbeton", "Betonfertigteil"]
            + ["Stahlprofil", "Stahlblech", "Armierungsstahl", "Edelstahl"]
            + ["Massivholz", "Brettschichtholz", "OSB-Platte"]
            + ["Glaswolle Dämmung", "EPS", "Steinwolle"]
            + ["Fensterrahmen Holz", "Isolierverglasung"]
        )
        candidate_ids = {
            "Beton C20/25": "1.4.01",
            "Beton C30/37": "1.4.01",
            "Hochbaubeton": "1.4.01",
            "Ortbeton": "1.4.01",
            "Betonfertigteil": "1.3.05",
            "Stahlprofil": "4.1.03",
            "Stahlblech": "4.1.04",
            "Armierungsstahl": "4.1.01",
            "Edelstahl": "4.2.01",
            "Massivholz": "3.1.01",
            "Brettschichtholz": "3.1.04",
            "OSB-Platte": "3.2.04",
            "Glaswolle Dämmung": "2.1.02",
            "EPS": "2.2.01",
            "Steinwolle": "2.1.03",
            "Fensterrahmen Holz": "7.1.01",
            "Isolierverglasung": "7.2.01",
        }
        filtered = filter_candidates_by_element_type(
            candidates, candidate_ids, "IfcBeam", source="oekobaudat"
        )
        assert "Beton C20/25" in filtered
        assert "Stahlprofil" in filtered
        assert "Massivholz" in filtered
        assert "Glaswolle Dämmung" not in filtered
        assert "Fensterrahmen Holz" not in filtered

    def test_filter_unknown_element_type_returns_all(self):
        candidates = ["A", "B"]
        candidate_ids = {"A": "1.4.01", "B": "2.1.01"}
        result = filter_candidates_by_element_type(
            candidates, candidate_ids, "IfcUnknown", source="oekobaudat"
        )
        assert result == candidates

    def test_filter_safety_fallback(self):
        candidates = ["Only one match"]
        candidate_ids = {"Only one match": "1.4.01"}
        result = filter_candidates_by_element_type(
            candidates, candidate_ids, "IfcBeam", source="oekobaudat"
        )
        assert len(result) >= 1
        # Safety: if filtered < 10, returns full list
        assert "Only one match" in result


class TestElementTypeCategoriesOekobaudat:
    def test_wall_has_mineral_insulation_wood_metals(self):
        prefixes = ELEMENT_TYPE_CATEGORIES_OEKOBAUDAT.get("IfcWall", [])
        assert "1" in prefixes
        assert "2" in prefixes
        assert "3" in prefixes
        assert "4" in prefixes

    def test_railing_metals_only(self):
        prefixes = ELEMENT_TYPE_CATEGORIES_OEKOBAUDAT.get("IfcRailing", [])
        assert prefixes == ["4"]
