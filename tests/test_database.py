"""Tests for the database abstraction layer."""

import json
import pytest

from ifc_matching.databases.archetype import (
    ConnectionType,
    TOTEMDatabase,
    OKOBAUDATDatabase,
    KBOBDatabase,
    get_database,
)


class TestConnectionType:
    def test_chemical_values(self):
        assert ConnectionType.CHEMICAL_BOND.value == 0.1
        assert ConnectionType.DRY_LOOSE.value == 1.0

    def test_from_string_exact(self):
        assert ConnectionType.from_string("adhesive") == ConnectionType.CHEMICAL_BOND

    def test_from_string_partial(self):
        assert ConnectionType.from_string("bolted connection") == ConnectionType.BOLTED

    def test_from_string_case_insensitive(self):
        assert ConnectionType.from_string("WELDED") == ConnectionType.WELDED

    def test_from_string_unknown_defaults_chemical(self):
        assert ConnectionType.from_string("unknown_connection") == ConnectionType.CHEMICAL_BOND

    def test_snap_fit(self):
        assert ConnectionType.from_string("snap-fit click") == ConnectionType.SNAP_FIT


class TestArchetypeEntry:
    def test_disassembly_score_average(self):
        from ifc_matching.databases.archetype import ArchetypeEntry
        entry = ArchetypeEntry(
            id="a1", name="Test", element_type="IfcWall",
            connections=[ConnectionType.CHEMICAL_BOND, ConnectionType.DRY_LOOSE],
        )
        # (0.1 + 1.0) / 2 = 0.55
        assert entry.disassembly_score == pytest.approx(0.55)

    def test_disassembly_score_empty(self):
        from ifc_matching.databases.archetype import ArchetypeEntry
        entry = ArchetypeEntry(id="a1", name="Test", element_type="IfcWall")
        assert entry.disassembly_score == 0.1


class TestTOTEMDatabase:
    def test_load_json(self, tmp_path):
        data = {
            "materials": [
                {"id": "m1", "name": "Concrete C30", "category": "Structural", "gwp": 350.0, "connection_type": "chemical"},
                {"id": "m2", "name": "Steel S235", "category": "Structural", "gwp": 1200.0, "connection_type": "bolted"},
            ],
            "archetypes": [
                {
                    "id": "a1", "name": "Exterior Wall Type 1", "element_type": "IfcWall",
                    "layers": [{"id": "m1", "name": "Concrete C30"}],
                    "connections": ["chemical"],
                },
            ],
        }
        p = tmp_path / "totem.json"
        p.write_text(json.dumps(data))

        db = TOTEMDatabase()
        db.load(p)

        materials = db.get_materials()
        assert len(materials) == 2
        assert materials[0].name == "Concrete C30"
        assert materials[0].default_connection == ConnectionType.CHEMICAL_BOND
        assert materials[1].default_connection == ConnectionType.BOLTED

        archetypes = db.get_archetypes()
        assert len(archetypes) == 1
        assert archetypes[0].name == "Exterior Wall Type 1"

    def test_get_material_names(self, tmp_path):
        data = {"materials": [{"id": "m1", "name": "Brick"}, {"id": "m2", "name": "Mortar"}]}
        p = tmp_path / "totem.json"
        p.write_text(json.dumps(data))

        db = TOTEMDatabase()
        db.load(p)
        assert db.get_material_names() == ["Brick", "Mortar"]

    def test_load_csv(self, tmp_path):
        csv_content = "id,name,category,density,gwp,recyclability,connection_type\nm1,Concrete,Structural,2400,350,,chemical\n"
        p = tmp_path / "totem.csv"
        p.write_text(csv_content)

        db = TOTEMDatabase()
        db.load(p)
        assert len(db.get_materials()) == 1
        assert db.get_materials()[0].density == 2400.0


class TestGetDatabase:
    def test_known_databases(self):
        assert isinstance(get_database("totem"), TOTEMDatabase)
        assert isinstance(get_database("okobaudat"), OKOBAUDATDatabase)
        assert isinstance(get_database("kbob"), KBOBDatabase)

    def test_case_insensitive(self):
        assert isinstance(get_database("TOTEM"), TOTEMDatabase)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown database"):
            get_database("nonexistent")
