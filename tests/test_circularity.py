"""Tests for circularity calculation."""

import pytest

from ifc_matching.databases.archetype import ConnectionType
from ifc_matching.graph.circularity import (
    MaterialNode,
    ConnectionEdge,
    ElementAssembly,
    building_circularity_indicator,
)


class TestMaterialCircularityIndex:
    def test_fully_recyclable(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            materials=[MaterialNode(id="m1", name="Steel", mass=100, recyclability=1.0, reuse_potential=1.0)],
        )
        assert e.material_circularity_index() == pytest.approx(1.0)

    def test_zero_recyclability(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            materials=[MaterialNode(id="m1", name="Concrete", mass=100, recyclability=0.0, reuse_potential=0.0)],
        )
        assert e.material_circularity_index() == pytest.approx(0.0)

    def test_mass_weighted(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            materials=[
                MaterialNode(id="m1", name="Heavy", mass=900, recyclability=0.0, reuse_potential=0.0),
                MaterialNode(id="m2", name="Light", mass=100, recyclability=1.0, reuse_potential=1.0),
            ],
        )
        # (900*0 + 100*1) / 1000 = 0.1
        assert e.material_circularity_index() == pytest.approx(0.1)

    def test_empty_materials(self):
        e = ElementAssembly(element_id="e1", element_type="IfcWall", name="Wall")
        assert e.material_circularity_index() == 0.0


class TestDisassemblyIndex:
    def test_all_chemical_bonds(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.CHEMICAL_BOND),
            ],
        )
        assert e.disassembly_index() == pytest.approx(0.1)

    def test_all_dry_loose(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.DRY_LOOSE),
            ],
        )
        assert e.disassembly_index() == pytest.approx(1.0)

    def test_mixed_connections(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.CHEMICAL_BOND),
                ConnectionEdge(id="c2", source_id="m2", target_id="m3", connection_type=ConnectionType.BOLTED),
            ],
        )
        # (0.1 + 0.6) / 2 = 0.35
        assert e.disassembly_index() == pytest.approx(0.35)

    def test_no_connections_defaults_worst(self):
        e = ElementAssembly(element_id="e1", element_type="IfcWall", name="Wall")
        assert e.disassembly_index() == pytest.approx(0.1)


class TestElementCircularityIndicator:
    def test_eci_formula(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            materials=[MaterialNode(id="m1", name="X", mass=100, recyclability=0.8, reuse_potential=0.6)],
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.BOLTED),
            ],
        )
        mci = 0.5 * 0.8 + 0.5 * 0.6  # 0.7
        di = 0.6  # BOLTED
        expected = 0.5 * mci + 0.5 * di  # 0.5 * 0.7 + 0.5 * 0.6 = 0.65
        assert e.element_circularity_indicator() == pytest.approx(expected)


class TestBuildingCircularityIndicator:
    def test_single_element(self):
        e = ElementAssembly(
            element_id="e1", element_type="IfcWall", name="Wall",
            materials=[MaterialNode(id="m1", name="X", mass=100, recyclability=1.0, reuse_potential=1.0)],
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.DRY_LOOSE),
            ],
        )
        bci = building_circularity_indicator([e])
        assert bci == pytest.approx(1.0)

    def test_mass_weighted_bci(self):
        heavy = ElementAssembly(
            element_id="e1", element_type="IfcSlab", name="Slab",
            materials=[MaterialNode(id="m1", name="Concrete", mass=1000, recyclability=0.0, reuse_potential=0.0)],
            connections=[
                ConnectionEdge(id="c1", source_id="m1", target_id="m2", connection_type=ConnectionType.CHEMICAL_BOND),
            ],
        )
        light = ElementAssembly(
            element_id="e2", element_type="IfcWall", name="Partition",
            materials=[MaterialNode(id="m3", name="Steel", mass=100, recyclability=1.0, reuse_potential=1.0)],
            connections=[
                ConnectionEdge(id="c2", source_id="m3", target_id="m4", connection_type=ConnectionType.DRY_LOOSE),
            ],
        )
        # heavy ECI = 0.5*0 + 0.5*0.1 = 0.05
        # light ECI = 0.5*1 + 0.5*1 = 1.0
        # BCI = (1000*0.05 + 100*1.0) / 1100 = (50+100)/1100 = 150/1100
        bci = building_circularity_indicator([heavy, light])
        assert bci == pytest.approx(150 / 1100, abs=0.01)

    def test_empty_elements(self):
        assert building_circularity_indicator([]) == 0.0
