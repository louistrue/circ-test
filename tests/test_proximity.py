"""Tests for geometric proximity analysis."""

import pytest

from ifc_matching.geometry.proximity import (
    BoundingBox,
    GeometricProximityAnalyzer,
)


class TestBoundingBox:
    def test_center(self):
        bb = BoundingBox("e1", "IfcWall", (0, 0, 0), (2, 4, 6))
        center = bb.center
        assert center[0] == pytest.approx(1.0)
        assert center[1] == pytest.approx(2.0)
        assert center[2] == pytest.approx(3.0)

    def test_volume(self):
        bb = BoundingBox("e1", "IfcWall", (0, 0, 0), (2, 3, 4))
        assert bb.volume == pytest.approx(24.0)

    def test_expanded(self):
        bb = BoundingBox("e1", "IfcWall", (1, 1, 1), (3, 3, 3))
        expanded = bb.expanded(0.5)
        assert expanded.min_pt == (0.5, 0.5, 0.5)
        assert expanded.max_pt == (3.5, 3.5, 3.5)


class TestBoxDistance:
    def setup_method(self):
        self.analyzer = GeometricProximityAnalyzer()

    def test_overlapping_boxes(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (2, 2, 2))
        b = BoundingBox("b", "IfcWall", (1, 1, 1), (3, 3, 3))
        assert self.analyzer._box_distance(a, b) == pytest.approx(0.0)

    def test_touching_boxes(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1))
        b = BoundingBox("b", "IfcWall", (1, 0, 0), (2, 1, 1))
        assert self.analyzer._box_distance(a, b) == pytest.approx(0.0)

    def test_separated_boxes(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1))
        b = BoundingBox("b", "IfcWall", (2, 0, 0), (3, 1, 1))
        assert self.analyzer._box_distance(a, b) == pytest.approx(1.0)


class TestOverlapVolume:
    def setup_method(self):
        self.analyzer = GeometricProximityAnalyzer()

    def test_full_overlap(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (2, 2, 2))
        b = BoundingBox("b", "IfcWall", (0, 0, 0), (2, 2, 2))
        assert self.analyzer._overlap_volume(a, b) == pytest.approx(8.0)

    def test_partial_overlap(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (2, 2, 2))
        b = BoundingBox("b", "IfcWall", (1, 1, 1), (3, 3, 3))
        assert self.analyzer._overlap_volume(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1))
        b = BoundingBox("b", "IfcWall", (2, 2, 2), (3, 3, 3))
        assert self.analyzer._overlap_volume(a, b) == pytest.approx(0.0)


class TestFindAdjacencies:
    def test_touching_elements(self):
        analyzer = GeometricProximityAnalyzer(touch_tolerance=0.01)
        boxes = [
            BoundingBox("wall1", "IfcWall", (0, 0, 0), (1, 1, 3)),
            BoundingBox("wall2", "IfcWall", (1.005, 0, 0), (2, 1, 3)),  # 0.005m gap
        ]
        edges = analyzer.find_adjacencies(boxes)
        assert len(edges) == 1
        assert edges[0].contact_type == "touching"

    def test_overlapping_elements(self):
        analyzer = GeometricProximityAnalyzer()
        boxes = [
            BoundingBox("slab", "IfcSlab", (0, 0, 2.9), (5, 5, 3.1)),
            BoundingBox("wall", "IfcWall", (0, 0, 0), (0.3, 5, 3.0)),
        ]
        edges = analyzer.find_adjacencies(boxes)
        assert len(edges) == 1
        assert edges[0].contact_type == "overlapping"

    def test_near_elements(self):
        analyzer = GeometricProximityAnalyzer(near_threshold=0.5)
        boxes = [
            BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1)),
            BoundingBox("b", "IfcColumn", (1.3, 0, 0), (1.6, 1, 1)),  # 0.3m gap
        ]
        edges = analyzer.find_adjacencies(boxes)
        assert len(edges) == 1
        assert edges[0].contact_type == "near"

    def test_far_elements_excluded(self):
        analyzer = GeometricProximityAnalyzer(near_threshold=0.5)
        boxes = [
            BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1)),
            BoundingBox("b", "IfcWall", (10, 0, 0), (11, 1, 1)),
        ]
        edges = analyzer.find_adjacencies(boxes)
        assert len(edges) == 0

    def test_empty_input(self):
        analyzer = GeometricProximityAnalyzer()
        assert analyzer.find_adjacencies([]) == []

    def test_single_element(self):
        analyzer = GeometricProximityAnalyzer()
        boxes = [BoundingBox("a", "IfcWall", (0, 0, 0), (1, 1, 1))]
        assert analyzer.find_adjacencies(boxes) == []
