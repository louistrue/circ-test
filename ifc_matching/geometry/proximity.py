"""Geometric proximity analysis using IFC bounding boxes.

Overcomes the limitation of relying solely on explicit IFC schema
relationships (``IfcRelSpaceBoundary``, etc.) by computing which
elements physically touch or overlap based on their geometry.

This enables the circularity assessment tool to work on "dirty" BIM
models where architects haven't properly defined relationships, and
on MEP models which are typically excluded from schema-based analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-Aligned Bounding Box for an IFC element."""

    element_id: str
    element_type: str
    min_pt: tuple[float, float, float]
    max_pt: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                (self.min_pt[i] + self.max_pt[i]) / 2.0
                for i in range(3)
            ]
        )

    @property
    def volume(self) -> float:
        return float(np.prod([self.max_pt[i] - self.min_pt[i] for i in range(3)]))

    def expanded(self, tolerance: float) -> BoundingBox:
        """Return a new BBox expanded by *tolerance* in all directions."""
        return BoundingBox(
            element_id=self.element_id,
            element_type=self.element_type,
            min_pt=tuple(self.min_pt[i] - tolerance for i in range(3)),
            max_pt=tuple(self.max_pt[i] + tolerance for i in range(3)),
        )


@dataclass
class ProximityEdge:
    """A discovered geometric adjacency between two elements."""

    element_a: str
    element_b: str
    distance: float
    overlap_volume: float
    contact_type: str  # "touching", "overlapping", "near"


class GeometricProximityAnalyzer:
    """Compute element adjacency from bounding-box geometry.

    Parameters
    ----------
    touch_tolerance:
        Maximum gap (in model units, typically metres) between two
        bounding boxes to still consider them "touching".
    near_threshold:
        Maximum center-to-surface distance to consider elements "near".
    """

    def __init__(
        self,
        touch_tolerance: float = 0.01,
        near_threshold: float = 0.5,
    ) -> None:
        self.touch_tolerance = touch_tolerance
        self.near_threshold = near_threshold

    # ------------------------------------------------------------------
    # IFC extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_bounding_boxes(ifc_path: str) -> list[BoundingBox]:
        """Extract AABBs for every product in an IFC file.

        Uses ifcopenshell's built-in geometry processing to compute
        placement-aware bounding boxes.
        """
        import ifcopenshell
        import ifcopenshell.geom

        ifc_file = ifcopenshell.open(ifc_path)
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)

        boxes: list[BoundingBox] = []
        for product in ifc_file.by_type("IfcProduct"):
            if not product.Representation:
                continue

            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
                verts = shape.geometry.verts
                if not verts:
                    continue

                # verts is a flat list [x0,y0,z0, x1,y1,z1, ...]
                pts = np.array(verts).reshape(-1, 3)
                min_pt = tuple(pts.min(axis=0).tolist())
                max_pt = tuple(pts.max(axis=0).tolist())

                boxes.append(
                    BoundingBox(
                        element_id=str(product.GlobalId),
                        element_type=product.is_a(),
                        min_pt=min_pt,
                        max_pt=max_pt,
                    )
                )
            except Exception:
                # Geometry processing can fail for abstract elements
                continue

        return boxes

    # ------------------------------------------------------------------
    # Proximity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _box_distance(a: BoundingBox, b: BoundingBox) -> float:
        """Compute minimum distance between two AABBs (0 if overlapping)."""
        dist_sq = 0.0
        for i in range(3):
            gap = max(0.0, a.min_pt[i] - b.max_pt[i], b.min_pt[i] - a.max_pt[i])
            dist_sq += gap * gap
        return float(np.sqrt(dist_sq))

    @staticmethod
    def _overlap_volume(a: BoundingBox, b: BoundingBox) -> float:
        """Compute volume of overlap between two AABBs."""
        overlap = 1.0
        for i in range(3):
            lo = max(a.min_pt[i], b.min_pt[i])
            hi = min(a.max_pt[i], b.max_pt[i])
            if lo >= hi:
                return 0.0
            overlap *= hi - lo
        return overlap

    def find_adjacencies(
        self,
        boxes: list[BoundingBox],
    ) -> list[ProximityEdge]:
        """Find all pairs of elements that are touching, overlapping, or near.

        Uses a sweep-and-prune on the X axis to avoid O(n^2) full checks
        for large models.
        """
        if not boxes:
            return []

        # Sort by min X for sweep-and-prune
        sorted_boxes = sorted(boxes, key=lambda b: b.min_pt[0])
        edges: list[ProximityEdge] = []

        for i in range(len(sorted_boxes)):
            a = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                b = sorted_boxes[j]

                # Prune: if b's min X is beyond a's max X + threshold, skip rest
                if b.min_pt[0] > a.max_pt[0] + self.near_threshold:
                    break

                dist = self._box_distance(a, b)
                overlap = self._overlap_volume(a, b)

                if overlap > 0:
                    contact_type = "overlapping"
                elif dist <= self.touch_tolerance:
                    contact_type = "touching"
                elif dist <= self.near_threshold:
                    contact_type = "near"
                else:
                    continue

                edges.append(
                    ProximityEdge(
                        element_a=a.element_id,
                        element_b=b.element_id,
                        distance=dist,
                        overlap_volume=overlap,
                        contact_type=contact_type,
                    )
                )

        return edges

    def build_adjacency_graph(
        self, ifc_path: str
    ) -> dict[str, list[ProximityEdge]]:
        """Full pipeline: extract boxes → find adjacencies → return graph.

        Returns a dict mapping element IDs to their adjacency edges.
        """
        boxes = self.extract_bounding_boxes(ifc_path)
        edges = self.find_adjacencies(boxes)

        graph: dict[str, list[ProximityEdge]] = {}
        for edge in edges:
            graph.setdefault(edge.element_a, []).append(edge)
            graph.setdefault(edge.element_b, []).append(edge)
        return graph
