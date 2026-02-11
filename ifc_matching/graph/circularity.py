"""Building Circularity Indicator (BCI) calculation.

Implements the circularity metrics from the paper, including the
Durmisevic-based connection scoring and element/building-level
aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ifc_matching.databases.archetype import ConnectionType


@dataclass
class MaterialNode:
    """A material within an element layer."""

    id: str
    name: str
    mass: float = 0.0  # kg
    recyclability: float = 0.0  # 0.0–1.0 (fraction recyclable)
    reuse_potential: float = 0.0  # 0.0–1.0


@dataclass
class ConnectionEdge:
    """A connection between two materials/layers."""

    id: str
    source_id: str
    target_id: str
    connection_type: ConnectionType = ConnectionType.CHEMICAL_BOND

    @property
    def disassembly_score(self) -> float:
        return self.connection_type.value


@dataclass
class ElementAssembly:
    """An IFC element composed of material layers and connections."""

    element_id: str
    element_type: str
    name: str
    materials: list[MaterialNode] = field(default_factory=list)
    connections: list[ConnectionEdge] = field(default_factory=list)

    @property
    def total_mass(self) -> float:
        return sum(m.mass for m in self.materials)

    def material_circularity_index(self) -> float:
        """MCI: mass-weighted average of recyclability and reuse potential."""
        if not self.materials or self.total_mass == 0:
            return 0.0

        weighted_sum = sum(
            m.mass * (0.5 * m.recyclability + 0.5 * m.reuse_potential)
            for m in self.materials
        )
        return weighted_sum / self.total_mass

    def disassembly_index(self) -> float:
        """DI: average disassembly potential of all connections."""
        if not self.connections:
            return 0.1  # worst case: assume chemical bond
        return sum(c.disassembly_score for c in self.connections) / len(
            self.connections
        )

    def element_circularity_indicator(self) -> float:
        """ECI = 0.5 * MCI + 0.5 * DI (equally weighted)."""
        return 0.5 * self.material_circularity_index() + 0.5 * self.disassembly_index()


def building_circularity_indicator(elements: list[ElementAssembly]) -> float:
    """BCI: mass-weighted average of all element circularity indicators.

    BCI = Σ(mass_i * ECI_i) / Σ(mass_i)
    """
    total_mass = sum(e.total_mass for e in elements)
    if total_mass == 0:
        if not elements:
            return 0.0
        # If no mass data, use simple average
        return sum(e.element_circularity_indicator() for e in elements) / len(elements)

    return sum(
        e.total_mass * e.element_circularity_indicator() for e in elements
    ) / total_mass
