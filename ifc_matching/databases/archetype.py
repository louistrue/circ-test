"""Abstracted archetype database layer.

Provides a unified interface for loading material/product data from
different international LCA databases (TOTEM, OKOBAUDAT, KBOB) and
mapping their connection types to the standardised Durmisevic
disassembly scale (0.1–1.0).

This addresses the paper's limitation of being locked to the Belgian
TOTEM database and validates whether the fused classification system
holds across international standards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ======================================================================
# Durmisevic connection type scale
# ======================================================================

class ConnectionType(Enum):
    """Standardised connection types on the Durmisevic scale.

    Values represent the Disassembly Potential score (0.1 = worst, 1.0 = best).
    """

    CHEMICAL_BOND = 0.1          # Adhesive, resin, mortar
    SOFT_CHEMICAL_BOND = 0.2     # Silicone, foam sealant
    WELDED = 0.3                 # Metal welds
    NAILED_STAPLED = 0.4         # Nails, staples
    SCREWED = 0.5                # Wood screws, self-tapping
    BOLTED = 0.6                 # Bolted connections
    CLAMPED = 0.7                # Clamps, clips
    SNAP_FIT = 0.8               # Click systems
    FRICTION_BASED = 0.9         # Gravity, friction, resting
    DRY_LOOSE = 1.0              # Loose-laid, removable

    @classmethod
    def from_string(cls, name: str) -> ConnectionType:
        """Fuzzy lookup by name substring."""
        name_lower = name.lower().strip()
        mapping = {
            "chemical": cls.CHEMICAL_BOND,
            "adhesive": cls.CHEMICAL_BOND,
            "mortar": cls.CHEMICAL_BOND,
            "resin": cls.CHEMICAL_BOND,
            "glue": cls.CHEMICAL_BOND,
            "silicone": cls.SOFT_CHEMICAL_BOND,
            "foam": cls.SOFT_CHEMICAL_BOND,
            "sealant": cls.SOFT_CHEMICAL_BOND,
            "weld": cls.WELDED,
            "nail": cls.NAILED_STAPLED,
            "staple": cls.NAILED_STAPLED,
            "screw": cls.SCREWED,
            "bolt": cls.BOLTED,
            "clamp": cls.CLAMPED,
            "clip": cls.CLAMPED,
            "snap": cls.SNAP_FIT,
            "click": cls.SNAP_FIT,
            "friction": cls.FRICTION_BASED,
            "gravity": cls.FRICTION_BASED,
            "dry": cls.DRY_LOOSE,
            "loose": cls.DRY_LOOSE,
        }
        for key, conn in mapping.items():
            if key in name_lower:
                return conn
        return cls.CHEMICAL_BOND  # conservative default


@dataclass
class MaterialEntry:
    """A single material record from a database."""

    id: str
    name: str
    category: str = ""
    density: float | None = None
    unit: str = "kg"
    gwp: float | None = None  # Global Warming Potential (kg CO2 eq)
    recyclability: float | None = None  # 0.0–1.0
    default_connection: ConnectionType = ConnectionType.CHEMICAL_BOND
    source_db: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchetypeEntry:
    """A building component archetype composed of material layers."""

    id: str
    name: str
    element_type: str  # e.g., "IfcWall", "IfcSlab"
    layers: list[MaterialEntry] = field(default_factory=list)
    connections: list[ConnectionType] = field(default_factory=list)
    source_db: str = ""

    @property
    def disassembly_score(self) -> float:
        """Average disassembly potential from Durmisevic connection scores."""
        if not self.connections:
            return 0.1
        return sum(c.value for c in self.connections) / len(self.connections)


# ======================================================================
# Abstract database interface
# ======================================================================

class ArchetypeDatabase(ABC):
    """Interface for loading archetype databases."""

    @property
    @abstractmethod
    def db_name(self) -> str:
        """Short identifier for this database (e.g., 'TOTEM')."""

    @abstractmethod
    def load(self, source: str | Path) -> None:
        """Load database from a file path, URL, or other identifier."""

    @abstractmethod
    def get_materials(self) -> list[MaterialEntry]:
        """Return all material entries."""

    @abstractmethod
    def get_archetypes(self) -> list[ArchetypeEntry]:
        """Return all archetype entries."""

    def get_material_names(self) -> list[str]:
        """Convenience: return just the names for matching."""
        return [m.name for m in self.get_materials()]

    def get_archetype_names(self) -> list[str]:
        """Convenience: return just the names for matching."""
        return [a.name for a in self.get_archetypes()]


# ======================================================================
# TOTEM database (Belgian)
# ======================================================================

class TOTEMDatabase(ArchetypeDatabase):
    """Belgian TOTEM environmental database.

    Expected input format: CSV or JSON with columns for material name,
    category, GWP values, and connection metadata.
    """

    def __init__(self) -> None:
        self._materials: list[MaterialEntry] = []
        self._archetypes: list[ArchetypeEntry] = []

    @property
    def db_name(self) -> str:
        return "TOTEM"

    def load(self, source: str | Path) -> None:
        path = Path(source)
        if path.suffix == ".json":
            self._load_json(path)
        elif path.suffix == ".csv":
            self._load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_json(self, path: Path) -> None:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("materials", []):
            conn = ConnectionType.from_string(item.get("connection_type", "chemical"))
            self._materials.append(
                MaterialEntry(
                    id=item.get("id", ""),
                    name=item["name"],
                    category=item.get("category", ""),
                    density=item.get("density"),
                    gwp=item.get("gwp"),
                    recyclability=item.get("recyclability"),
                    default_connection=conn,
                    source_db=self.db_name,
                )
            )

        for item in data.get("archetypes", []):
            layers = [
                MaterialEntry(
                    id=l.get("id", ""),
                    name=l["name"],
                    source_db=self.db_name,
                )
                for l in item.get("layers", [])
            ]
            connections = [
                ConnectionType.from_string(c) for c in item.get("connections", [])
            ]
            self._archetypes.append(
                ArchetypeEntry(
                    id=item.get("id", ""),
                    name=item["name"],
                    element_type=item.get("element_type", ""),
                    layers=layers,
                    connections=connections,
                    source_db=self.db_name,
                )
            )

    def _load_csv(self, path: Path) -> None:
        import csv

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conn = ConnectionType.from_string(row.get("connection_type", "chemical"))
                self._materials.append(
                    MaterialEntry(
                        id=row.get("id", ""),
                        name=row["name"],
                        category=row.get("category", ""),
                        density=float(row["density"]) if row.get("density") else None,
                        gwp=float(row["gwp"]) if row.get("gwp") else None,
                        recyclability=float(row["recyclability"]) if row.get("recyclability") else None,
                        default_connection=conn,
                        source_db=self.db_name,
                    )
                )

    def get_materials(self) -> list[MaterialEntry]:
        return self._materials

    def get_archetypes(self) -> list[ArchetypeEntry]:
        return self._archetypes


# ======================================================================
# ÖKOBAUDAT database (German)
# ======================================================================

class OKOBAUDATDatabase(ArchetypeDatabase):
    """German ÖKOBAUDAT environmental database.

    Expected input format: CSV or JSON export from the ÖKOBAUDAT portal.
    The database uses EPD (Environmental Product Declaration) categories.
    """

    def __init__(self) -> None:
        self._materials: list[MaterialEntry] = []
        self._archetypes: list[ArchetypeEntry] = []

    @property
    def db_name(self) -> str:
        return "OKOBAUDAT"

    def load(self, source: str | Path) -> None:
        path = Path(source)
        if path.suffix == ".json":
            self._load_json(path)
        elif path.suffix == ".csv":
            self._load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_json(self, path: Path) -> None:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("materials", data.get("datasets", [])):
            conn = ConnectionType.from_string(item.get("connection_type", "chemical"))
            self._materials.append(
                MaterialEntry(
                    id=item.get("uuid", item.get("id", "")),
                    name=item.get("name", item.get("name_de", "")),
                    category=item.get("category", item.get("kategorie", "")),
                    density=item.get("density", item.get("rohdichte")),
                    gwp=item.get("gwp", item.get("gwp_total")),
                    recyclability=item.get("recyclability"),
                    default_connection=conn,
                    source_db=self.db_name,
                )
            )

    def _load_csv(self, path: Path) -> None:
        import csv

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")  # German CSVs often use ;
            for row in reader:
                name = row.get("name", row.get("Name", row.get("Bezeichnung", "")))
                conn = ConnectionType.from_string(row.get("connection_type", "chemical"))
                self._materials.append(
                    MaterialEntry(
                        id=row.get("uuid", row.get("UUID", "")),
                        name=name,
                        category=row.get("category", row.get("Kategorie", "")),
                        density=float(row.get("density", row.get("Rohdichte", 0)) or 0) or None,
                        gwp=float(row.get("gwp", row.get("GWP", 0)) or 0) or None,
                        default_connection=conn,
                        source_db=self.db_name,
                    )
                )

    def get_materials(self) -> list[MaterialEntry]:
        return self._materials

    def get_archetypes(self) -> list[ArchetypeEntry]:
        return self._archetypes


# ======================================================================
# KBOB database (Swiss)
# ======================================================================

class KBOBDatabase(ArchetypeDatabase):
    """Swiss KBOB environmental database.

    Expected input format: CSV or JSON export.  KBOB uses its own
    category structure with UBP (Umweltbelastungspunkte) scores.
    """

    def __init__(self) -> None:
        self._materials: list[MaterialEntry] = []
        self._archetypes: list[ArchetypeEntry] = []

    @property
    def db_name(self) -> str:
        return "KBOB"

    def load(self, source: str | Path) -> None:
        path = Path(source)
        if path.suffix == ".json":
            self._load_json(path)
        elif path.suffix == ".csv":
            self._load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_json(self, path: Path) -> None:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("materials", data.get("baumaterialien", [])):
            conn = ConnectionType.from_string(item.get("connection_type", "chemical"))
            self._materials.append(
                MaterialEntry(
                    id=item.get("id", item.get("kbob_id", "")),
                    name=item.get("name", item.get("bezeichnung", "")),
                    category=item.get("category", item.get("gruppe", "")),
                    density=item.get("density", item.get("rohdichte")),
                    gwp=item.get("gwp", item.get("treibhausgasemissionen")),
                    recyclability=item.get("recyclability"),
                    default_connection=conn,
                    source_db=self.db_name,
                    extra={"ubp": item.get("ubp")},
                )
            )

    def _load_csv(self, path: Path) -> None:
        import csv

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                name = row.get("name", row.get("Bezeichnung", ""))
                conn = ConnectionType.from_string(row.get("connection_type", "chemical"))
                self._materials.append(
                    MaterialEntry(
                        id=row.get("id", row.get("KBOB-ID", "")),
                        name=name,
                        category=row.get("category", row.get("Gruppe", "")),
                        density=float(row.get("density", row.get("Rohdichte", 0)) or 0) or None,
                        gwp=float(row.get("gwp", row.get("GWP", 0)) or 0) or None,
                        default_connection=conn,
                        source_db=self.db_name,
                        extra={"ubp": float(row.get("UBP", 0) or 0) or None},
                    )
                )

    def get_materials(self) -> list[MaterialEntry]:
        return self._materials

    def get_archetypes(self) -> list[ArchetypeEntry]:
        return self._archetypes


# ======================================================================
# Registry / factory
# ======================================================================

_REGISTRY: dict[str, type[ArchetypeDatabase]] = {
    "totem": TOTEMDatabase,
    "okobaudat": OKOBAUDATDatabase,
    "kbob": KBOBDatabase,
}


def get_database(name: str) -> ArchetypeDatabase:
    """Instantiate a database by short name.

    >>> db = get_database("totem")
    >>> db.load("path/to/totem_data.json")
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown database {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]()


def register_database(name: str, cls: type[ArchetypeDatabase]) -> None:
    """Register a custom database implementation."""
    _REGISTRY[name.lower().strip()] = cls
