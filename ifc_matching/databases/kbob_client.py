"""Client for the lcadata.ch KBOB API.

Provides access to Swiss KBOB (Koordinationskonferenz der Bau- und
Liegenschaftsorgane der öffentlichen Bauherren) construction material
environmental data.

API: https://lcadata.ch/api-access

Usage::

    from ifc_matching.databases.kbob_client import KBOBClient

    client = KBOBClient(api_key="your-key")  # or from LCADATA_API_KEY env var
    materials = client.get_all_materials()
    names = client.get_material_names()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error


@dataclass
class KBOBMaterial:
    """A single material entry from the KBOB database."""

    id: str
    name_de: str  # German name (original)
    name_fr: str = ""  # French name (if available)
    density: float | None = None  # kg/m³
    unit: str = ""
    gwp_total: float | None = None  # kg CO₂ eq
    ubp_total: float | None = None  # UBP (Umweltbelastungspunkte)
    penre_total: float | None = None  # Primary energy non-renewable (kWh oil-eq)
    group: str = ""  # KBOB category / group
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Primary name (German)."""
        return self.name_de

    def __repr__(self) -> str:
        gwp = f", gwp={self.gwp_total:.1f}" if self.gwp_total is not None else ""
        return f"KBOBMaterial({self.id!r}, {self.name_de!r}{gwp})"


class KBOBClient:
    """Client for the lcadata.ch KBOB API.

    Parameters
    ----------
    api_key:
        API key for lcadata.ch. If not provided, reads from
        ``LCADATA_API_KEY`` environment variable.
    base_url:
        Base URL for the API (default: ``https://lcadata.ch``).
    version:
        KBOB data version (e.g., ``"7.0"``). If None, uses the current
        version from the API.
    cache_path:
        Optional path to cache the full materials list as JSON.
        Avoids repeated API calls during development/benchmarking.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://lcadata.ch",
        version: str | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LCADATA_API_KEY", "") or os.environ.get("api_key", "")
        if not self.api_key:
            raise ValueError(
                "KBOB API key required. Set LCADATA_API_KEY env var or pass api_key=."
            )
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.cache_path = Path(cache_path) if cache_path else None
        self._materials: list[KBOBMaterial] | None = None

    # ------------------------------------------------------------------
    # Low-level API calls
    # ------------------------------------------------------------------

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> Any:
        """Make an authenticated GET request to the API."""
        url = f"{self.base_url}/api/kbob/{endpoint}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        req = urllib.request.Request(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"KBOB API error {e.code}: {body}"
            ) from e

    # ------------------------------------------------------------------
    # Material retrieval
    # ------------------------------------------------------------------

    def get_all_materials(self, *, force_refresh: bool = False) -> list[KBOBMaterial]:
        """Fetch all materials from the KBOB database.

        Uses cache if available (and not force_refresh).
        """
        if self._materials is not None and not force_refresh:
            return self._materials

        # Try cache first
        if self.cache_path and self.cache_path.exists() and not force_refresh:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._materials = [self._parse_material(item) for item in data]
            return self._materials

        # Fetch from API -- use /materials/all for complete dataset
        params = {}
        if self.version:
            params["version"] = self.version
        data = self._request("materials/all", params)

        # API may return list directly or wrapped in an object
        if isinstance(data, dict):
            items = data.get("materials", data.get("data", []))
        elif isinstance(data, list):
            items = data
        else:
            items = []

        self._materials = [self._parse_material(item) for item in items]

        # Write cache
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return self._materials

    def get_random_materials(self, count: int = 10) -> list[KBOBMaterial]:
        """Fetch random materials (useful for quick testing)."""
        params = {"count": str(min(count, 10))}
        if self.version:
            params["version"] = self.version
        data = self._request("materials/random", params)

        if isinstance(data, dict):
            items = data.get("materials", data.get("data", []))
        elif isinstance(data, list):
            items = data
        else:
            items = []

        return [self._parse_material(item) for item in items]

    def search_materials(self, query: str) -> list[KBOBMaterial]:
        """Search materials by name."""
        params = {"q": query}
        if self.version:
            params["version"] = self.version
        data = self._request("materials/search", params)

        if isinstance(data, dict):
            items = data.get("materials", data.get("data", []))
        elif isinstance(data, list):
            items = data
        else:
            items = []

        return [self._parse_material(item) for item in items]

    def get_material_names(self, *, force_refresh: bool = False) -> list[str]:
        """Return just the German names of all materials."""
        return [m.name_de for m in self.get_all_materials(force_refresh=force_refresh)]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_material(item: dict[str, Any]) -> KBOBMaterial:
        """Parse a single material entry from the API response."""
        return KBOBMaterial(
            id=str(item.get("id", item.get("ID", ""))),
            name_de=item.get("nameDE", item.get("name_de", item.get("Name", item.get("name", "")))),
            name_fr=item.get("nameFR", item.get("name_fr", "")),
            density=_safe_float(item.get("density", item.get("Density"))),
            unit=str(item.get("unit", item.get("Unit", ""))),
            gwp_total=_safe_float(item.get("gwpTotal", item.get("GWP Total", item.get("gwp_total")))),
            ubp_total=_safe_float(item.get("ubpTotal", item.get("UBP Total", item.get("ubp_total")))),
            penre_total=_safe_float(item.get("penreTotal", item.get("Primary Energy Non-Renewable Total"))),
            group=str(item.get("group", item.get("Group", item.get("kategorie", "")))),
            extra={k: v for k, v in item.items() if k not in {
                "id", "ID", "nameDE", "name_de", "Name", "name",
                "nameFR", "name_fr", "density", "Density",
                "unit", "Unit", "gwpTotal", "GWP Total", "gwp_total",
                "ubpTotal", "UBP Total", "ubp_total",
                "penreTotal", "Primary Energy Non-Renewable Total",
                "group", "Group", "kategorie",
            }},
        )


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None
