"""Client for the ÖKOBAUDAT (Ökobaudat) API.

Provides search access to the German ÖKOBAUDAT EPD database via the
ILCD/ServiceAPI. The database contains millions of EPDs, so search is
used instead of loading the full dataset.

API: https://www.oekobaudat.de/anleitungen/softwareentwickler.html
Example: https://github.com/ocni-dtu/okobau_example

Usage::

    from ifc_matching.databases.oekobaudat_client import OekobaudatClient

    client = OekobaudatClient()
    results = client.search_processes(name="Beton", limit=20)
    names = [r.display_name for r in results]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import urllib.request
import urllib.error


# Default datastock (ÖKOBAUDAT 2021-II)
DEFAULT_DATASTOCK_UUID = "cd2bda71-760b-4fcc-8a0b-3877c10000a8"

# EN 15804+A2 compliance system UUID (default per plan)
COMPLIANCE_EN15804_A2 = "c0016b33-8cf7-415c-ac6e-deba0d21440d"
# EN 15804+A1 compliance system UUID
COMPLIANCE_EN15804_A1 = "b00f9ec0-7874-11e3-981f-0800200c9a66"


@dataclass
class OekobaudatProcess:
    """A single process/EPD entry from ÖKOBAUDAT."""

    uuid: str
    name: str
    classific_id: str = ""  # e.g. "1.4.01"
    classific: str = ""  # e.g. "Mineralische Baustoffe / Mörtel und Beton / Beton"
    classific_system: str = ""
    sub_type: str = ""  # generic/specific dataset
    ref_year: int | None = None
    valid_until: int | None = None
    geo: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Primary display string for matching (name + category hint)."""
        return self.name

    def __repr__(self) -> str:
        return f"OekobaudatProcess({self.uuid!r}, {self.name!r})"


class OekobaudatClient:
    """Client for the ÖKOBAUDAT ILCD/ServiceAPI.

    Parameters
    ----------
    base_url:
        Base URL for the API (default: ÖKOBAUDAT resource root).
    datastock_uuid:
        Datastock UUID (default: 2021-II datastock).
    compliance_uuid:
        Compliance system UUID for filtering (default: EN 15804+A2).
    cache_path:
        Optional path to cache search results for repeated benchmarks.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://oekobaudat.de/OEKOBAU.DAT/resource",
        datastock_uuid: str = DEFAULT_DATASTOCK_UUID,
        compliance_uuid: str = COMPLIANCE_EN15804_A2,
        cache_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.datastock_uuid = datastock_uuid
        self.compliance_uuid = compliance_uuid
        self.cache_path = Path(cache_path) if cache_path else None

    def _build_url(
        self,
        *,
        name: str | None = None,
        class_id: str | None = None,
        start_index: int = 0,
        page_size: int = 50,
        compliance: str | None = None,
    ) -> str:
        """Build search URL for processes endpoint."""
        path = f"{self.base_url}/datastocks/{self.datastock_uuid}/processes"
        params: dict[str, str | int] = {
            "search": "true",
            "format": "json",
            "startIndex": start_index,
            "pageSize": page_size,
        }
        if name:
            params["name"] = name
        if class_id:
            params["classId"] = class_id
        params["compliance"] = compliance or self.compliance_uuid
        return f"{path}?{urlencode(params)}"

    def _request(self, url: str) -> dict[str, Any]:
        """Make GET request and return JSON."""
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ÖKOBAUDAT API error {e.code}: {body}") from e

    def _parse_process(self, item: dict[str, Any]) -> OekobaudatProcess:
        """Parse a single process from API response."""
        return OekobaudatProcess(
            uuid=str(item.get("uuid", "")),
            name=str(item.get("name", "")),
            classific_id=str(item.get("classificId", "")),
            classific=str(item.get("classific", "")),
            classific_system=str(item.get("classificSystem", "")),
            sub_type=str(item.get("subType", "")),
            ref_year=item.get("refYear"),
            valid_until=item.get("validUntil"),
            geo=str(item.get("geo", "")),
            extra={k: v for k, v in item.items() if k not in {
                "uuid", "name", "classificId", "classific", "classificSystem",
                "subType", "refYear", "validUntil", "geo",
            }},
        )

    def search_processes(
        self,
        name: str | None = None,
        class_id: str | None = None,
        limit: int = 50,
        start_index: int = 0,
        *,
        use_cache: bool = True,
    ) -> list[OekobaudatProcess]:
        """Search for processes/EPDs by name and optionally classId.

        Parameters
        ----------
        name:
            Search term (matches name and synonyms, AND for multiple words).
        class_id:
            ÖKOBAU.DAT class ID for category filter (e.g. "1.4.01" for Beton).
        limit:
            Max number of results to return (page size).
        start_index:
            Pagination offset.
        use_cache:
            If True and cache_path is set, use cached result when available.

        Returns
        -------
        List of OekobaudatProcess objects.
        """
        cache_key = f"search_{name or ''}_{class_id or ''}_{start_index}_{limit}"
        if use_cache and self.cache_path and self.cache_path.exists():
            try:
                cache_data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if cache_key in cache_data:
                    return [self._parse_process(i) for i in cache_data[cache_key]]
            except (json.JSONDecodeError, KeyError):
                pass

        url = self._build_url(
            name=name,
            class_id=class_id,
            start_index=start_index,
            page_size=limit,
        )
        data = self._request(url)

        items = data.get("data", [])
        if isinstance(items, dict):
            items = list(items.values()) if items else []
        results = [self._parse_process(i) for i in items]

        if use_cache and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                existing = {}
                if self.cache_path.exists():
                    existing = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            raw_items = [
                {
                    "uuid": p.uuid,
                    "name": p.name,
                    "classificId": p.classific_id,
                    "classific": p.classific,
                    "classificSystem": p.classific_system,
                    "subType": p.sub_type,
                    "refYear": p.ref_year,
                    "validUntil": p.valid_until,
                    "geo": p.geo,
                }
                for p in results
            ]
            existing[cache_key] = raw_items
            self.cache_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return results

    def get_display_names(
        self,
        name: str | None = None,
        class_id: str | None = None,
        limit: int = 50,
    ) -> list[str]:
        """Return display names from search (convenience for matching)."""
        results = self.search_processes(name=name, class_id=class_id, limit=limit)
        return [r.display_name for r in results]

    def get_id_lookup(
        self,
        name: str | None = None,
        class_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, str]:
        """Return mapping display_name -> uuid for final match output."""
        results = self.search_processes(name=name, class_id=class_id, limit=limit)
        return {r.display_name: r.uuid for r in results}

    def get_category_lookup(
        self,
        name: str | None = None,
        class_id: str | None = None,
        limit: int = 200,
    ) -> dict[str, str]:
        """Return mapping display_name -> classific_id for element-type filtering.

        Use this as candidate_ids in filter_candidates_by_element_type when
        source is ÖKOBAUDAT (classific_id uses prefixes like 1.4, 2.1).
        """
        results = self.search_processes(name=name, class_id=class_id, limit=limit)
        return {r.display_name: r.classific_id for r in results}
