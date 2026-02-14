"""Extract materials and element metadata from IFC files.

Uses ifcopenshell to parse real IFC models and produce structured
data suitable for the matching pipeline.

Usage::

    from ifc_matching.ifc_extractor import IfcMaterialExtractor

    extractor = IfcMaterialExtractor("path/to/model.ifc")
    elements = extractor.extract_all()

    for el in elements:
        print(el.element_type, el.name, el.materials)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MaterialLayer:
    """A single material layer within an element."""

    name: str
    thickness: float | None = None  # metres
    category: str = ""
    is_ventilated: bool = False


@dataclass
class IfcElementData:
    """Structured data extracted from a single IFC element."""

    global_id: str
    element_type: str  # e.g. "IfcWall", "IfcSlab", "IfcColumn"
    name: str  # Element name from IFC
    description: str = ""
    type_name: str = ""  # From IfcTypeObject if available
    predefined_type: str = ""
    materials: list[MaterialLayer] = field(default_factory=list)
    properties: dict[str, str | float | bool] = field(default_factory=dict)
    quantities: dict[str, float] = field(default_factory=dict)
    classification: str = ""  # e.g. DIN 276 cost group

    @property
    def material_names(self) -> list[str]:
        """Flat list of material names for matching."""
        return [m.name for m in self.materials if m.name]

    @property
    def primary_material(self) -> str:
        """Best single material name for matching (thickest layer or first)."""
        if not self.materials:
            return self.name  # Fall back to element name
        # Pick thickest layer if thicknesses are available
        with_thickness = [m for m in self.materials if m.thickness]
        if with_thickness:
            return max(with_thickness, key=lambda m: m.thickness or 0).name
        return self.materials[0].name

    @property
    def match_query(self) -> str:
        """Construct the best query string for database matching.

        Prioritises: primary material name > element name > type name.
        """
        if self.materials:
            return self.primary_material
        if self.name and self.name.lower() not in ("default", "generic", "n/a", ""):
            return self.name
        if self.type_name:
            return self.type_name
        return self.name or self.element_type


class IfcMaterialExtractor:
    """Extract material and element data from an IFC file.

    Parameters
    ----------
    ifc_path:
        Path to the IFC file.
    skip_types:
        IFC entity types to skip (e.g. IfcSpace, IfcOpeningElement).
    """

    # Element types that typically don't have materials or aren't relevant
    DEFAULT_SKIP = frozenset({
        "IfcSpace",
        "IfcOpeningElement",
        "IfcSite",
        "IfcBuilding",
        "IfcBuildingStorey",
        "IfcProject",
        "IfcAnnotation",
        "IfcGrid",
        "IfcVirtualElement",
        "IfcDistributionPort",
    })

    def __init__(
        self,
        ifc_path: str | Path,
        *,
        skip_types: frozenset[str] | None = None,
    ) -> None:
        self.ifc_path = str(ifc_path)
        self.skip_types = skip_types or self.DEFAULT_SKIP
        self._ifc_file = None

    def _open(self):
        """Lazy-load the IFC file."""
        if self._ifc_file is None:
            import ifcopenshell

            self._ifc_file = ifcopenshell.open(self.ifc_path)
        return self._ifc_file

    # ------------------------------------------------------------------
    # Material extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_materials(element) -> list[MaterialLayer]:
        """Extract material layers from an IFC element via IfcRelAssociatesMaterial."""
        import ifcopenshell.util.element as elem_util

        materials: list[MaterialLayer] = []

        try:
            material = elem_util.get_material(element)
        except Exception:
            return materials

        if material is None:
            return materials

        mat_type = material.is_a()

        if mat_type == "IfcMaterial":
            materials.append(MaterialLayer(
                name=material.Name or "",
                category=getattr(material, "Category", "") or "",
            ))

        elif mat_type == "IfcMaterialLayerSet":
            for layer in material.MaterialLayers:
                mat = layer.Material
                materials.append(MaterialLayer(
                    name=mat.Name if mat else "",
                    thickness=layer.LayerThickness,
                    category=getattr(mat, "Category", "") or "" if mat else "",
                    is_ventilated=getattr(layer, "IsVentilated", False) or False,
                ))

        elif mat_type == "IfcMaterialLayerSetUsage":
            layer_set = material.ForLayerSet
            if layer_set:
                for layer in layer_set.MaterialLayers:
                    mat = layer.Material
                    materials.append(MaterialLayer(
                        name=mat.Name if mat else "",
                        thickness=layer.LayerThickness,
                        category=getattr(mat, "Category", "") or "" if mat else "",
                        is_ventilated=getattr(layer, "IsVentilated", False) or False,
                    ))

        elif mat_type == "IfcMaterialConstituentSet":
            for constituent in (material.MaterialConstituents or []):
                mat = constituent.Material
                materials.append(MaterialLayer(
                    name=mat.Name if mat else (constituent.Name or ""),
                    category=getattr(constituent, "Category", "") or "",
                ))

        elif mat_type == "IfcMaterialProfileSet":
            for profile in (material.MaterialProfiles or []):
                mat = profile.Material
                materials.append(MaterialLayer(
                    name=mat.Name if mat else (profile.Name or ""),
                    category=getattr(mat, "Category", "") or "" if mat else "",
                ))

        elif mat_type == "IfcMaterialProfileSetUsage":
            profile_set = material.ForProfileSet
            if profile_set:
                for profile in (profile_set.MaterialProfiles or []):
                    mat = profile.Material
                    materials.append(MaterialLayer(
                        name=mat.Name if mat else (profile.Name or ""),
                        category=getattr(mat, "Category", "") or "" if mat else "",
                    ))

        elif mat_type == "IfcMaterialList":
            for mat in (material.Materials or []):
                materials.append(MaterialLayer(
                    name=mat.Name or "",
                ))

        return materials

    # ------------------------------------------------------------------
    # Property extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_properties(element) -> dict[str, str | float | bool]:
        """Extract key properties from psets."""
        import ifcopenshell.util.element as elem_util

        props: dict[str, str | float | bool] = {}
        try:
            psets = elem_util.get_psets(element)
        except Exception:
            return props

        # Flatten psets into a single dict, keeping useful properties
        interesting = {
            "ThermalTransmittance", "Thermal Transmittance",
            "IsExternal", "Is External",
            "LoadBearing", "Load Bearing",
            "FireRating", "Fire Rating",
            "Reference",
            "AcousticRating", "Acoustic Rating",
            "Combustible",
            "SurfaceSpreadOfFlame",
            "Status",
        }

        for pset_name, pset_data in psets.items():
            if isinstance(pset_data, dict):
                for key, val in pset_data.items():
                    if key in interesting or key == "id":
                        continue
                    # Only include properties with actual values
                    if val is not None and val != "" and key in interesting:
                        props[key] = val

                # Also grab any property that looks useful
                for key, val in pset_data.items():
                    if key == "id":
                        continue
                    if key in interesting and val is not None and val != "":
                        props[key] = val

        return props

    # ------------------------------------------------------------------
    # Quantity extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_quantities(element) -> dict[str, float]:
        """Extract base quantities (area, volume, etc.)."""
        import ifcopenshell.util.element as elem_util

        quantities: dict[str, float] = {}
        try:
            qtos = elem_util.get_psets(element, qtos_only=True)
        except Exception:
            return quantities

        for qto_name, qto_data in qtos.items():
            if isinstance(qto_data, dict):
                for key, val in qto_data.items():
                    if key == "id":
                        continue
                    if isinstance(val, (int, float)) and val > 0:
                        quantities[key] = float(val)

        return quantities

    # ------------------------------------------------------------------
    # Type / classification
    # ------------------------------------------------------------------

    @staticmethod
    def _get_type_info(element) -> tuple[str, str]:
        """Get type name and predefined type."""
        import ifcopenshell.util.element as elem_util

        type_name = ""
        predefined = ""

        try:
            etype = elem_util.get_type(element)
            if etype:
                type_name = getattr(etype, "Name", "") or ""
                predefined = getattr(etype, "PredefinedType", "") or ""
        except Exception:
            pass

        if not predefined:
            predefined = getattr(element, "PredefinedType", "") or ""

        return type_name, predefined

    @staticmethod
    def _get_classification(element) -> str:
        """Get classification reference (e.g. Uniclass, DIN 276)."""
        try:
            for rel in getattr(element, "HasAssociations", []):
                if rel.is_a("IfcRelAssociatesClassification"):
                    ref = rel.RelatingClassification
                    if hasattr(ref, "Identification"):
                        return f"{ref.Identification or ''} {ref.Name or ''}".strip()
                    elif hasattr(ref, "ItemReference"):
                        return f"{ref.ItemReference or ''} {ref.Name or ''}".strip()
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_all(self) -> list[IfcElementData]:
        """Extract structured data for all relevant elements."""
        ifc_file = self._open()
        results: list[IfcElementData] = []

        for element in ifc_file.by_type("IfcProduct"):
            etype = element.is_a()

            # Skip non-physical elements
            if etype in self.skip_types:
                continue

            # Skip elements without a name
            name = getattr(element, "Name", "") or ""

            type_name, predefined = self._get_type_info(element)
            materials = self._get_materials(element)
            properties = self._get_properties(element)
            quantities = self._get_quantities(element)
            classification = self._get_classification(element)

            results.append(IfcElementData(
                global_id=str(element.GlobalId),
                element_type=etype,
                name=name,
                description=getattr(element, "Description", "") or "",
                type_name=type_name,
                predefined_type=predefined,
                materials=materials,
                properties=properties,
                quantities=quantities,
                classification=classification,
            ))

        return results

    def extract_for_matching(self) -> list[dict[str, str]]:
        """Extract minimal data needed for the matching pipeline.

        Returns a list of dicts with keys:
        - ``query``: the material/product name to match
        - ``element_type``: the IFC element type
        - ``global_id``: the element's GlobalId
        - ``name``: the element's name
        - ``layers``: semicolon-separated material layer names
        """
        elements = self.extract_all()
        match_data = []

        for el in elements:
            # Skip elements with no useful name or materials
            if not el.match_query:
                continue

            # If element has multiple material layers, create one entry per layer
            if el.materials:
                for mat in el.materials:
                    if mat.name:
                        match_data.append({
                            "query": mat.name,
                            "element_type": el.element_type,
                            "global_id": el.global_id,
                            "element_name": el.name,
                            "layer_thickness": str(mat.thickness) if mat.thickness else "",
                            "all_layers": "; ".join(el.material_names),
                        })
            else:
                # Fall back to element name
                match_data.append({
                    "query": el.match_query,
                    "element_type": el.element_type,
                    "global_id": el.global_id,
                    "element_name": el.name,
                    "layer_thickness": "",
                    "all_layers": "",
                })

        return match_data

    def summary(self) -> dict[str, int]:
        """Quick summary of what's in the IFC file."""
        elements = self.extract_all()
        type_counts: dict[str, int] = {}
        with_materials = 0
        total_layers = 0

        for el in elements:
            type_counts[el.element_type] = type_counts.get(el.element_type, 0) + 1
            if el.materials:
                with_materials += 1
                total_layers += len(el.materials)

        return {
            "total_elements": len(elements),
            "with_materials": with_materials,
            "total_material_layers": total_layers,
            "element_types": type_counts,
        }
