"""Neo4j graph store with write-back capability.

Turns the circularity assessment viewer from read-only to an
interactive editor.  Users can modify connection types and trigger
real-time BCI recalculation.
"""

from __future__ import annotations

from typing import Any

from ifc_matching.databases.archetype import ConnectionType
from ifc_matching.graph.circularity import (
    MaterialNode,
    ConnectionEdge,
    ElementAssembly,
    building_circularity_indicator,
)


class Neo4jCircularityStore:
    """Manages circularity graph data in Neo4j with write-back support.

    Parameters
    ----------
    uri:
        Neo4j connection URI (e.g., ``bolt://localhost:7687``).
    auth:
        Tuple of ``(username, password)``.
    database:
        Neo4j database name (default ``neo4j``).
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "password"),
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.auth = auth
        self.database = database
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=self.auth)
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create constraints and indexes for the circularity graph."""
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (e:Element) REQUIRE e.element_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (m:Material) REQUIRE m.id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Element) ON (e.element_type)"
            )

    # ------------------------------------------------------------------
    # Write: push element assemblies to graph
    # ------------------------------------------------------------------

    def upsert_element(self, assembly: ElementAssembly) -> None:
        """Write or update an element assembly in Neo4j."""
        driver = self._get_driver()
        eci = assembly.element_circularity_indicator()

        with driver.session(database=self.database) as session:
            # Create/update element node
            session.run(
                "MERGE (e:Element {element_id: $eid}) "
                "SET e.element_type = $etype, "
                "    e.name = $name, "
                "    e.eci = $eci, "
                "    e.mci = $mci, "
                "    e.di = $di, "
                "    e.total_mass = $mass",
                eid=assembly.element_id,
                etype=assembly.element_type,
                name=assembly.name,
                eci=eci,
                mci=assembly.material_circularity_index(),
                di=assembly.disassembly_index(),
                mass=assembly.total_mass,
            )

            # Create material nodes and relationships
            for mat in assembly.materials:
                session.run(
                    "MERGE (m:Material {id: $mid}) "
                    "SET m.name = $name, "
                    "    m.mass = $mass, "
                    "    m.recyclability = $rec, "
                    "    m.reuse_potential = $reuse "
                    "WITH m "
                    "MATCH (e:Element {element_id: $eid}) "
                    "MERGE (e)-[:HAS_MATERIAL]->(m)",
                    mid=mat.id,
                    name=mat.name,
                    mass=mat.mass,
                    rec=mat.recyclability,
                    reuse=mat.reuse_potential,
                    eid=assembly.element_id,
                )

            # Create connection edges
            for conn in assembly.connections:
                session.run(
                    "MATCH (s:Material {id: $sid}), (t:Material {id: $tid}) "
                    "MERGE (s)-[c:CONNECTED_TO {id: $cid}]->(t) "
                    "SET c.connection_type = $ctype, "
                    "    c.disassembly_score = $score",
                    sid=conn.source_id,
                    tid=conn.target_id,
                    cid=conn.id,
                    ctype=conn.connection_type.name,
                    score=conn.disassembly_score,
                )

    # ------------------------------------------------------------------
    # Write-back: update connection type (the key editor feature)
    # ------------------------------------------------------------------

    def update_connection(
        self,
        connection_id: str,
        new_type: ConnectionType,
    ) -> dict[str, float]:
        """Update a connection type and recalculate affected metrics.

        This is the core "write-back" operation: when a user clicks a
        connection in the viewer and changes it (e.g., from
        "Chemical Bond - 0.1" to "Mechanical Bolt - 0.6"), this method:

        1. Updates the connection edge in Neo4j.
        2. Recalculates the Disassembly Index (DI) of affected elements.
        3. Recalculates the Element Circularity Indicator (ECI).
        4. Recalculates the Building Circularity Indicator (BCI).

        Returns
        -------
        dict with keys: ``connection_id``, ``new_type``, ``new_score``,
        ``affected_element_eci``, ``new_bci``.
        """
        driver = self._get_driver()

        with driver.session(database=self.database) as session:
            # 1. Update the connection
            session.run(
                "MATCH ()-[c:CONNECTED_TO {id: $cid}]->() "
                "SET c.connection_type = $ctype, "
                "    c.disassembly_score = $score",
                cid=connection_id,
                ctype=new_type.name,
                score=new_type.value,
            )

            # 2. Find affected elements and recalculate
            result = session.run(
                "MATCH (e:Element)-[:HAS_MATERIAL]->(m:Material)"
                "-[c:CONNECTED_TO]->(:Material)<-[:HAS_MATERIAL]-(e) "
                "WHERE c.id = $cid "
                "RETURN DISTINCT e.element_id AS eid",
                cid=connection_id,
            )
            affected_eids = [r["eid"] for r in result]

            # 3. Recalculate ECI for affected elements
            affected_ecis = {}
            for eid in affected_eids:
                assembly = self._load_element(session, eid)
                if assembly:
                    eci = assembly.element_circularity_indicator()
                    affected_ecis[eid] = eci
                    session.run(
                        "MATCH (e:Element {element_id: $eid}) "
                        "SET e.eci = $eci, e.di = $di, e.mci = $mci",
                        eid=eid,
                        eci=eci,
                        di=assembly.disassembly_index(),
                        mci=assembly.material_circularity_index(),
                    )

            # 4. Recalculate BCI
            new_bci = self.calculate_bci()

        return {
            "connection_id": connection_id,
            "new_type": new_type.name,
            "new_score": new_type.value,
            "affected_elements": affected_ecis,
            "new_bci": new_bci,
        }

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _load_element(self, session, element_id: str) -> ElementAssembly | None:
        """Load a single element assembly from the graph."""
        # Get element
        result = session.run(
            "MATCH (e:Element {element_id: $eid}) RETURN e",
            eid=element_id,
        )
        record = result.single()
        if not record:
            return None
        e = record["e"]

        # Get materials
        mat_result = session.run(
            "MATCH (e:Element {element_id: $eid})-[:HAS_MATERIAL]->(m:Material) "
            "RETURN m",
            eid=element_id,
        )
        materials = []
        for r in mat_result:
            m = r["m"]
            materials.append(
                MaterialNode(
                    id=m["id"],
                    name=m.get("name", ""),
                    mass=m.get("mass", 0.0),
                    recyclability=m.get("recyclability", 0.0),
                    reuse_potential=m.get("reuse_potential", 0.0),
                )
            )

        # Get connections between this element's materials
        conn_result = session.run(
            "MATCH (e:Element {element_id: $eid})-[:HAS_MATERIAL]->(s:Material) "
            "-[c:CONNECTED_TO]->(t:Material)<-[:HAS_MATERIAL]-(e) "
            "RETURN c, s.id AS sid, t.id AS tid",
            eid=element_id,
        )
        connections = []
        for r in conn_result:
            c = r["c"]
            ctype_name = c.get("connection_type", "CHEMICAL_BOND")
            try:
                ctype = ConnectionType[ctype_name]
            except KeyError:
                ctype = ConnectionType.CHEMICAL_BOND

            connections.append(
                ConnectionEdge(
                    id=c.get("id", ""),
                    source_id=r["sid"],
                    target_id=r["tid"],
                    connection_type=ctype,
                )
            )

        return ElementAssembly(
            element_id=element_id,
            element_type=e.get("element_type", ""),
            name=e.get("name", ""),
            materials=materials,
            connections=connections,
        )

    def get_all_elements(self) -> list[ElementAssembly]:
        """Load all element assemblies from the graph."""
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            result = session.run("MATCH (e:Element) RETURN e.element_id AS eid")
            eids = [r["eid"] for r in result]

            assemblies = []
            for eid in eids:
                a = self._load_element(session, eid)
                if a:
                    assemblies.append(a)
            return assemblies

    def calculate_bci(self) -> float:
        """Calculate the Building Circularity Indicator across all elements."""
        elements = self.get_all_elements()
        return building_circularity_indicator(elements)

    # ------------------------------------------------------------------
    # "What-if" scenario support
    # ------------------------------------------------------------------

    def simulate_connection_change(
        self,
        connection_id: str,
        new_type: ConnectionType,
    ) -> float:
        """Simulate a connection change and return the projected BCI.

        Unlike ``update_connection``, this does NOT persist the change.
        It reads the current state, applies the change in-memory, and
        returns what the BCI would be.
        """
        elements = self.get_all_elements()

        for element in elements:
            for conn in element.connections:
                if conn.id == connection_id:
                    conn.connection_type = new_type

        return building_circularity_indicator(elements)
