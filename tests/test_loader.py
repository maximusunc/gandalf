"""Unit tests for gandalf.loader module."""

import os
import tempfile

import pytest

from gandalf.loader import build_graph_from_jsonl
from gandalf.graph import CSRGraph


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


class TestBuildGraphFromJsonl:
    """Tests for build_graph_from_jsonl function."""

    def test_returns_csr_graph(self, graph):
        """Should return a CSRGraph instance."""
        assert isinstance(graph, CSRGraph)

    def test_loads_correct_number_of_nodes(self, graph):
        """Should load all unique nodes from edges."""
        # Nodes in edges: CHEBI:6801, MONDO:0005148, NCBIGene:5468, NCBIGene:3643,
        # HP:0001943, CHEBI:17234, GO:0006006, NCBIGene:2645, NCBIGene:7124 (TNF)
        assert graph.num_nodes == 9

    def test_loads_edges(self, graph):
        """Should load edges into the graph."""
        # 10 edges in file, some may be duplicated for symmetric predicates
        assert len(graph.fwd_targets) >= 10

    def test_node_id_to_idx_mapping(self, graph):
        """Should create bidirectional node ID mapping."""
        # Check that all expected nodes are in the mapping
        expected_nodes = [
            "CHEBI:6801",
            "MONDO:0005148",
            "NCBIGene:5468",
            "NCBIGene:3643",
            "HP:0001943",
            "CHEBI:17234",
            "GO:0006006",
            "NCBIGene:2645",
            "NCBIGene:7124",  # TNF - added for qualifier constraint tests
        ]
        for node_id in expected_nodes:
            assert node_id in graph.node_id_to_idx
            idx = graph.node_id_to_idx[node_id]
            assert graph.idx_to_node_id[idx] == node_id

    def test_predicate_vocabulary(self, graph):
        """Should build predicate vocabulary from edges."""
        expected_predicates = [
            "biolink:treats",
            "biolink:affects",
            "biolink:gene_associated_with_condition",
            "biolink:has_phenotype",
            "biolink:interacts_with",
            "biolink:participates_in",
        ]
        for pred in expected_predicates:
            assert pred in graph.predicate_to_idx


class TestNodeProperties:
    """Tests for node properties loading."""

    def test_node_properties_loaded(self, graph):
        """Should load node properties from nodes.jsonl."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        assert metformin_idx in graph.node_properties

    def test_node_name_property(self, graph):
        """Should correctly load node name property."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        assert graph.get_node_property(metformin_idx, "name") == "Metformin"

    def test_node_category_property(self, graph):
        """Should correctly load node category property."""
        pparg_idx = graph.node_id_to_idx["NCBIGene:5468"]
        categories = graph.get_node_property(pparg_idx, "categories")
        assert "biolink:Gene" in categories

    def test_node_information_content(self, graph):
        """Should correctly load information_content property."""
        diabetes_idx = graph.node_id_to_idx["MONDO:0005148"]
        ic = graph.get_node_property(diabetes_idx, "information_content")
        assert ic == pytest.approx(78.2)


class TestEdgeProperties:
    """Tests for edge properties loading."""

    def test_edge_properties_stored(self, graph):
        """Should store edge properties."""
        assert len(graph.edge_properties) > 0

    def test_edge_predicate_property(self, graph):
        """Should correctly store predicate in edge properties."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["MONDO:0005148"]
        predicate = graph.get_edge_property(src_idx, dst_idx, "biolink:treats", "predicate")
        assert predicate == "biolink:treats"

    def test_edge_source_property(self, graph):
        """Should correctly store knowledge source."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["MONDO:0005148"]
        sources = graph.get_edge_property(src_idx, dst_idx, "biolink:treats", "sources")
        assert sources is not None
        assert len(sources) > 0
        assert sources[0]["resource_id"] == "infores:drugcentral"

    def test_edge_publications_property(self, graph):
        """Should correctly store publications."""
        src_idx = graph.node_id_to_idx["CHEBI:6801"]
        dst_idx = graph.node_id_to_idx["NCBIGene:5468"]
        pubs = graph.get_edge_property(src_idx, dst_idx, "biolink:affects", "publications")
        assert "PMID:23456789" in pubs


class TestGraphStructure:
    """Tests for CSR graph structure."""

    def test_neighbors_returns_correct_targets(self, graph):
        """Should return correct neighbor nodes."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(metformin_idx)

        # Metformin should connect to: MONDO:0005148, NCBIGene:5468, CHEBI:17234,
        # NCBIGene:3643 (INSR), NCBIGene:2645 (GCK), NCBIGene:7124 (TNF)
        neighbor_ids = {graph.idx_to_node_id[int(n)] for n in neighbors}
        assert "MONDO:0005148" in neighbor_ids
        assert "NCBIGene:5468" in neighbor_ids
        assert "CHEBI:17234" in neighbor_ids
        assert "NCBIGene:3643" in neighbor_ids  # INSR - qualifier test edge
        assert "NCBIGene:2645" in neighbor_ids  # GCK - qualifier test edge
        assert "NCBIGene:7124" in neighbor_ids  # TNF - qualifier test edge

    def test_neighbors_with_predicate_filter(self, graph):
        """Should filter neighbors by predicate."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(metformin_idx, predicate_filter="biolink:treats")

        neighbor_ids = {graph.idx_to_node_id[int(n)] for n in neighbors}
        assert "MONDO:0005148" in neighbor_ids
        # Should not include affects edges
        assert len(neighbor_ids) == 1

    def test_degree_calculation(self, graph):
        """Should correctly calculate node degree."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        degree = graph.degree(metformin_idx)
        # Metformin has 8 outgoing edges:
        # - treats, ameliorates, preventative (to MONDO:0005148)
        # - affects NCBIGene:5468 (PPARG)
        # - affects CHEBI:17234 (Glucose)
        # - affects NCBIGene:3643 (INSR) with qualifiers
        # - affects NCBIGene:2645 (GCK) with qualifiers
        # - affects NCBIGene:7124 (TNF) with qualifiers
        assert degree == 8

    def test_degree_with_predicate_filter(self, graph):
        """Should correctly calculate filtered degree."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        degree = graph.degree(metformin_idx, predicate_filter="biolink:treats")
        assert degree == 1

    def test_get_edges_returns_tuples(self, graph):
        """Should return edges as (neighbor_idx, predicate) tuples."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        edges = graph.get_edges(metformin_idx)

        # Metformin has 8 outgoing edges (including qualifier test edges)
        assert len(edges) == 8
        for neighbor_idx, predicate in edges:
            assert isinstance(neighbor_idx, int)
            assert isinstance(predicate, str)
            assert predicate.startswith("biolink:")


class TestGraphSaveLoad:
    """Tests for graph serialization."""

    def test_save_and_load_roundtrip(self, graph):
        """Should correctly save and load graph."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            graph.save(temp_path)
            loaded_graph = CSRGraph.load(temp_path)

            # Verify key attributes match
            assert loaded_graph.num_nodes == graph.num_nodes
            assert len(loaded_graph.fwd_targets) == len(graph.fwd_targets)
            assert loaded_graph.node_id_to_idx == graph.node_id_to_idx
            assert loaded_graph.predicate_to_idx == graph.predicate_to_idx
        finally:
            os.unlink(temp_path)

    def test_loaded_graph_neighbors_work(self, graph):
        """Should be able to query neighbors after load."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            graph.save(temp_path)
            loaded_graph = CSRGraph.load(temp_path)

            metformin_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            neighbors = loaded_graph.neighbors(metformin_idx)
            # Metformin has 8 outgoing edges (including qualifier test edges)
            assert len(neighbors) == 8
        finally:
            os.unlink(temp_path)


class TestGraphMmapSaveLoad:
    """Tests for memory-mapped graph serialization."""

    def test_save_mmap_creates_directory(self, graph):
        """Should create directory with expected files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)

            # Check all expected files exist
            expected_files = [
                "fwd_targets.npy",
                "fwd_predicates.npy",
                "fwd_offsets.npy",
                "rev_sources.npy",
                "rev_predicates.npy",
                "rev_offsets.npy",
                "metadata.pkl",
                "edge_properties.pkl",
            ]
            for filename in expected_files:
                assert os.path.exists(os.path.join(temp_dir, filename)), f"Missing {filename}"

    def test_mmap_save_and_load_roundtrip(self, graph):
        """Should correctly save and load graph in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Verify key attributes match
            assert loaded_graph.num_nodes == graph.num_nodes
            assert len(loaded_graph.fwd_targets) == len(graph.fwd_targets)
            assert loaded_graph.node_id_to_idx == graph.node_id_to_idx
            assert loaded_graph.predicate_to_idx == graph.predicate_to_idx

    def test_mmap_loaded_graph_neighbors_work(self, graph):
        """Should be able to query neighbors after mmap load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            metformin_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            neighbors = loaded_graph.neighbors(metformin_idx)
            # Metformin has 8 outgoing edges (including qualifier test edges)
            assert len(neighbors) == 8

    def test_mmap_loaded_graph_edge_properties(self, graph):
        """Should correctly load edge properties in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Test edge property lookup
            src_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            dst_idx = loaded_graph.node_id_to_idx["MONDO:0005148"]
            predicate = loaded_graph.get_edge_property(
                src_idx, dst_idx, "biolink:treats", "predicate"
            )
            assert predicate == "biolink:treats"

    def test_mmap_loaded_graph_node_properties(self, graph):
        """Should correctly load node properties in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            metformin_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            name = loaded_graph.get_node_property(metformin_idx, "name")
            assert name == "Metformin"

    def test_mmap_loaded_graph_qualifiers(self, graph):
        """Should correctly load edge qualifiers in mmap format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir)

            # Check an edge with qualifiers
            metformin_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            insr_idx = loaded_graph.node_id_to_idx["NCBIGene:3643"]

            qualifiers = loaded_graph.get_edge_property(
                metformin_idx, insr_idx, "biolink:affects", "qualifiers"
            )
            assert qualifiers is not None
            assert len(qualifiers) == 2

    def test_mmap_with_no_mmap_mode(self, graph):
        """Should load fully into memory when mmap_mode=None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph.save_mmap(temp_dir)
            loaded_graph = CSRGraph.load_mmap(temp_dir, mmap_mode=None)

            # Verify it still works
            assert loaded_graph.num_nodes == graph.num_nodes
            metformin_idx = loaded_graph.node_id_to_idx["CHEBI:6801"]
            neighbors = loaded_graph.neighbors(metformin_idx)
            assert len(neighbors) == 8


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_predicate_filter_returns_empty(self, graph):
        """Should return empty array for nonexistent predicate."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        neighbors = graph.neighbors(metformin_idx, predicate_filter="biolink:nonexistent")
        assert len(neighbors) == 0

    def test_node_with_no_outgoing_edges(self, graph):
        """Should handle nodes with no outgoing edges."""
        # HP:0001943 (Hypoglycemia) has no outgoing edges in our test data
        hypoglycemia_idx = graph.node_id_to_idx["HP:0001943"]
        neighbors = graph.neighbors(hypoglycemia_idx)
        assert len(neighbors) == 0

    def test_get_node_idx_returns_none_for_unknown(self, graph):
        """Should return None for unknown node ID."""
        assert graph.get_node_idx("UNKNOWN:12345") is None

    def test_get_node_property_default_value(self, graph):
        """Should return default for missing property."""
        metformin_idx = graph.node_id_to_idx["CHEBI:6801"]
        value = graph.get_node_property(metformin_idx, "nonexistent_key", default="default")
        assert value == "default"
