"""Unit tests for gandalf.search module, specifically the lookup function."""

import os

import pytest

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


class TestLookupOneHop:
    """Tests for single-hop queries using the lookup function."""

    def test_one_hop_pinned_both_ends_single_result(self, graph):
        """Query with pinned start and end should return 1 result with multiple edge bindings.

        Note: biolink:treats has descendants ameliorates_condition and
        preventative_for_condition, so all 3 Metformin-T2D edges match.
        Results are aggregated by unique node paths, so we get 1 result.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Results are aggregated by unique node paths
        # Same node pair (Metformin -> T2D) = 1 result with 3 edge bindings
        assert len(results) == 1
        result = results[0]
        assert "n0" in result["node_bindings"]
        assert "n1" in result["node_bindings"]
        assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert result["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"

        # Edge bindings should contain all 3 matching edges
        edge_bindings = result["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 3

    def test_one_hop_pinned_start_unpinned_end(self, graph):
        """Query with pinned start should return all matching edges."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # CHEBI:6801 (Metformin) affects NCBIGene:5468 (PPARG)
        # Should return 1 path to a Gene
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

    def test_one_hop_no_matching_predicate(self, graph):
        """Query with non-matching predicate should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:causes"],  # Not in our data
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_one_hop_multiple_results_same_predicate(self, graph):
        """Query should return all edges matching the predicate."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Three genes associated with MONDO:0005148:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK)
        assert len(results) == 3

        # Collect all gene IDs from results
        gene_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}


class TestLookupTwoHop:
    """Tests for two-hop queries using the lookup function."""

    def test_two_hop_linear_path(self, graph):
        """Two-hop query should return paths through intermediate nodes."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Path: CHEBI:6801 --affects--> NCBIGene:5468 --gene_associated--> MONDO:0005148
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "MONDO:0005148"

    def test_two_hop_multiple_intermediate_nodes(self, graph):
        """Two-hop query with multiple valid intermediate nodes."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["GO:0006006"]},
                        "n2": {"categories": ["biolink:Disease"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:participates_in"],
                        },
                        "e1": {
                            "subject": "n0",
                            "object": "n2",
                            "predicates": ["biolink:gene_associated_with_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # NCBIGene:2645 (GCK) participates_in GO:0006006 AND
        # NCBIGene:2645 (GCK) gene_associated_with_condition MONDO:0005148
        # This is a query where n0 appears in both edges
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:2645"


class TestLookupEdgeCases:
    """Tests for edge cases in the lookup function."""

    def test_nonexistent_start_node(self, graph):
        """Query with non-existent start node should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NONEXISTENT:12345"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_nonexistent_end_node(self, graph):
        """Query with non-existent end node should return 0 paths."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["NONEXISTENT:99999"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_category_filter_excludes_non_matching(self, graph):
        """Category filter should exclude nodes not matching the category."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Pathway"]},  # Only pathways
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # CHEBI:6801 affects NCBIGene:5468 (Gene) and CHEBI:17234 (SmallMolecule)
        # Neither is a Pathway, so should return 0
        assert len(results) == 0


class TestLookupResponseStructure:
    """Tests for verifying the response structure from lookup."""

    def test_response_contains_knowledge_graph(self, graph):
        """Response should contain knowledge_graph with nodes and edges."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)

        assert "message" in response
        assert "knowledge_graph" in response["message"]
        assert "nodes" in response["message"]["knowledge_graph"]
        assert "edges" in response["message"]["knowledge_graph"]

    def test_response_nodes_have_required_fields(self, graph):
        """Knowledge graph nodes should have id, category, and name."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]

        # Check Metformin node
        assert "CHEBI:6801" in kg_nodes
        metformin = kg_nodes["CHEBI:6801"]
        assert metformin["id"] == "CHEBI:6801"
        assert metformin["name"] == "Metformin"
        assert "biolink:SmallMolecule" in metformin["category"]

    def test_response_edges_have_required_fields(self, graph):
        """Knowledge graph edges should have predicate, subject, object."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Should have 3 edges (treats includes descendants: ameliorates_condition, preventative_for_condition)
        assert len(kg_edges) == 3

        # All edges should have required fields and correct subject/object
        for edge in kg_edges.values():
            assert "predicate" in edge
            assert edge["subject"] == "CHEBI:6801"
            assert edge["object"] == "MONDO:0005148"

        # Verify all 3 predicates are present
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert predicates == {
            "biolink:treats",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        }

        # Results should be aggregated: 1 result with all 3 edges in bindings
        results = response["message"]["results"]
        assert len(results) == 1
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 3

    def test_results_have_node_and_edge_bindings(self, graph):
        """Each result should have node_bindings and edge_bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        result = response["message"]["results"][0]

        assert "node_bindings" in result
        assert "analyses" in result
        assert "edge_bindings" in result["analyses"][0]

        # Check node bindings map to query graph nodes
        assert "n0" in result["node_bindings"]
        assert "n1" in result["node_bindings"]

        # Check edge bindings map to query graph edges
        assert "e0" in result["analyses"][0]["edge_bindings"]

        # Edge bindings should be a list containing multiple edges
        edge_bindings = result["analyses"][0]["edge_bindings"]["e0"]
        assert isinstance(edge_bindings, list)
        assert len(edge_bindings) == 3  # treats, ameliorates_condition, preventative_for_condition


class TestMetforminType2DiabetesEdges:
    """Tests specifically for Metformin to Type 2 Diabetes edges."""

    def test_metformin_treats_type2_diabetes(self, graph):
        """Query for biolink:treats returns 1 result with 3 edge bindings.

        Note: biolink:treats is a parent predicate that includes descendants
        ameliorates_condition and preventative_for_condition.
        Results are aggregated by unique node paths.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Results aggregated by node path: 1 result with 3 edge bindings
        assert len(results) == 1
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert "biolink:treats" in predicates

        # Verify edge bindings contain all 3 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 3

    def test_metformin_ameliorates_type2_diabetes(self, graph):
        """Query for Metformin ameliorates_condition Type 2 Diabetes edge."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:ameliorates_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:ameliorates_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_metformin_prevents_type2_diabetes(self, graph):
        """Query for Metformin preventative_for_condition Type 2 Diabetes edge."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:preventative_for_condition"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:preventative_for_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_all_metformin_to_type2_diabetes_edges(self, graph):
        """Query for all edges from Metformin to Type 2 Diabetes returns 1 result with 3 edge bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": [
                                "biolink:treats",
                                "biolink:ameliorates_condition",
                                "biolink:preventative_for_condition",
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Results aggregated by node path: 1 result with 3 edge bindings
        assert len(results) == 1

        # Collect all predicates from knowledge graph
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert predicates == {
            "biolink:treats",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        }

        # Verify edge bindings contain all 3 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 3
