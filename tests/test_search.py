"""Unit tests for gandalf.search module, specifically the lookup function."""

import os

import pytest

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup, _edge_matches_qualifier_constraints


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

        # CHEBI:6801 (Metformin) affects 4 genes:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK), NCBIGene:7124 (TNF)
        assert len(results) == 4
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645", "NCBIGene:7124"}

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

        # Three paths from Metformin through genes to Type 2 Diabetes:
        # CHEBI:6801 --affects--> NCBIGene:5468 (PPARG) --gene_associated--> MONDO:0005148
        # CHEBI:6801 --affects--> NCBIGene:3643 (INSR) --gene_associated--> MONDO:0005148
        # CHEBI:6801 --affects--> NCBIGene:2645 (GCK) --gene_associated--> MONDO:0005148
        assert len(results) == 3
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}
        # All paths should start with Metformin and end with Type 2 Diabetes
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
            assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005148"

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
        assert "biolink:SmallMolecule" in metformin["categories"]

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


class TestQualifierConstraintMatching:
    """Unit tests for the _edge_matches_qualifier_constraints helper function."""

    def test_no_constraints_returns_true(self):
        """No qualifier constraints should match any edge."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, None) is True
        assert _edge_matches_qualifier_constraints(edge_qualifiers, []) is True

    def test_empty_qualifier_set_matches_any_edge(self):
        """Empty qualifier_set should match any edge."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [{"qualifier_set": []}]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_single_qualifier_match(self):
        """Edge with matching single qualifier should match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_single_qualifier_no_match(self):
        """Edge with non-matching qualifier should not match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"}
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_multiple_qualifiers_all_match(self):
        """Edge with all required qualifiers should match (AND semantics within set)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                    {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_multiple_qualifiers_partial_match(self):
        """Edge with only some required qualifiers should not match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                    {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_or_semantics_between_qualifier_sets(self):
        """Edge matching any qualifier_set should match (OR semantics between sets)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            },
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                ]
            },
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_edge_with_no_qualifiers(self):
        """Edge with no qualifiers should not match constraints requiring qualifiers."""
        edge_qualifiers = []
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_edge_with_extra_qualifiers_still_matches(self):
        """Edge with extra qualifiers beyond required should still match."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
            {"qualifier_type_id": "biolink:qualified_predicate", "qualifier_value": "biolink:causes"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True


class TestLookupWithQualifierConstraints:
    """Tests for lookup function with qualifier constraints."""

    def test_qualifier_constraint_filters_edges(self, graph):
        """Qualifier constraints should filter to only matching edges."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:3643 (INSR) has qualifiers matching activity+increased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_qualifier_constraint_decreased_direction(self, graph):
        """Query for edges with decreased direction qualifier."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "decreased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:2645 (GCK) has qualifiers matching activity+decreased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:2645"

    def test_qualifier_constraint_abundance_aspect(self, graph):
        """Query for edges with abundance aspect qualifier."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:7124 (TNF) has abundance qualifier
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"

    def test_qualifier_constraint_or_semantics(self, graph):
        """Multiple qualifier sets should use OR semantics."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                },
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"},
                                    ]
                                },
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should match both INSR (activity+increased) and TNF (abundance)
        assert len(results) == 2
        result_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert result_ids == {"NCBIGene:3643", "NCBIGene:7124"}

    def test_no_qualifier_constraints_returns_all(self, graph):
        """Without qualifier constraints, all edges should match."""
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

        # Should return all 4 affects edges to genes:
        # PPARG (no qualifiers), INSR (activity+increased), GCK (activity+decreased), TNF (abundance+increased)
        assert len(results) == 4

    def test_qualifier_constraint_no_matches(self, graph):
        """Query with non-matching qualifier constraints should return 0 results."""
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
                            "qualifier_constraints": [
                                {
                                    "qualifier_set": [
                                        {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "expression"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # No edges have expression qualifier
        assert len(results) == 0


class TestSymmetricPredicates:
    """Tests for symmetric predicate handling (e.g., interacts_with).

    The test fixture contains:
    - NCBIGene:5468 (PPARG) --interacts_with--> NCBIGene:3643 (INSR)

    Symmetric predicates should be found regardless of query direction:
    - Query PPARG -> INSR should find the direct edge
    - Query INSR -> PPARG should also find this edge (via symmetric property)
    """

    def test_symmetric_predicate_forward_direction(self, graph):
        """Query in the direction the edge is stored should return the edge."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find the direct edge PPARG -> INSR
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

        # The edge in the knowledge graph should be in the stored direction
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        assert len(kg_edges) == 1
        edge = list(kg_edges.values())[0]
        assert edge["predicate"] == "biolink:interacts_with"
        # Edge should be in the actual stored direction
        assert edge["subject"] == "NCBIGene:5468"
        assert edge["object"] == "NCBIGene:3643"

    def test_symmetric_predicate_reverse_direction(self, graph):
        """Query in reverse direction should also find the edge via symmetric property.

        Graph has: PPARG --interacts_with--> INSR
        Query asks: INSR --interacts_with--> PPARG (reverse)
        Should find the edge because interacts_with is symmetric.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},  # INSR (query subject)
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (query object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find 1 result - the symmetric edge
        assert len(results) == 1
        # Node bindings should reflect the query structure
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:3643"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"

        # The edge in knowledge graph should be the ACTUAL stored edge
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        assert len(kg_edges) == 1
        edge = list(kg_edges.values())[0]
        assert edge["predicate"] == "biolink:interacts_with"
        # Edge should be in the actual stored direction (PPARG -> INSR)
        assert edge["subject"] == "NCBIGene:5468"
        assert edge["object"] == "NCBIGene:3643"

    def test_symmetric_predicate_pinned_start_unpinned_end(self, graph):
        """Query with pinned start should find neighbors via symmetric predicate."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"categories": ["biolink:Gene"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as interacting partner
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_unpinned_start_pinned_end(self, graph):
        """Query with pinned end should find neighbors via symmetric predicate.

        Graph has: PPARG --interacts_with--> INSR
        Query: ? --interacts_with--> PPARG
        Should find INSR via symmetric property.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"categories": ["biolink:Gene"]},
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (pinned as object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as the subject (interacts with PPARG)
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_forward(self, graph):
        """Two-hop query with symmetric predicate in forward direction.

        Path: Metformin --affects--> PPARG --interacts_with--> INSR
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},  # PPARG
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
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
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find 1 path: Metformin -> PPARG -> INSR
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_reverse(self, graph):
        """Two-hop query with symmetric predicate in reverse direction.

        Query path: Metformin --affects--> PPARG <--interacts_with-- INSR
        Stored as: PPARG --interacts_with--> INSR
        The symmetric predicate should allow INSR to be found as connecting to PPARG.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},  # Will be PPARG
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n2",  # INSR as subject
                            "object": "n1",  # PPARG as object
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        results = response["message"]["results"]

        # Should find 1 path via symmetric interacts_with
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "NCBIGene:3643"


class TestSymmetricPredicateValidation:
    """Tests that validate returned edges actually exist in the graph.

    These tests ensure that the edges returned in the knowledge_graph
    are real edges that exist in the graph, not "phantom edges".
    """

    def test_symmetric_edge_validation_forward(self, graph):
        """Edges returned from forward symmetric query should exist in graph."""
        from gandalf.validation import validate_edge_exists

        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Every edge in the knowledge graph should exist in the actual graph
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, f"Edge {edge_id} not found: {edge}"

    def test_symmetric_edge_validation_reverse(self, graph):
        """Edges returned from reverse symmetric query should exist in graph.

        This is the key test for the phantom edge bug. When querying in
        the reverse direction of a symmetric predicate, the returned
        edge should still be the actual stored edge.
        """
        from gandalf.validation import validate_edge_exists

        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},  # INSR (query subject)
                        "n1": {"ids": ["NCBIGene:5468"]},  # PPARG (query object)
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Should have found at least one edge
        assert len(kg_edges) >= 1, "Expected at least one edge in response"

        # Every edge in the knowledge graph should exist in the actual graph
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, (
                f"Phantom edge detected! Edge {edge_id} not found in graph: "
                f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
            )

    def test_symmetric_two_hop_validation(self, graph):
        """All edges in multi-hop symmetric query should exist in graph."""
        from gandalf.validation import validate_edge_exists

        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},  # Metformin
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},  # INSR
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:affects"],
                        },
                        "e1": {
                            "subject": "n2",  # Reverse direction
                            "object": "n1",
                            "predicates": ["biolink:interacts_with"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Validate every edge
        for edge_id, edge in kg_edges.items():
            error = validate_edge_exists(
                graph,
                edge["subject"],
                edge["predicate"],
                edge["object"],
            )
            assert error is None, (
                f"Phantom edge detected! Edge {edge_id} not found in graph: "
                f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
            )
