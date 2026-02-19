"""Unit tests for gandalf.search module, specifically the lookup function."""

import os

import pytest

from gandalf.loader import build_graph_from_jsonl
from gandalf.search import lookup, _edge_matches_qualifier_constraints, QualifierExpander


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
NODES_FILE = os.path.join(FIXTURES_DIR, "nodes.jsonl")
EDGES_FILE = os.path.join(FIXTURES_DIR, "edges.jsonl")


@pytest.fixture
def graph():
    """Build a graph from test fixtures."""
    return build_graph_from_jsonl(EDGES_FILE, NODES_FILE)


class TestLookupOneHop:
    """Tests for single-hop queries using the lookup function."""

    def test_one_hop_pinned_both_ends_single_result(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Results are aggregated by unique node paths
        # Same node pair (Metformin -> T2D) = 1 result with 4 edge bindings
        assert len(results) == 1
        result = results[0]
        assert "n0" in result["node_bindings"]
        assert "n1" in result["node_bindings"]
        assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert result["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"

        # Edge bindings should contain all 4 matching edges
        edge_bindings = result["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4

    def test_one_hop_pinned_start_unpinned_end(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # CHEBI:6801 (Metformin) affects 4 genes:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK), NCBIGene:7124 (TNF)
        assert len(results) == 4
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645", "NCBIGene:7124"}

    def test_one_hop_no_matching_predicate(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_one_hop_multiple_results_same_predicate(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Three genes associated with MONDO:0005148:
        # NCBIGene:5468 (PPARG), NCBIGene:3643 (INSR), NCBIGene:2645 (GCK)
        assert len(results) == 3

        # Collect all gene IDs from results
        gene_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}


class TestLookupTwoHop:
    """Tests for two-hop queries using the lookup function."""

    def test_two_hop_linear_path(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_two_hop_multiple_intermediate_nodes(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # NCBIGene:2645 (GCK) participates_in GO:0006006 AND
        # NCBIGene:2645 (GCK) gene_associated_with_condition MONDO:0005148
        # This is a query where n0 appears in both edges
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:2645"


class TestLookupEdgeCases:
    """Tests for edge cases in the lookup function."""

    def test_nonexistent_start_node(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_nonexistent_end_node(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 0

    def test_category_filter_excludes_non_matching(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # CHEBI:6801 affects NCBIGene:5468 (Gene) and CHEBI:17234 (SmallMolecule)
        # Neither is a Pathway, so should return 0
        assert len(results) == 0


class TestLookupResponseStructure:
    """Tests for verifying the response structure from lookup."""

    def test_response_contains_knowledge_graph(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)

        assert "message" in response
        assert "knowledge_graph" in response["message"]
        assert "nodes" in response["message"]["knowledge_graph"]
        assert "edges" in response["message"]["knowledge_graph"]

    def test_response_nodes_have_required_fields(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_nodes = response["message"]["knowledge_graph"]["nodes"]

        # Check Metformin node
        assert "CHEBI:6801" in kg_nodes
        metformin = kg_nodes["CHEBI:6801"]
        assert metformin["id"] == "CHEBI:6801"
        assert metformin["name"] == "Metformin"
        assert "biolink:SmallMolecule" in metformin["categories"]

    def test_response_edges_have_required_fields(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Should have 4 edges (treats includes descendants: ameliorates_condition, preventative_for_condition)
        assert len(kg_edges) == 4

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
        assert len(edge_bindings) == 4

    def test_results_have_node_and_edge_bindings(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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
        assert len(edge_bindings) == 4  # treats, ameliorates_condition, preventative_for_condition


class TestMetforminType2DiabetesEdges:
    """Tests specifically for Metformin to Type 2 Diabetes edges."""

    def test_metformin_treats_type2_diabetes(self, graph, bmt):
        """Query for biolink:treats returns 1 result with 4 edge bindings.

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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Results aggregated by node path: 1 result with 4 edge bindings
        assert len(results) == 1
        kg_edges = response["message"]["knowledge_graph"]["edges"]
        predicates = {edge["predicate"] for edge in kg_edges.values()}
        assert "biolink:treats" in predicates

        # Verify edge bindings contain all 4 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4

    def test_metformin_ameliorates_type2_diabetes(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:ameliorates_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_metformin_prevents_type2_diabetes(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 1
        edge = list(response["message"]["knowledge_graph"]["edges"].values())[0]
        assert edge["predicate"] == "biolink:preventative_for_condition"
        assert edge["subject"] == "CHEBI:6801"
        assert edge["object"] == "MONDO:0005148"

    def test_all_metformin_to_type2_diabetes_edges(self, graph, bmt):
        """Query for all edges from Metformin to Type 2 Diabetes returns 1 result with 4 edge bindings."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

        # Verify edge bindings contain all 4 edges
        edge_bindings = results[0]["analyses"][0]["edge_bindings"]["e0"]
        assert len(edge_bindings) == 4


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

    def test_expanded_format_single_value_match(self):
        """Expanded format with qualifier_values (plural) should match if edge has any value."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
        ]
        # Expanded format: qualifier_values (plural) with list of acceptable values
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # Edge has "activity"
                    }
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_expanded_format_no_match(self):
        """Expanded format should not match if edge value is not in the list."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "expression"}
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # "expression" not in list
                    }
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False

    def test_expanded_format_multiple_types_all_match(self):
        """Expanded format with multiple qualifier types - all must match (AND semantics)."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],
                    },
                    {
                        "qualifier_type_id": "biolink:object_direction_qualifier",
                        "qualifier_values": ["increased", "decreased"],
                    },
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is True

    def test_expanded_format_multiple_types_partial_match(self):
        """Expanded format with multiple qualifier types - partial match should fail."""
        edge_qualifiers = [
            {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"},
            {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "unchanged"},
        ]
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_values": ["activity", "abundance"],  # Matches
                    },
                    {
                        "qualifier_type_id": "biolink:object_direction_qualifier",
                        "qualifier_values": ["increased", "decreased"],  # "unchanged" not in list
                    },
                ]
            }
        ]
        assert _edge_matches_qualifier_constraints(edge_qualifiers, constraints) is False


class TestQualifierExpander:
    """Tests for the QualifierExpander class which handles qualifier value hierarchy."""

    def test_get_value_descendants_unknown_value(self, bmt):
        """Unknown values should return just the original value."""
        expander = QualifierExpander(bmt)
        descendants = expander.get_value_descendants("unknown_value_xyz")
        assert "unknown_value_xyz" in descendants
        # May only have the original value if not in any enum
        assert len(descendants) >= 1

    def test_get_value_descendants_activity(self, bmt):
        """Activity value should include itself (may have no children)."""
        expander = QualifierExpander(bmt)
        descendants = expander.get_value_descendants("activity")
        assert "activity" in descendants

    def test_expand_qualifier_constraints_empty(self, bmt):
        """Empty constraints should return empty."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints([])
        assert result == []

    def test_expand_qualifier_constraints_none(self, bmt):
        """None constraints should return None."""
        expander = QualifierExpander(bmt)
        result = expander.expand_qualifier_constraints(None)
        assert result is None

    def test_expand_qualifier_constraints_empty_qualifier_set(self, bmt):
        """Empty qualifier_set should be preserved."""
        expander = QualifierExpander(bmt)
        constraints = [{"qualifier_set": []}]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 1
        assert result[0]["qualifier_set"] == []

    def test_expand_qualifier_constraints_creates_qualifier_values(self, bmt):
        """Expansion should create qualifier_values (plural) format."""
        expander = QualifierExpander(bmt)
        constraints = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": "biolink:object_aspect_qualifier",
                        "qualifier_value": "activity",
                    }
                ]
            }
        ]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 1
        assert len(result[0]["qualifier_set"]) == 1
        expanded_qualifier = result[0]["qualifier_set"][0]
        assert expanded_qualifier["qualifier_type_id"] == "biolink:object_aspect_qualifier"
        assert "qualifier_values" in expanded_qualifier
        assert "activity" in expanded_qualifier["qualifier_values"]

    def test_expand_qualifier_constraints_preserves_or_semantics(self, bmt):
        """Multiple qualifier_sets should be preserved (OR semantics)."""
        expander = QualifierExpander(bmt)
        constraints = [
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "activity"}
                ]
            },
            {
                "qualifier_set": [
                    {"qualifier_type_id": "biolink:object_aspect_qualifier", "qualifier_value": "abundance"}
                ]
            },
        ]
        result = expander.expand_qualifier_constraints(constraints)
        assert len(result) == 2

    def test_caching_works(self, bmt):
        """Repeated calls should use cache."""
        expander = QualifierExpander(bmt)
        # First call
        descendants1 = expander.get_value_descendants("activity")
        # Second call should use cache
        descendants2 = expander.get_value_descendants("activity")
        assert descendants1 == descendants2
        # Check cache was populated
        assert ("_all_", "activity") in expander._descendants_cache


class TestLookupWithQualifierConstraints:
    """Tests for lookup function with qualifier constraints."""

    def test_qualifier_constraint_filters_edges(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:3643 (INSR) has qualifiers matching activity+increased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_qualifier_constraint_decreased_direction(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:2645 (GCK) has qualifiers matching activity+decreased
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:2645"

    def test_qualifier_constraint_abundance_aspect(self, graph, bmt):
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
                                        {"qualifier_type_id": "biolink:object_direction_qualifier", "qualifier_value": "increased"},
                                    ]
                                }
                            ],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Only NCBIGene:7124 (TNF) has abundance increased qualifiers
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:7124"

    def test_qualifier_constraint_or_semantics(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should match both INSR (activity+increased) and TNF (abundance)
        assert len(results) == 2
        result_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert result_ids == {"NCBIGene:3643", "NCBIGene:7124"}

    def test_no_qualifier_constraints_returns_all(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should return all 4 affects edges to genes:
        # PPARG (no qualifiers), INSR (activity+increased), GCK (activity+decreased), TNF (abundance+increased)
        assert len(results) == 4

    def test_qualifier_constraint_no_matches(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_predicate_forward_direction(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_predicate_reverse_direction(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_predicate_pinned_start_unpinned_end(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as interacting partner
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_unpinned_start_pinned_end(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find INSR as the subject (interacts with PPARG)
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_forward(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find 1 path: Metformin -> PPARG -> INSR
        assert len(results) == 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:5468"
        assert results[0]["node_bindings"]["n2"][0]["id"] == "NCBIGene:3643"

    def test_symmetric_predicate_two_hop_reverse(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_edge_validation_forward(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_edge_validation_reverse(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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

    def test_symmetric_two_hop_validation(self, graph, bmt):
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
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


class TestSubclassHandling:
    """Tests for subclass reasoning feature.

    The test fixtures contain:
    - MONDO:0005148 (Type 2 Diabetes) --subclass_of--> MONDO:0005015 (Diabetes Mellitus)
    - MONDO:0005015 (Diabetes Mellitus) --subclass_of--> MONDO:0004995 (Cardiovascular Disease)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005148 (Type 2 Diabetes)
    - CHEBI:6801 (Metformin) --treats--> MONDO:0005015 (Diabetes Mellitus)
    """

    def test_subclass_off(self, graph, bmt):
        """Without subclass=True, querying for Diabetes Mellitus only returns exact matches."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=False)
        results = response["message"]["results"]

        # Only exact match: Metformin treats Diabetes Mellitus
        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_depth_one_expands_to_children(self, graph, bmt):
        """With subclass=True, querying for Diabetes Mellitus also finds Type 2 Diabetes results.

        Diabetes Mellitus (MONDO:0005015) has child Type 2 Diabetes (MONDO:0005148).
        Metformin treats both, so we should see results for both.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Should find results for both Diabetes Mellitus (direct) and Type 2 Diabetes (subclass)
        assert len(results) == 1

        # Node bindings should reference the originally queried ID (superclass)
        # for results that came via subclass expansion
        bound_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in bound_ids

    def test_subclass_depth_zero_is_identity(self, graph, bmt):
        """With subclass_depth=0, only the exact node matches (equivalent to no subclass)."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response_no_subclass = lookup(graph, query, bmt=bmt, verbose=False, subclass=False)
        response_depth_zero = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=0)

        results_no = response_no_subclass["message"]["results"]
        results_zero = response_depth_zero["message"]["results"]

        # Both should return 1 result: exact match only
        assert len(results_no) == len(results_zero) == 1

    def test_subclass_skips_explicit_hierarchy_edges(self, graph, bmt):
        """Nodes already in explicit subclass_of edges are not rewritten."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},  # Type 2 Diabetes
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:subclass_of"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Should find the explicit edge without creating synthetic superclass nodes
        assert len(results) == 1
        # Node bindings should use the exact queried IDs (no rewriting happened)
        assert results[0]["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005015"

    def test_subclass_auxiliary_graphs_present(self, graph, bmt):
        """Results from subclass expansion should include auxiliary_graphs."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        aux_graphs = response["message"]["auxiliary_graphs"]

        # auxiliary_graphs should exist in the response
        assert isinstance(aux_graphs, dict)

        # If there were subclass expansions that found edges via subclass hops,
        # there should be auxiliary graphs with edge lists
        if aux_graphs:
            for ag_id, ag in aux_graphs.items():
                assert "edges" in ag
                assert len(ag["edges"]) >= 2  # At least the real edge + the subclass edge

    def test_subclass_inferred_edges_have_logical_entailment(self, graph, bmt):
        """Inferred composite edges should have knowledge_level=logical_entailment."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        kg_edges = response["message"]["knowledge_graph"]["edges"]

        # Find inferred edges (those with support_graphs attribute)
        inferred_edges = [
            e for e in kg_edges.values()
            if any(
                attr.get("attribute_type_id") == "biolink:support_graphs"
                for attr in e.get("attributes", [])
            )
        ]

        # There should be at least one inferred edge (from subclass expansion)
        if inferred_edges:
            for edge in inferred_edges:
                attr_map = {a["attribute_type_id"]: a["value"] for a in edge["attributes"]}
                assert attr_map["biolink:knowledge_level"] == "logical_entailment"
                assert attr_map["biolink:agent_type"] == "automated_agent"

    def test_subclass_node_binding_uses_superclass_id(self, graph, bmt):
        """When a result comes via subclass, node binding should reference the queried (superclass) ID."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},  # Diabetes Mellitus (parent)
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # All results should have n0 bound to Metformin
        for result in results:
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"

        # n1 bindings: direct match uses MONDO:0005015, subclass match also uses MONDO:0005015
        # (the superclass ID, since that's what was queried)
        n1_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "MONDO:0005015" in n1_ids

    def test_subclass_superclass_nodes_hidden_from_bindings(self, graph, bmt):
        """Synthetic superclass nodes should not appear in result node_bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        for result in results:
            # Only original query node IDs should be in bindings
            assert set(result["node_bindings"].keys()) == {"n0", "n1"}
            # No "_superclass" keys
            for key in result["node_bindings"]:
                assert "_superclass" not in key

    def test_subclass_subclass_edges_hidden_from_bindings(self, graph, bmt):
        """Synthetic subclass edges should not appear in result edge_bindings."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        for result in results:
            edge_binding_keys = set(result["analyses"][0]["edge_bindings"].keys())
            # Only original query edge IDs should be in bindings
            assert "e0" in edge_binding_keys
            # No "_subclass_edge" keys
            for key in edge_binding_keys:
                assert "_subclass" not in key

    def test_subclass_two_hop_with_expansion(self, graph, bmt):
        """Two-hop query with subclass expansion on one end.

        Query: Metformin --treats--> ? --has_phenotype--> Hypoglycemia
        With subclass on, the disease node should expand to include subclasses.
        Type 2 Diabetes has phenotype Hypoglycemia, and Type 2 Diabetes is
        a subclass of Diabetes Mellitus.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Disease"]},
                        "n2": {"ids": ["HP:0001943"]},  # Hypoglycemia
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        # Without subclass: n1 must have treats edges AND has_phenotype edges
        response_no = lookup(graph, query, bmt=bmt, verbose=False, subclass=False)
        results_no = response_no["message"]["results"]

        # With subclass: same query but subclass expansion might find more paths
        response_yes = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results_yes = response_yes["message"]["results"]

        # Both should find results through Type 2 Diabetes
        # (Metformin treats T2D, T2D has_phenotype Hypoglycemia)
        assert len(results_no) >= 1
        assert len(results_yes) >= 1

    def test_subclass_response_has_auxiliary_graphs_key(self, graph, bmt):
        """Even without subclass, response should have auxiliary_graphs key."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        assert "auxiliary_graphs" in response["message"]
        assert isinstance(response["message"]["auxiliary_graphs"], dict)


class TestRelatedToPredicateExpansion:
    """Tests for biolink:related_to predicate handling.

    biolink:related_to is the root of the predicate hierarchy and should act
    as a wildcard matching any predicate in both forward AND inverse directions.
    """

    def test_related_to_both_pinned_forward_direction(self, graph, bmt):
        """related_to should match edges stored in the query direction.

        Graph has: CHEBI:6801 --treats--> MONDO:0005148
        Query:     CHEBI:6801 --related_to--> MONDO:0005148
        Should find the treats edge (and other forward edges).
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
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) >= 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "MONDO:0005148"

    def test_related_to_both_pinned_inverse_direction(self, graph, bmt):
        """related_to should match edges stored in the REVERSE direction.

        Graph has: CHEBI:6801 --treats--> MONDO:0005148
        Query:     MONDO:0005148 --related_to--> CHEBI:6801  (reversed)
        Should still find edges via inverse lookup.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Before the fix this returned 0 results because inverse edges
        # were not checked when related_to was used
        assert len(results) >= 1
        assert results[0]["node_bindings"]["n0"][0]["id"] == "MONDO:0005148"
        assert results[0]["node_bindings"]["n1"][0]["id"] == "CHEBI:6801"

    def test_related_to_pinned_start_unpinned_end(self, graph, bmt):
        """related_to with pinned start should find ALL neighbors (both directions).

        Graph has: CHEBI:6801 --affects--> NCBIGene:5468 (outgoing)
                   NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148 (outgoing)
                   NCBIGene:5468 --interacts_with--> NCBIGene:3643 (outgoing)
        Query:     NCBIGene:5468 --related_to--> ?
        Should find neighbors in both outgoing AND incoming directions.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        result_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}

        # Forward (outgoing) edges from PPARG:
        #   PPARG --gene_associated_with_condition--> MONDO:0005148
        #   PPARG --interacts_with--> NCBIGene:3643
        assert "MONDO:0005148" in result_ids, "Should find outgoing edge targets"
        assert "NCBIGene:3643" in result_ids, "Should find outgoing edge targets"

        # Inverse (incoming) edges to PPARG:
        #   CHEBI:6801 --affects--> PPARG (incoming to PPARG)
        assert "CHEBI:6801" in result_ids, (
            "Should find incoming edge sources via inverse direction"
        )

    def test_related_to_unpinned_start_pinned_end(self, graph, bmt):
        """related_to with pinned end should find ALL neighbors (both directions).

        Graph has edges pointing TO MONDO:0005148:
            CHEBI:6801 --treats--> MONDO:0005148
            NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148
        And edges pointing FROM MONDO:0005148:
            MONDO:0005148 --has_phenotype--> HP:0001943
            MONDO:0005148 --subclass_of--> MONDO:0005015
        Query: ? --related_to--> MONDO:0005148
        Should find sources of incoming edges AND targets of outgoing edges
        (via inverse).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {},
                        "n1": {"ids": ["MONDO:0005148"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        result_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}

        # Forward (incoming edges to MONDO:0005148, i.e. subjects of edges
        # pointing at it):
        #   CHEBI:6801 --treats--> MONDO:0005148
        #   NCBIGene:5468 --gene_associated_with_condition--> MONDO:0005148
        assert "CHEBI:6801" in result_ids, "Should find subjects of incoming edges"
        assert "NCBIGene:5468" in result_ids, "Should find subjects of incoming edges"

        # Inverse (outgoing edges from MONDO:0005148, found via inverse lookup):
        #   MONDO:0005148 --has_phenotype--> HP:0001943
        assert "HP:0001943" in result_ids, (
            "Should find outgoing edge targets via inverse direction"
        )

    def test_related_to_no_predicates_same_as_related_to(self, graph, bmt):
        """Query with no predicates should behave the same as related_to."""
        # Query with related_to
        query_related = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        # Query with no predicates at all
        query_none = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:5468"]},  # PPARG
                        "n1": {},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                        },
                    },
                },
            },
        }

        response_related = lookup(graph, query_related, bmt=bmt, verbose=False)
        response_none = lookup(graph, query_none, bmt=bmt, verbose=False)

        ids_related = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_related["message"]["results"]
        }
        ids_none = {
            r["node_bindings"]["n1"][0]["id"]
            for r in response_none["message"]["results"]
        }

        assert ids_related == ids_none

    def test_related_to_two_hop_with_inverse(self, graph, bmt):
        """Two-hop query where related_to must use inverse to find the path.

        Query: MONDO:0005148 --related_to--> ? --related_to--> NCBIGene:3643
        One valid path (requiring inverse on first hop):
            MONDO:0005148 <--gene_associated_with_condition-- NCBIGene:5468
            NCBIGene:5468 --interacts_with--> NCBIGene:3643
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        # Should find at least PPARG as the intermediate node
        intermediate_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert "NCBIGene:5468" in intermediate_ids, (
            "Should find PPARG via inverse lookup on first hop"
        )

    def test_related_to_two_hop_symmetric_result_count(self, graph, bmt):
        """Two-hop related_to queries should return the same results regardless of direction.

        Query A: CHEBI:6801 --related_to--> Gene --related_to--> NCBIGene:3643
        Query B: NCBIGene:3643 --related_to--> Gene --related_to--> CHEBI:6801
        Should find the same intermediate Gene nodes.
        """
        query_a = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["NCBIGene:3643"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        query_b = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["NCBIGene:3643"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response_a = lookup(graph, query_a, bmt=bmt, verbose=False)
        response_b = lookup(graph, query_b, bmt=bmt, verbose=False)

        intermediates_a = {r["node_bindings"]["n1"][0]["id"]
                          for r in response_a["message"]["results"]}
        intermediates_b = {r["node_bindings"]["n1"][0]["id"]
                          for r in response_b["message"]["results"]}

        assert intermediates_a == intermediates_b, (
            f"Forward and reverse two-hop queries should find the same intermediate "
            f"nodes, but got {intermediates_a} vs {intermediates_b}"
        )
        assert len(response_a["message"]["results"]) == len(response_b["message"]["results"]), (
            f"Forward and reverse two-hop queries should return the same number of "
            f"results, but got {len(response_a['message']['results'])} vs "
            f"{len(response_b['message']['results'])}"
        )

    def test_related_to_two_hop_inverse_intermediate_pinning(self, graph, bmt):
        """Verify intermediate nodes are correctly pinned when first hop uses inverse.

        Query: MONDO:0005148 --related_to--> Gene --related_to--> CHEBI:6801
        First hop must use inverse to find Gene nodes (edges stored as Gene -> MONDO).
        Those Gene nodes must then correctly pin the second hop.
        The reversed query should produce the same results.
        """
        query_forward = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["MONDO:0005148"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"ids": ["CHEBI:6801"]},
                    },
                    "edges": {
                        "e0": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        query_reversed = {
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
                            "predicates": ["biolink:related_to"],
                        },
                        "e1": {
                            "subject": "n1",
                            "object": "n2",
                            "predicates": ["biolink:related_to"],
                        },
                    },
                },
            },
        }

        response_fwd = lookup(graph, query_forward, bmt=bmt, verbose=False)
        response_rev = lookup(graph, query_reversed, bmt=bmt, verbose=False)

        intermediates_fwd = {r["node_bindings"]["n1"][0]["id"]
                            for r in response_fwd["message"]["results"]}
        intermediates_rev = {r["node_bindings"]["n1"][0]["id"]
                            for r in response_rev["message"]["results"]}

        # Both directions should find genes that connect MONDO:0005148 and CHEBI:6801
        assert len(intermediates_fwd) > 0, "Should find at least one intermediate gene"
        assert intermediates_fwd == intermediates_rev, (
            f"Intermediate genes should be the same regardless of query direction: "
            f"{intermediates_fwd} vs {intermediates_rev}"
        )


class TestNodeBindingGrouping:
    """Tests that results are correctly grouped by node, with exactly one node per binding."""

    def test_three_hop_pinned_endpoints_one_node_per_binding(self, graph, bmt):
        """Three-hop query with pinned endpoints should have one node per binding per result.

        Query: Metformin --affects--> Gene --gene_associated--> Disease --has_phenotype--> Hypoglycemia
        n0 (pinned), n1 (unpinned Gene), n2 (unpinned Disease), n3 (pinned)

        Should produce 3 results (one per gene), each with exactly one node per binding.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"categories": ["biolink:Disease"]},
                        "n3": {"ids": ["HP:0001943"]},
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
                        "e2": {
                            "subject": "n2",
                            "object": "n3",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=False)
        results = response["message"]["results"]

        # Three genes connect Metformin to Type 2 Diabetes: PPARG, INSR, GCK
        assert len(results) == 3

        for result in results:
            # Every node binding must have exactly one entry
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

            # Pinned endpoints should match the query
            assert result["node_bindings"]["n0"][0]["id"] == "CHEBI:6801"
            assert result["node_bindings"]["n3"][0]["id"] == "HP:0001943"

            # Intermediate disease should be T2D (only disease with has_phenotype to Hypoglycemia)
            assert result["node_bindings"]["n2"][0]["id"] == "MONDO:0005148"

        # Each result should have a distinct gene for n1
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643", "NCBIGene:2645"}

    def test_three_hop_pinned_endpoints_with_subclass(self, graph, bmt):
        """Three-hop with subclass expansion still produces one node per binding.

        Same query but with subclass=True. Subclass expansion on pinned nodes
        should not produce duplicate results or multi-entry bindings.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"categories": ["biolink:Gene"]},
                        "n2": {"categories": ["biolink:Disease"]},
                        "n3": {"ids": ["HP:0001943"]},
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
                        "e2": {
                            "subject": "n2",
                            "object": "n3",
                            "predicates": ["biolink:has_phenotype"],
                        },
                    },
                },
            },
        }

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        assert len(results) >= 3

        for result in results:
            # Every node binding must have exactly one entry
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

    def test_two_hop_each_result_has_single_node_per_binding(self, graph, bmt):
        """Two-hop query: each result must have exactly one node per binding."""
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

        response = lookup(graph, query, bmt=bmt, verbose=False)
        results = response["message"]["results"]

        assert len(results) == 3

        for result in results:
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )

    def test_subclass_grouping_no_duplicate_results(self, graph, bmt):
        """Subclass expansion should not produce duplicate results with identical node_bindings.

        When querying with a superclass ID (MONDO:0005015 = Diabetes Mellitus),
        subclass expansion matches both the superclass and its subclass
        (MONDO:0005148 = Type 2 Diabetes).  Both paths should be grouped into
        a single result because the bound ID is the queried superclass ID.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {"ids": ["MONDO:0005015"]},
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

        response = lookup(graph, query, bmt=bmt, verbose=False, subclass=True, subclass_depth=1)
        results = response["message"]["results"]

        # Collect the node binding fingerprints (n0_id, n1_id) for each result
        fingerprints = []
        for result in results:
            fp = tuple(
                (qn, result["node_bindings"][qn][0]["id"])
                for qn in sorted(result["node_bindings"])
            )
            fingerprints.append(fp)

        # No two results should have the same fingerprint (no duplicates)
        assert len(fingerprints) == len(set(fingerprints)), (
            f"Duplicate results found: {fingerprints}"
        )

        # Every result should have exactly one node per binding
        for result in results:
            for qnode_id, bindings in result["node_bindings"].items():
                assert len(bindings) == 1, (
                    f"node_bindings[{qnode_id!r}] has {len(bindings)} entries, expected 1"
                )
