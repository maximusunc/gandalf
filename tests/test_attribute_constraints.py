"""Tests for TRAPI attribute constraint matching and filtering."""

import pytest

from gandalf.search.attribute_constraints import matches_attribute_constraints
from tests.search_fixtures import graph  # noqa: F401

from gandalf.search.lookup import lookup


# ---------------------------------------------------------------------------
# Unit tests for the matching function
# ---------------------------------------------------------------------------

class TestMatchesAttributeConstraints:
    """Unit tests for matches_attribute_constraints."""

    def test_empty_constraints_returns_true(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.05}]
        assert matches_attribute_constraints(attrs, []) is True
        assert matches_attribute_constraints(attrs, None) is True

    def test_no_attributes_with_constraints_returns_false(self):
        constraints = [{
            "id": "biolink:p_value",
            "name": "p-value",
            "operator": "<",
            "value": 0.05,
        }]
        assert matches_attribute_constraints([], constraints) is False
        assert matches_attribute_constraints(None, constraints) is False

    def test_equals_operator(self):
        attrs = [{"attribute_type_id": "biolink:knowledge_level", "value": "knowledge_assertion"}]
        constraint_match = [{
            "id": "biolink:knowledge_level",
            "name": "knowledge level",
            "operator": "==",
            "value": "knowledge_assertion",
        }]
        constraint_no_match = [{
            "id": "biolink:knowledge_level",
            "name": "knowledge level",
            "operator": "==",
            "value": "prediction",
        }]
        assert matches_attribute_constraints(attrs, constraint_match) is True
        assert matches_attribute_constraints(attrs, constraint_no_match) is False

    def test_greater_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": ">", "value": 0.01,
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": ">", "value": 0.05,
        }]) is False

    def test_less_than_operator(self):
        attrs = [{"attribute_type_id": "biolink:p_value", "value": 0.03}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05,
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.01,
        }]) is False

    def test_matches_operator_regex(self):
        attrs = [{"attribute_type_id": "biolink:description", "value": "Metformin treats diabetes"}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:description", "name": "desc", "operator": "matches",
            "value": "treats.*diabetes",
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:description", "name": "desc", "operator": "matches",
            "value": "^prevents",
        }]) is False

    def test_strict_equals_operator(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": 42}]
        # Same type and value
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "score", "operator": "===", "value": 42,
        }]) is True
        # Different type (float vs int)
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "score", "operator": "===", "value": 42.0,
        }]) is False

    def test_strict_equals_list_order(self):
        attrs = [{"attribute_type_id": "biolink:tags", "value": ["a", "b", "c"]}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:tags", "name": "tags", "operator": "===", "value": ["a", "b", "c"],
        }]) is True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:tags", "name": "tags", "operator": "===", "value": ["c", "b", "a"],
        }]) is False

    def test_not_negation(self):
        attrs = [{"attribute_type_id": "biolink:knowledge_level", "value": "prediction"}]
        # "not prediction" should pass for "prediction" -> negated match -> False
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:knowledge_level", "name": "kl", "operator": "==",
            "value": "prediction", "not": True,
        }]) is False
        # "not knowledge_assertion" should pass for "prediction" -> no match -> negated -> True
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:knowledge_level", "name": "kl", "operator": "==",
            "value": "knowledge_assertion", "not": True,
        }]) is True

    def test_and_logic_multiple_constraints(self):
        attrs = [
            {"attribute_type_id": "biolink:p_value", "value": 0.03},
            {"attribute_type_id": "biolink:score", "value": 0.95},
        ]
        # Both pass
        constraints = [
            {"id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05},
            {"id": "biolink:score", "name": "score", "operator": ">", "value": 0.9},
        ]
        assert matches_attribute_constraints(attrs, constraints) is True

        # First passes, second fails
        constraints_fail = [
            {"id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05},
            {"id": "biolink:score", "name": "score", "operator": ">", "value": 0.99},
        ]
        assert matches_attribute_constraints(attrs, constraints_fail) is False

    def test_missing_attribute_fails(self):
        attrs = [{"attribute_type_id": "biolink:score", "value": 0.95}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:p_value", "name": "p", "operator": "<", "value": 0.05,
        }]) is False

    def test_numeric_comparison_with_list_value_or_logic(self):
        """Per TRAPI spec: with lists and > or <, at least one must be true (OR)."""
        attrs = [{"attribute_type_id": "biolink:score", "value": 5}]
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "s", "operator": ">", "value": [10, 3],
        }]) is True  # 5 > 3 is true
        assert matches_attribute_constraints(attrs, [{
            "id": "biolink:score", "name": "s", "operator": ">", "value": [10, 20],
        }]) is False  # neither

    def test_match_by_original_attribute_name(self):
        """Constraints should also match on original_attribute_name."""
        attrs = [{"attribute_type_id": "biolink:Attribute",
                  "original_attribute_name": "information_content", "value": 92.3}]
        assert matches_attribute_constraints(attrs, [{
            "id": "information_content", "name": "IC", "operator": ">", "value": 90,
        }]) is True

    def test_no_attributes_all_negated_returns_true(self):
        """If all constraints are negated and there are no attributes, they all pass."""
        constraints = [{
            "id": "biolink:p_value", "name": "p", "operator": "<",
            "value": 0.05, "not": True,
        }]
        assert matches_attribute_constraints([], constraints) is True


# ---------------------------------------------------------------------------
# Integration tests: node constraints filtering through lookup
# ---------------------------------------------------------------------------

class TestNodeConstraintsIntegration:
    """Test node attribute constraints in full TRAPI queries."""

    def test_node_constraint_filters_by_information_content(self, graph, bmt):
        """Node constraints with '>' on information_content should filter nodes.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC > 90 should keep PPARG and TNF.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "information content",
                                "operator": ">",
                                "value": 90,
                            }],
                        },
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

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:7124"}

    def test_node_constraint_less_than_ic(self, graph, bmt):
        """Node constraint with '<' on information_content.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC < 85 should keep only GCK.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "information content",
                                "operator": "<",
                                "value": 85,
                            }],
                        },
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

        assert len(results) == 1
        assert results[0]["node_bindings"]["n1"][0]["id"] == "NCBIGene:2645"

    def test_node_constraint_not_negation(self, graph, bmt):
        """Negated node constraint should exclude matching nodes.

        Genes associated with T2D: PPARG(IC=92.3), INSR(IC=88.7), GCK(IC=81.2).
        Constraint: NOT IC > 90 -> keeps INSR and GCK.
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 90,
                                "not": True,
                            }],
                        },
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

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n0"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:3643", "NCBIGene:2645"}

    def test_empty_constraints_no_filtering(self, graph, bmt):
        """Empty constraints list should not filter anything."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [],
                        },
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

        # All 4 genes should still be returned
        assert len(results) == 4

    def test_node_constraint_multiple_and_logic(self, graph, bmt):
        """Multiple node constraints use AND logic.

        Metformin affects 4 genes with ICs: PPARG(92.3), INSR(88.7), GCK(81.2), TNF(94.5).
        Constraint IC > 85 AND IC < 93 should keep only PPARG(92.3) and INSR(88.7).
        """
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [
                                {
                                    "id": "information_content",
                                    "name": "IC",
                                    "operator": ">",
                                    "value": 85,
                                },
                                {
                                    "id": "information_content",
                                    "name": "IC",
                                    "operator": "<",
                                    "value": 93,
                                },
                            ],
                        },
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

        assert len(results) == 2
        gene_ids = {r["node_bindings"]["n1"][0]["id"] for r in results}
        assert gene_ids == {"NCBIGene:5468", "NCBIGene:3643"}

    def test_node_constraint_filters_all_returns_empty(self, graph, bmt):
        """Constraint that no node satisfies should return empty results."""
        query = {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {"ids": ["CHEBI:6801"]},
                        "n1": {
                            "categories": ["biolink:Gene"],
                            "constraints": [{
                                "id": "information_content",
                                "name": "IC",
                                "operator": ">",
                                "value": 999,
                            }],
                        },
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
        assert len(results) == 0
