"""Validation utilities for verifying query results against the graph.

This module provides tools for development and testing to ensure that
query results are consistent with the actual graph data.
"""

from dataclasses import dataclass
from typing import Optional

from gandalf.graph import CSRGraph


@dataclass
class ValidationError:
    """Represents a validation error found in a result."""
    error_type: str
    message: str
    path_index: Optional[int] = None
    edge_id: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating query results against the graph."""
    valid: bool
    total_paths: int
    valid_paths: int
    errors: list[ValidationError]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Validation {'PASSED' if self.valid else 'FAILED'}",
            f"  Total paths: {self.total_paths}",
            f"  Valid paths: {self.valid_paths}",
            f"  Invalid paths: {self.total_paths - self.valid_paths}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors[:10]:  # Show first 10 errors
                lines.append(f"    - [{err.error_type}] {err.message}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more errors")
        return "\n".join(lines)


def validate_node_exists(graph: CSRGraph, node_id: str) -> Optional[ValidationError]:
    """Check if a node exists in the graph."""
    idx = graph.get_node_idx(node_id)
    if idx is None:
        return ValidationError(
            error_type="NODE_NOT_FOUND",
            message=f"Node '{node_id}' not found in graph",
            node_id=node_id,
        )
    return None


def validate_edge_exists(
    graph: CSRGraph,
    subject_id: str,
    predicate: str,
    object_id: str,
    check_reverse: bool = True,
) -> Optional[ValidationError]:
    """
    Check if an edge exists in the graph.

    Args:
        graph: The CSRGraph to validate against
        subject_id: Subject node ID
        predicate: Edge predicate
        object_id: Object node ID
        check_reverse: If True, also check for the edge in reverse direction

    Returns:
        ValidationError if edge not found, None otherwise
    """
    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    if subj_idx is None:
        return ValidationError(
            error_type="EDGE_SUBJECT_NOT_FOUND",
            message=f"Edge subject '{subject_id}' not found in graph",
            node_id=subject_id,
        )

    if obj_idx is None:
        return ValidationError(
            error_type="EDGE_OBJECT_NOT_FOUND",
            message=f"Edge object '{object_id}' not found in graph",
            node_id=object_id,
        )

    # Check forward direction: subject -> object
    edges = graph.get_all_edges_between(subj_idx, obj_idx)
    for edge_pred, _ in edges:
        if edge_pred == predicate:
            return None  # Found the edge

    # Check reverse direction: object -> subject (for symmetric/inverse predicates)
    if check_reverse:
        reverse_edges = graph.get_all_edges_between(obj_idx, subj_idx)
        for edge_pred, _ in reverse_edges:
            if edge_pred == predicate:
                return None  # Found the edge in reverse

    return ValidationError(
        error_type="EDGE_NOT_FOUND",
        message=f"Edge '{subject_id}' --[{predicate}]--> '{object_id}' not found in graph",
    )


def validate_trapi_response(
    graph: CSRGraph,
    response: dict,
    check_reverse_edges: bool = True,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate a TRAPI response against the graph.

    Checks that all nodes and edges in the knowledge graph and results
    actually exist in the source graph.

    Args:
        graph: The CSRGraph to validate against
        response: TRAPI response dict with message.knowledge_graph and message.results
        check_reverse_edges: If True, also check for edges in reverse direction
        verbose: Print progress information

    Returns:
        ValidationResult with validation status and any errors found
    """
    errors = []
    message = response.get("message", {})
    kg = message.get("knowledge_graph", {})
    results = message.get("results", [])

    if verbose:
        print(f"Validating response with {len(kg.get('nodes', {}))} KG nodes, "
              f"{len(kg.get('edges', {}))} KG edges, {len(results)} results")

    # Validate knowledge graph nodes
    kg_nodes = kg.get("nodes", {})
    for node_id, node_data in kg_nodes.items():
        err = validate_node_exists(graph, node_id)
        if err:
            errors.append(err)

    if verbose:
        print(f"  Validated {len(kg_nodes)} KG nodes, {len([e for e in errors if e.error_type.startswith('NODE')])} errors")

    # Validate knowledge graph edges
    kg_edges = kg.get("edges", {})
    edge_errors_before = len(errors)
    for edge_id, edge_data in kg_edges.items():
        subject_id = edge_data.get("subject")
        predicate = edge_data.get("predicate")
        object_id = edge_data.get("object")

        if not all([subject_id, predicate, object_id]):
            errors.append(ValidationError(
                error_type="INVALID_EDGE_DATA",
                message=f"Edge '{edge_id}' missing required fields",
                edge_id=edge_id,
            ))
            continue

        err = validate_edge_exists(
            graph, subject_id, predicate, object_id,
            check_reverse=check_reverse_edges
        )
        if err:
            err.edge_id = edge_id
            errors.append(err)

    if verbose:
        print(f"  Validated {len(kg_edges)} KG edges, {len(errors) - edge_errors_before} errors")

    # Count valid paths (results where all bindings are valid)
    valid_paths = 0
    for i, result in enumerate(results):
        path_valid = True

        # Check node bindings
        node_bindings = result.get("node_bindings", {})
        for qnode_id, bindings in node_bindings.items():
            for binding in bindings:
                node_id = binding.get("id")
                if node_id and node_id not in kg_nodes:
                    errors.append(ValidationError(
                        error_type="BINDING_NODE_NOT_IN_KG",
                        message=f"Result {i}: Node binding '{node_id}' not in knowledge graph",
                        path_index=i,
                        node_id=node_id,
                    ))
                    path_valid = False

        # Check edge bindings
        analyses = result.get("analyses", [])
        for analysis in analyses:
            edge_bindings = analysis.get("edge_bindings", {})
            for qedge_id, bindings in edge_bindings.items():
                for binding in bindings:
                    edge_id = binding.get("id")
                    if edge_id and edge_id not in kg_edges:
                        errors.append(ValidationError(
                            error_type="BINDING_EDGE_NOT_IN_KG",
                            message=f"Result {i}: Edge binding '{edge_id}' not in knowledge graph",
                            path_index=i,
                            edge_id=edge_id,
                        ))
                        path_valid = False

        if path_valid:
            valid_paths += 1

    if verbose:
        print(f"  Validated {len(results)} results, {valid_paths} valid")

    return ValidationResult(
        valid=len(errors) == 0,
        total_paths=len(results),
        valid_paths=valid_paths,
        errors=errors,
    )


def validate_edge_list(
    graph: CSRGraph,
    edges: list[tuple[int, str, int]],
    check_reverse: bool = True,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate a list of edges (as returned by _query_edge).

    Args:
        graph: The CSRGraph to validate against
        edges: List of (subject_idx, predicate, object_idx) tuples
        check_reverse: If True, also check for edges in reverse direction
        verbose: Print progress information

    Returns:
        ValidationResult with validation status and any errors found
    """
    errors = []
    valid_count = 0

    for i, (subj_idx, predicate, obj_idx) in enumerate(edges):
        # Get node IDs
        subj_id = graph.get_node_id(subj_idx)
        obj_id = graph.get_node_id(obj_idx)

        if subj_id is None:
            errors.append(ValidationError(
                error_type="INVALID_SUBJECT_IDX",
                message=f"Edge {i}: Subject index {subj_idx} has no node ID",
                path_index=i,
            ))
            continue

        if obj_id is None:
            errors.append(ValidationError(
                error_type="INVALID_OBJECT_IDX",
                message=f"Edge {i}: Object index {obj_idx} has no node ID",
                path_index=i,
            ))
            continue

        # Check if edge exists
        err = validate_edge_exists(
            graph, subj_id, predicate, obj_id,
            check_reverse=check_reverse
        )
        if err:
            err.path_index = i
            errors.append(err)
        else:
            valid_count += 1

    if verbose:
        print(f"Validated {len(edges)} edges: {valid_count} valid, {len(errors)} errors")

    return ValidationResult(
        valid=len(errors) == 0,
        total_paths=len(edges),
        valid_paths=valid_count,
        errors=errors,
    )


def find_edge_in_graph(
    graph: CSRGraph,
    subject_id: str,
    object_id: str,
) -> list[dict]:
    """
    Find all edges between two nodes in both directions.

    Useful for debugging when an expected edge is not found.

    Args:
        graph: The CSRGraph to search
        subject_id: First node ID
        object_id: Second node ID

    Returns:
        List of edge dicts with direction, predicate, and properties
    """
    results = []

    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    if subj_idx is None or obj_idx is None:
        return results

    # Forward direction: subject -> object
    forward_edges = graph.get_all_edges_between(subj_idx, obj_idx)
    for predicate, props in forward_edges:
        results.append({
            "direction": "forward",
            "subject": subject_id,
            "predicate": predicate,
            "object": object_id,
            "properties": props,
        })

    # Reverse direction: object -> subject
    reverse_edges = graph.get_all_edges_between(obj_idx, subj_idx)
    for predicate, props in reverse_edges:
        results.append({
            "direction": "reverse",
            "subject": object_id,
            "predicate": predicate,
            "object": subject_id,
            "properties": props,
        })

    return results


def debug_missing_edge(
    graph: CSRGraph,
    subject_id: str,
    predicate: str,
    object_id: str,
) -> str:
    """
    Generate a debug report for a missing edge.

    Args:
        graph: The CSRGraph to search
        subject_id: Expected subject node ID
        predicate: Expected predicate
        object_id: Expected object node ID

    Returns:
        Human-readable debug report
    """
    lines = [
        f"Debug report for missing edge:",
        f"  Expected: {subject_id} --[{predicate}]--> {object_id}",
        "",
    ]

    # Check if nodes exist
    subj_idx = graph.get_node_idx(subject_id)
    obj_idx = graph.get_node_idx(object_id)

    lines.append("Node existence:")
    lines.append(f"  Subject '{subject_id}': {'EXISTS (idx={subj_idx})' if subj_idx is not None else 'NOT FOUND'}")
    lines.append(f"  Object '{object_id}': {'EXISTS (idx={obj_idx})' if obj_idx is not None else 'NOT FOUND'}")

    if subj_idx is None or obj_idx is None:
        return "\n".join(lines)

    # Find all edges between these nodes
    lines.append("")
    lines.append("Edges found between these nodes:")

    found_edges = find_edge_in_graph(graph, subject_id, object_id)
    if not found_edges:
        lines.append("  (none)")
    else:
        for edge in found_edges:
            lines.append(
                f"  [{edge['direction']}] {edge['subject']} --[{edge['predicate']}]--> {edge['object']}"
            )

    # Check neighbors
    lines.append("")
    lines.append(f"Subject '{subject_id}' neighbors (first 10):")
    neighbors = graph.neighbors(subj_idx)
    for i, neighbor_idx in enumerate(neighbors[:10]):
        neighbor_id = graph.get_node_id(neighbor_idx)
        edges = graph.get_all_edges_between(subj_idx, neighbor_idx)
        preds = [p for p, _ in edges]
        lines.append(f"  -> {neighbor_id} via {preds}")
    if len(neighbors) > 10:
        lines.append(f"  ... and {len(neighbors) - 10} more")

    lines.append("")
    lines.append(f"Object '{object_id}' incoming neighbors (first 10):")
    incoming = graph.incoming_neighbors(obj_idx)
    for i, neighbor_idx in enumerate(incoming[:10]):
        neighbor_id = graph.get_node_id(neighbor_idx)
        edges = graph.get_all_edges_between(neighbor_idx, obj_idx)
        preds = [p for p, _ in edges]
        lines.append(f"  <- {neighbor_id} via {preds}")
    if len(incoming) > 10:
        lines.append(f"  ... and {len(incoming) - 10} more")

    return "\n".join(lines)
