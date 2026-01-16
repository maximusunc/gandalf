"""General graph search functions."""
import time
from collections import defaultdict

from gandalf.graph import CSRGraph


def _return_with_properties(
    graph: CSRGraph, paths: list[list[int]], verbose=True
) -> list[dict[str, dict[str, str]]]:
    """Given paths, return them with useful properties attached."""
    start_time = time.time()
    if verbose:
        print("Assembling enriched paths...")
    hydrated_paths = []
    for path_idx in paths:
        [start_idx, n1_idx, n2_idx, end_idx] = path_idx
        n0 = {
            "id": graph.get_node_id(start_idx),
            "category": graph.get_node_property(start_idx, "category", []),
            "name": graph.get_node_property(start_idx, "name"),
        }
        e0 = {
            "predicate": graph.get_edge_property(start_idx, n1_idx, "predicate"),
        }
        n1 = {
            "id": graph.get_node_id(n1_idx),
            "category": graph.get_node_property(n1_idx, "category", []),
            "name": graph.get_node_property(n1_idx, "name"),
        }
        e1 = {
            "predicate": graph.get_edge_property(n1_idx, n2_idx, "predicate"),
        }
        n2 = {
            "id": graph.get_node_id(n2_idx),
            "category": graph.get_node_property(n2_idx, "category", []),
            "name": graph.get_node_property(n2_idx, "name"),
        }
        e2 = {
            "predicate": graph.get_edge_property(n2_idx, end_idx, "predicate"),
        }
        n3 = {
            "id": graph.get_node_id(end_idx),
            "category": graph.get_node_property(end_idx, "category", []),
            "name": graph.get_node_property(end_idx, "name"),
        }
        hydrated_paths.append({
            "n0": n0,
            "e0": e0,
            "n1": n1,
            "e1": e1,
            "n2": n2,
            "e2": e2,
            "n3": n3,
        })
    if verbose:
        print(
            f"Done! Hydrating {len(hydrated_paths):,} paths took {time.time() - start_time}"
        )
    return hydrated_paths


def find_3hop_paths_with_properties(
    graph: CSRGraph, start_id, end_id, verbose=True, max_paths=None
):
    """
    Find all 3-hop paths between two nodes with edge and node properties.
    Returns list of dicts with path information.

    Args:
        max_paths: If specified, only enrich the first N paths (for performance)
    """
    # Convert IDs to indices
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    if start_idx is None or end_idx is None:
        return []

    if start_idx == end_idx:
        return []

    if verbose:
        print(f"Start node '{start_id}' has degree: {graph.degree(start_idx):,}")
        print(f"End node '{end_id}' has degree: {graph.degree(end_idx):,}")

    # Get raw paths (as indices)
    paths_idx = _find_3hop_paths_directed_idx(
        graph,
        start_idx,
        end_idx,
        start_from_end=(graph.degree(start_idx) > graph.degree(end_idx)),
    )

    if verbose:
        print(f"Found {len(paths_idx):,} paths")

    # Limit paths if requested
    if max_paths and len(paths_idx) > max_paths:
        if verbose:
            print(f"Limiting to first {max_paths:,} paths for property enrichment")
        paths_idx = paths_idx[:max_paths]

    return _return_with_properties(graph, paths_idx, verbose)


def _find_3hop_paths_directed_idx(graph, start_idx, end_idx, start_from_end=False):
    """Helper: find paths using indices, returns paths as index lists"""
    if start_from_end:
        # Search from end to start, then reverse
        paths = _do_search(graph, end_idx, start_idx)
        return [[p[3], p[2], p[1], p[0]] for p in paths]
    else:
        return _do_search(graph, start_idx, end_idx)


def _do_search(graph: CSRGraph, start_idx, end_idx):
    """Actual bidirectional search implementation"""
    # Forward: start -> n1 -> n2
    forward_1 = graph.neighbors(start_idx)

    if len(forward_1) == 0:
        return []

    # Build mapping: n2_idx -> list of n1_idx that reach it
    forward_paths = defaultdict(list)
    for n1_idx in forward_1:
        if n1_idx == end_idx:  # Skip direct 1-hop connections
            continue
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx == start_idx:  # Don't go back to start
                continue
            forward_paths[n2_idx].append(n1_idx)

    # Backward: end -> n2 (just 1 hop)
    backward_1 = set(graph.neighbors(end_idx))

    # Find intersection and build paths
    paths = []
    for n2_idx in forward_paths:
        if n2_idx in backward_1:
            for n1_idx in forward_paths[n2_idx]:
                paths.append([start_idx, n1_idx, n2_idx, end_idx])

    return paths


def find_3hop_paths_filtered(
    graph: CSRGraph,
    start_id,
    end_id,
    allowed_predicates=None,
    excluded_predicates=None,
    verbose=True,
):
    """
    Find 3-hop paths with predicate filtering.

    Args:
        allowed_predicates: If provided, only use edges with these predicates
        excluded_predicates: If provided, skip edges with these predicates
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    if start_idx is None or end_idx is None:
        return []

    if start_idx == end_idx:
        return []

    # Helper to check if predicate is allowed
    def is_edge_allowed(src_idx, dst_idx):
        pred = graph.get_edge_property(src_idx, dst_idx, "predicate")

        if excluded_predicates and pred in excluded_predicates:
            return False

        if allowed_predicates and pred not in allowed_predicates:
            return False

        return True

    if verbose:
        print(f"Start node '{start_id}' has degree: {graph.degree(start_idx):,}")
        print(f"End node '{end_id}' has degree: {graph.degree(end_idx):,}")

    # Forward: start -> n1 -> n2 (with filtering)
    forward_1_filtered = []
    for n1_idx in graph.neighbors(start_idx):
        if n1_idx == end_idx:
            continue
        if is_edge_allowed(start_idx, n1_idx):
            forward_1_filtered.append(n1_idx)

    if verbose:
        print(f"After filtering edge 1: {len(forward_1_filtered):,} neighbors")

    if len(forward_1_filtered) == 0:
        return []

    # Build mapping: n2_idx -> list of n1_idx (with filtering)
    forward_paths = defaultdict(list)
    for n1_idx in forward_1_filtered:
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx == start_idx:
                continue
            if is_edge_allowed(n1_idx, n2_idx):
                forward_paths[n2_idx].append(n1_idx)

    if verbose:
        print(f"Nodes reachable in 2 hops: {len(forward_paths):,}")

    # Backward: end -> n2 (with filtering)
    backward_1_filtered = set()
    for n2_idx in graph.neighbors(end_idx):
        if is_edge_allowed(n2_idx, end_idx):
            backward_1_filtered.add(n2_idx)

    if verbose:
        print(f"After filtering edge 3: {len(backward_1_filtered):,} neighbors of end")

    # Find intersection and build paths
    paths = []
    for n2_idx in forward_paths:
        if n2_idx in backward_1_filtered:
            for n1_idx in forward_paths[n2_idx]:
                n0 = {
                    "id": graph.get_node_id(start_idx),
                    "category": graph.get_node_property(start_idx, "category", []),
                    "name": graph.get_node_property(start_idx, "name"),
                }
                e0 = {
                    "predicate": graph.get_edge_property(
                        start_idx, n1_idx, "predicate"
                    ),
                }
                n1 = {
                    "id": graph.get_node_id(n1_idx),
                    "category": graph.get_node_property(n1_idx, "category", []),
                    "name": graph.get_node_property(n1_idx, "name"),
                }
                e1 = {
                    "predicate": graph.get_edge_property(n1_idx, n2_idx, "predicate"),
                }
                n2 = {
                    "id": graph.get_node_id(n2_idx),
                    "category": graph.get_node_property(n2_idx, "category", []),
                    "name": graph.get_node_property(n2_idx, "name"),
                }
                e2 = {
                    "predicate": graph.get_edge_property(n2_idx, end_idx, "predicate"),
                }
                n3 = {
                    "id": graph.get_node_id(end_idx),
                    "category": graph.get_node_property(end_idx, "category", []),
                    "name": graph.get_node_property(end_idx, "name"),
                }
                paths.append({
                    "n0": n0,
                    "e0": e0,
                    "n1": n1,
                    "e1": e1,
                    "n2": n2,
                    "e2": e2,
                    "n3": n3,
                })

    if verbose:
        print(f"Found {len(paths):,} filtered paths")

    return paths


def find_meaningful_paths(graph: CSRGraph, start_id, end_id, verbose=True):
    """
    Convenience function: Find paths excluding ontology noise.

    Excludes common "uninformative" predicates that cause path explosion:
    - subclass_of: ontology hierarchies
    - related_to: too generic
    """
    excluded = {
        "biolink:subclass_of",
        "biolink:related_to",  # Often too generic
    }

    return find_3hop_paths_filtered(
        graph, start_id, end_id, excluded_predicates=excluded, verbose=verbose
    )


def find_mechanistic_paths(graph: CSRGraph, start_id, end_id, verbose=True):
    """
    Find paths using only mechanistic/causal predicates.

    Good for finding actual biological mechanisms rather than associations.
    """
    mechanistic_predicates = {
        "biolink:treats",
        "biolink:affects",
        "biolink:regulates",
        "biolink:increases_expression_of",
        "biolink:decreases_expression_of",
        "biolink:gene_associated_with_condition",
        "biolink:has_metabolite",
        "biolink:metabolized_by",
        "biolink:applied_to_treat",
        "biolink:contraindicated_for",
        "biolink:directly_physically_interacts_with",
        "biolink:has_contraindication",
        "biolink:subject_of_treatment_application_or_study_for_treatment_by",
        "biolink:contribution_from",
    }

    return find_3hop_paths_filtered(
        graph,
        start_id,
        end_id,
        allowed_predicates=mechanistic_predicates,
        verbose=verbose,
    )
