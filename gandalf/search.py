"""General graph search functions."""

import copy
import time
import uuid
from collections import defaultdict

import numpy as np
from bmt.toolkit import Toolkit

from gandalf.graph import CSRGraph
from gandalf.query_planner import get_next_qedge, remove_orphaned


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
        # Get predicates - take the first one if multiple exist
        edges_01 = graph.get_all_edges_between(start_idx, n1_idx)
        edges_12 = graph.get_all_edges_between(n1_idx, n2_idx)
        edges_23 = graph.get_all_edges_between(n2_idx, end_idx)
        print(edges_01)
        exit()

        pred_01 = edges_01[0][0] if edges_01 else None
        pred_12 = edges_12[0][0] if edges_12 else None
        pred_23 = edges_23[0][0] if edges_23 else None
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
        paths = _do_unfiltered_search(graph, end_idx, start_idx)
        return [[p[3], p[2], p[1], p[0]] for p in paths]
    else:
        return _do_unfiltered_search(graph, start_idx, end_idx)


def _do_unfiltered_search(graph: CSRGraph, start_idx, end_idx, verbose=False):
    """Actual bidirectional search implementation returning full 3-hop paths"""

    # ─────────────────────────────────────────────────────────────
    # Forward: start → n1
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    forward_1 = graph.neighbors(start_idx)
    t1 = time.perf_counter()
    if verbose:
        print("Imatinib 1-hop:", len(forward_1), t1 - t0)

    if len(forward_1) == 0:
        return []

    # ─────────────────────────────────────────────────────────────
    # Forward: start → n1 → n2  (vectorized parent tracking)
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    src_bufs = []  # n1 indices
    dst_bufs = []  # n2 indices

    for n1_idx in forward_1:
        if n1_idx == end_idx:  # skip direct edge
            continue

        n2s = graph.neighbors(n1_idx)
        if len(n2s) == 0:
            continue

        src_bufs.append(np.full(len(n2s), n1_idx, dtype=np.int32))
        dst_bufs.append(n2s)

    if not src_bufs:
        return []

    src_n1 = np.concatenate(src_bufs)
    dst_n2 = np.concatenate(dst_bufs)

    t1 = time.perf_counter()
    if verbose:
        print("Imatinib 2-hop edges:", len(dst_n2), t1 - t0)

    # ─────────────────────────────────────────────────────────────
    # Backward: end → n2
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    backward_1 = graph.incoming_neighbors(end_idx)
    t1 = time.perf_counter()
    if verbose:
        print("Asthma 1-hop:", len(backward_1), t1 - t0)

    if len(backward_1) == 0:
        return []

    backward_unique = np.unique(backward_1)

    # ─────────────────────────────────────────────────────────────
    # Intersection: keep only valid n2
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    mask = np.isin(dst_n2, backward_unique, assume_unique=False)
    src_n1 = src_n1[mask]
    dst_n2 = dst_n2[mask]
    t1 = time.perf_counter()
    if verbose:
        print("Intersection edges:", len(dst_n2), t1 - t0)

    if len(dst_n2) == 0:
        return []

    # ─────────────────────────────────────────────────────────────
    # Assemble full paths (vectorized)
    # ─────────────────────────────────────────────────────────────
    paths = np.column_stack([
        np.full(len(src_n1), start_idx, dtype=np.int32),
        src_n1,
        dst_n2,
        np.full(len(src_n1), end_idx, dtype=np.int32),
    ])

    return paths.tolist()


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
        graph: CSRGraph instance
        start_id: Starting node ID
        end_id: Ending node ID
        allowed_predicates: If provided, only use edges with these predicates
        excluded_predicates: If provided, skip edges with these predicates
        verbose: Print progress information
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    if start_idx is None or end_idx is None:
        return []

    if start_idx == end_idx:
        return []

    # Helper to check if predicate is allowed
    def is_predicate_allowed(predicate):
        if excluded_predicates and predicate in excluded_predicates:
            return False
        if allowed_predicates and predicate not in allowed_predicates:
            return False
        return True

    if verbose:
        print(f"Start node '{start_id}' has degree: {graph.degree(start_idx):,}")
        print(f"End node '{end_id}' has degree: {graph.degree(end_idx):,}")

    # Forward: start -> n1 (with filtering)
    # Get all edges from start, filter by predicate
    forward_1_filtered = []  # List of (n1_idx, predicate) tuples

    for n1_idx, predicate, _ in graph.neighbors_with_properties(start_idx):
        if n1_idx == end_idx:
            continue
        if is_predicate_allowed(predicate):
            forward_1_filtered.append((n1_idx, predicate))

    if verbose:
        print(f"After filtering edge 1: {len(forward_1_filtered):,} neighbors")

    if len(forward_1_filtered) == 0:
        return []

    # Build mapping: n2_idx -> list of (n1_idx, pred_01, pred_12) tuples
    # This tracks both intermediate nodes and the predicates used
    forward_paths = defaultdict(list)

    for n1_idx, pred_01 in forward_1_filtered:
        for n2_idx, pred_12, _ in graph.neighbors_with_properties(n1_idx):
            if n2_idx == start_idx:
                continue
            if is_predicate_allowed(pred_12):
                forward_paths[n2_idx].append((n1_idx, pred_01, pred_12))

    if verbose:
        print(f"Nodes reachable in 2 hops: {len(forward_paths):,}")

    # Backward: end -> n2 (with filtering)
    # Store as dict: n2_idx -> predicate (for the n2 -> end edge)
    # backward_1_filtered = {}

    # for n2_idx, predicate, _ in graph.neighbors_with_properties(end_idx):
    #     if is_predicate_allowed(predicate):
    #         # Note: This is the reverse edge, so we need the predicate from n2 -> end
    #         # We need to get edges going TO end_idx, not FROM end_idx
    #         # So we need to check all edges between n2 and end in the correct direction
    #         pass

    # Better approach: iterate through potential n2 nodes from forward pass
    # and check if they connect to end
    backward_connections = {}  # n2_idx -> predicate (n2 -> end)

    for n2_idx in forward_paths.keys():
        # Get all edges from n2 to end
        edges_to_end = graph.get_all_edges_between(n2_idx, end_idx)
        for predicate, _ in edges_to_end:
            if is_predicate_allowed(predicate):
                # Store first valid predicate (or could store all)
                if n2_idx not in backward_connections:
                    backward_connections[n2_idx] = []
                backward_connections[n2_idx].append(predicate)

    if verbose:
        print(
            f"After filtering edge 3: {len(backward_connections):,} nodes connect to end"
        )

    # Find intersection and build paths
    paths = []
    for n2_idx in forward_paths:
        if n2_idx in backward_connections:
            # For each path through this n2
            for n1_idx, pred_01, pred_12 in forward_paths[n2_idx]:
                # For each valid predicate from n2 to end
                for pred_23 in backward_connections[n2_idx]:
                    n0 = {
                        "id": graph.get_node_id(start_idx),
                        "category": graph.get_node_property(start_idx, "category", []),
                        "name": graph.get_node_property(start_idx, "name"),
                    }
                    e0 = {
                        "predicate": pred_01,
                    }
                    n1 = {
                        "id": graph.get_node_id(n1_idx),
                        "category": graph.get_node_property(n1_idx, "category", []),
                        "name": graph.get_node_property(n1_idx, "name"),
                    }
                    e1 = {
                        "predicate": pred_12,
                    }
                    n2 = {
                        "id": graph.get_node_id(n2_idx),
                        "category": graph.get_node_property(n2_idx, "category", []),
                        "name": graph.get_node_property(n2_idx, "name"),
                    }
                    e2 = {
                        "predicate": pred_23,
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


def do_one_hop(graph: CSRGraph, start_id: str, verbose=True):
    """Get all neighbors from a single node."""
    start_idx = graph.get_node_idx(start_id)

    neighbors = graph.neighbors(start_idx)

    return neighbors


def lookup(graph, query: dict, verbose=True):
    """
    Take an arbitrary Translator query graph and return all matching paths.

    Args:
        graph: CSRGraph instance
        query_graph: Query graph with nodes and edges
        bmt: Biolink Model Toolkit instance
        verbose: Print progress information

    Returns:
        List of enriched path dictionaries matching the query graph structure
    """
    bmt = Toolkit()
    query_graph = query["message"]["query_graph"]
    subqgraph = copy.deepcopy(query_graph)

    # Store results for each edge query
    # edge_id -> list of (subject_idx, predicate, object_idx) tuples
    edge_results = {}

    # Track original query graph structure for path reconstruction
    original_edges = list(query_graph["edges"].keys())
    original_nodes = set(query_graph["nodes"].keys())

    if verbose:
        print(f"Query graph: {len(original_nodes)} nodes, {len(original_edges)} edges")

    # Process edges one at a time
    while len(subqgraph["edges"].keys()) > 0:
        # Get next edge to query
        next_edge_id, next_edge = get_next_qedge(subqgraph)

        if verbose:
            print(
                f"\nProcessing edge '{next_edge_id}': {next_edge['subject']} -> {next_edge['object']}"
            )

        # Get node constraints
        start_node = subqgraph["nodes"][next_edge["subject"]]
        end_node = subqgraph["nodes"][next_edge["object"]]

        # Get pinned node indices
        start_node_idxes = None
        if len(start_node.get("ids", [])) > 0:
            start_node_idxes = [
                graph.get_node_idx(node_id)
                for node_id in start_node["ids"]
                if graph.get_node_idx(node_id) is not None
            ]

        end_node_idxes = None
        if len(end_node.get("ids", [])) > 0:
            end_node_idxes = [
                graph.get_node_idx(node_id)
                for node_id in end_node["ids"]
                if graph.get_node_idx(node_id) is not None
            ]

        # Get allowed predicates (including descendants)
        allowed_predicates = [
            predicate
            for edge_predicate in next_edge["predicates"]
            for predicate in bmt.get_descendants(edge_predicate, formatted=True)
        ]
        print(allowed_predicates)
        # allowed_predicates = next_edge["predicates"]

        # Query for matching edges
        edge_matches = _query_edge(
            graph,
            start_node_idxes,
            end_node_idxes,
            start_node.get("categories", []),
            end_node.get("categories", []),
            allowed_predicates,
            verbose,
        )

        # Store results for this edge
        edge_results[next_edge_id] = edge_matches

        if verbose:
            print(f"  Found {len(edge_matches):,} matching edges")

        if len(edge_matches) <= 0:
            print("Found no edge matches, returning 0 results.")
            original_edges = []
            break

        # Update subgraph with discovered nodes for next iteration
        discovered_subjects = set()
        discovered_objects = set()

        for subj_idx, pred, obj_idx in edge_matches:
            discovered_subjects.add(graph.get_node_id(subj_idx))
            discovered_objects.add(graph.get_node_id(obj_idx))

        # Update node IDs in subgraph
        if len(discovered_subjects) > 0:
            subqgraph["nodes"][next_edge["subject"]]["ids"] = list(discovered_subjects)
        if len(discovered_objects) > 0:
            subqgraph["nodes"][next_edge["object"]]["ids"] = list(discovered_objects)

        # Remove processed edge
        subqgraph["edges"].pop(next_edge_id)

        # Remove orphaned nodes
        remove_orphaned(subqgraph)

        if verbose:
            print(f"  Remaining edges: {len(subqgraph['edges'])}")

    # Reconstruct complete paths from edge results
    if verbose:
        print(f"\nReconstructing complete paths...")

    paths = _reconstruct_paths(
        graph, query_graph, edge_results, original_edges, verbose
    )

    if verbose:
        print(f"Found {len(paths):,} complete paths")

    response = {
        "message": {
            "query_graph": query_graph,
            "knowledge_graph": {
                "nodes": {},
                "edges": {},
            },
            "results": []
        }
    }
    for path in paths:
        result = {
            "node_bindings": {},
            "analyses": [{
                "resource_id": "infores:gandalf",
                "edge_bindings": {},
            }],
        }
        for node_id, node in path["nodes"].items():
            response["message"]["knowledge_graph"]["nodes"][node["id"]] = node
            result["node_bindings"][node_id] = [
                {
                    "id": node["id"],
                    "attributes": [],
                },
            ]
        for edge_id, edge in path["edges"].items():
            edge_uuid = str(uuid.uuid4())[:8]
            response["message"]["knowledge_graph"]["edges"][edge_uuid] = edge
            result["analyses"][0]["edge_bindings"][edge_id] = [
                {
                    "id": edge_uuid,
                    "attributes": [],
                },
            ]
        response["message"]["results"].append(result)
    return response


def _query_edge(
    graph,
    start_idxes,
    end_idxes,
    start_categories,
    end_categories,
    allowed_predicates,
    verbose,
):
    """
    Query for a single edge with given constraints.

    Returns:
        List of (subject_idx, predicate, object_idx) tuples
    """
    matches = []

    # Case 1: Start pinned, end unpinned
    if start_idxes is not None and end_idxes is None:
        if verbose:
            print(f"  Forward search from {len(start_idxes)} pinned nodes")

        for start_idx in start_idxes:
            for obj_idx, predicate, _ in graph.neighbors_with_properties(start_idx):
                # Check predicate
                if allowed_predicates and predicate not in allowed_predicates:
                    continue

                # Check object categories
                if end_categories:
                    obj_cats = graph.get_node_property(obj_idx, "category", [])
                    if not any(cat in obj_cats for cat in end_categories):
                        continue

                matches.append((start_idx, predicate, obj_idx))

    # Case 2: Start unpinned, end pinned
    elif start_idxes is None and end_idxes is not None:
        if verbose:
            print(f"  Backward search from {len(end_idxes)} pinned nodes")

        for end_idx in end_idxes:
            for subj_idx, predicate, _ in graph.incoming_neighbors_with_properties(
                end_idx
            ):
                # Check predicate
                if allowed_predicates and predicate not in allowed_predicates:
                    continue

                # Check subject categories
                if start_categories:
                    subj_cats = graph.get_node_property(subj_idx, "category", [])
                    if not any(cat in subj_cats for cat in start_categories):
                        continue

                matches.append((subj_idx, predicate, end_idx))

    # Case 3: Both pinned
    elif start_idxes is not None and end_idxes is not None:
        if verbose:
            print(f"  Both ends pinned: {len(start_idxes)} start, {len(end_idxes)} end")

        t0 = time.perf_counter()

        # Build forward edges with predicates
        forward_edges = defaultdict(list)  # obj_idx -> [(subj_idx, predicate), ...]

        for start_idx in start_idxes:
            for obj_idx, predicate, _ in graph.neighbors_with_properties(start_idx):
                if allowed_predicates and predicate not in allowed_predicates:
                    continue
                forward_edges[obj_idx].append((start_idx, predicate))

        # Find intersection with end nodes
        end_set = set(end_idxes)

        for obj_idx in forward_edges.keys():
            if obj_idx in end_set:
                for subj_idx, predicate in forward_edges[obj_idx]:
                    matches.append((subj_idx, predicate, obj_idx))

        t1 = time.perf_counter()
        if verbose:
            print(f"  Found {len(matches)} matches in {t1 - t0:.3f}s")

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    return matches


def _reconstruct_paths(graph, query_graph, edge_results, edge_order, verbose):
    """
    Reconstruct complete paths by iteratively joining edge results.

    Uses a fast iterative join strategy instead of recursion.
    Optimized to use tuples during joining for better performance.

    Args:
        graph: CSRGraph instance
        query_graph: Original query graph
        edge_results: Dict of edge_id -> [(subj_idx, pred, obj_idx), ...]
        edge_order: List of edge IDs in original query order
        verbose: Print progress

    Returns:
        List of enriched path dictionaries
    """
    if len(edge_order) == 0:
        return []

    t0 = time.perf_counter()

    # Build join order based on query graph structure
    join_order = _compute_join_order(query_graph, edge_results, edge_order, verbose)

    if verbose:
        print(f"  Join order: {join_order}")

    # Start with the first edge results
    first_edge_id = join_order[0]
    first_edge = query_graph["edges"][first_edge_id]
    subj_qnode = first_edge["subject"]
    obj_qnode = first_edge["object"]

    # Use tuple-based representation for efficiency during joining:
    # Each path is: (nodes_tuple, edges_tuple)
    # nodes_tuple: ((qnode_id, node_idx), ...)
    # edges_tuple: ((qedge_id, subj_idx, predicate, obj_idx), ...)
    # Tuples are immutable and much faster to create than dicts
    partial_paths = [
        (
            ((subj_qnode, subj_idx), (obj_qnode, obj_idx)),
            ((first_edge_id, subj_idx, predicate, obj_idx),),
        )
        for subj_idx, predicate, obj_idx in edge_results[first_edge_id]
    ]

    if verbose:
        print(
            f"  Starting with {len(partial_paths):,} paths from edge '{first_edge_id}'"
        )

    # Iteratively join with remaining edges
    for i, edge_id in enumerate(join_order[1:], 1):
        edge = query_graph["edges"][edge_id]
        subj_qnode = edge["subject"]
        obj_qnode = edge["object"]

        if verbose:
            print(
                f"  Join {i}/{len(join_order) - 1}: Adding edge '{edge_id}' ({len(partial_paths):,} paths)...",
                end="",
            )

        t_join_start = time.perf_counter()

        # Check which nodes are already in paths (check first path as representative)
        if partial_paths:
            first_path_nodes = {qn: idx for qn, idx in partial_paths[0][0]}
            subj_in_paths = subj_qnode in first_path_nodes
            obj_in_paths = obj_qnode in first_path_nodes
        else:
            subj_in_paths = False
            obj_in_paths = False

        new_paths = []

        if subj_in_paths and obj_in_paths:
            # Both nodes already in path - validate consistency using index
            # Build index: (subj_idx, obj_idx) -> [predicates]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_results[edge_id]:
                edge_index[(subj_idx, obj_idx)].append(predicate)

            for nodes_tuple, edges_tuple in partial_paths:
                nodes_dict = {qn: idx for qn, idx in nodes_tuple}
                expected_subj = nodes_dict.get(subj_qnode)
                expected_obj = nodes_dict.get(obj_qnode)
                key = (expected_subj, expected_obj)

                if key in edge_index:
                    for predicate in edge_index[key]:
                        new_edges = edges_tuple + (
                            (edge_id, expected_subj, predicate, expected_obj),
                        )
                        new_paths.append((nodes_tuple, new_edges))

        elif subj_in_paths:
            # Join on subject node
            # Build index: subj_idx -> [(pred, obj_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_results[edge_id]:
                edge_index[subj_idx].append((predicate, obj_idx))

            for nodes_tuple, edges_tuple in partial_paths:
                nodes_dict = {qn: idx for qn, idx in nodes_tuple}
                subj_idx = nodes_dict.get(subj_qnode)

                if subj_idx in edge_index:
                    for predicate, obj_idx in edge_index[subj_idx]:
                        new_nodes = nodes_tuple + ((obj_qnode, obj_idx),)
                        new_edges = edges_tuple + (
                            (edge_id, subj_idx, predicate, obj_idx),
                        )
                        new_paths.append((new_nodes, new_edges))

        elif obj_in_paths:
            # Join on object node
            # Build index: obj_idx -> [(subj_idx, pred), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_results[edge_id]:
                edge_index[obj_idx].append((subj_idx, predicate))

            for nodes_tuple, edges_tuple in partial_paths:
                nodes_dict = {qn: idx for qn, idx in nodes_tuple}
                obj_idx = nodes_dict.get(obj_qnode)

                if obj_idx in edge_index:
                    for subj_idx, predicate in edge_index[obj_idx]:
                        new_nodes = nodes_tuple + ((subj_qnode, subj_idx),)
                        new_edges = edges_tuple + (
                            (edge_id, subj_idx, predicate, obj_idx),
                        )
                        new_paths.append((new_nodes, new_edges))

        else:
            # Neither node in paths - cartesian product (should be rare with good join order)
            if verbose:
                print(f"\n    Warning: Cartesian product needed for edge '{edge_id}'")

            for nodes_tuple, edges_tuple in partial_paths:
                for subj_idx, predicate, obj_idx in edge_results[edge_id]:
                    new_nodes = nodes_tuple + (
                        (subj_qnode, subj_idx),
                        (obj_qnode, obj_idx),
                    )
                    new_edges = edges_tuple + (
                        (edge_id, subj_idx, predicate, obj_idx),
                    )
                    new_paths.append((new_nodes, new_edges))

        partial_paths = new_paths

        t_join_end = time.perf_counter()
        if verbose:
            print(
                f" -> {len(partial_paths):,} paths ({t_join_end - t_join_start:.2f}s)"
            )

        # Early termination if no paths remain
        if len(partial_paths) == 0:
            if verbose:
                print(f"  No valid paths found after joining edge '{edge_id}'")
            break

    t1 = time.perf_counter()
    if verbose:
        print(f"  Path reconstruction took {t1 - t0:.2f}s")

    # Build node property cache directly from tuple-based paths
    if verbose:
        print(f"  Enriching {len(partial_paths):,} paths...")

    t_cache_start = time.perf_counter()
    unique_node_indices = set()
    for nodes_tuple, edges_tuple in partial_paths:
        for qn, idx in nodes_tuple:
            unique_node_indices.add(idx)

    # Fetch properties for each unique node once
    node_cache = {}
    for node_idx in unique_node_indices:
        node_cache[node_idx] = {
            "id": graph.get_node_id(node_idx),
            "category": graph.get_node_property(node_idx, "category", []),
            "name": graph.get_node_property(node_idx, "name"),
        }

    t_cache_end = time.perf_counter()
    if verbose:
        print(
            f"  Cached properties for {len(unique_node_indices):,} unique nodes "
            f"({t_cache_end - t_cache_start:.2f}s)"
        )

    # Enrich paths directly from tuples (no dict conversion needed)
    enriched_paths = [
        _enrich_path_from_tuples(query_graph, nodes_tuple, edges_tuple, node_cache)
        for nodes_tuple, edges_tuple in partial_paths
    ]

    return enriched_paths


def _compute_join_order(query_graph, edge_results, edge_order, verbose):
    """
    Compute optimal join order to minimize intermediate results.

    Strategy:
    1. Start with smallest edge
    2. Greedily add edges that share nodes with current partial path
    3. Prefer edges that will filter (both nodes already in path)
    """
    remaining_edges = set(edge_order)
    join_order = []
    nodes_in_path = set()

    # Start with the edge with fewest results
    first_edge = min(remaining_edges, key=lambda e: len(edge_results.get(e, [])))
    join_order.append(first_edge)
    remaining_edges.remove(first_edge)

    # Add nodes from first edge to path
    first_edge_info = query_graph["edges"][first_edge]
    nodes_in_path.add(first_edge_info["subject"])
    nodes_in_path.add(first_edge_info["object"])

    # Greedily add remaining edges
    while remaining_edges:
        best_edge = None
        best_score = -1

        for edge_id in remaining_edges:
            print(edge_id)
            edge = query_graph["edges"][edge_id]
            subj = edge["subject"]
            obj = edge["object"]
            print(subj, obj)

            # Score based on:
            # - How many nodes are already in path (higher is better for joining)
            # - Size of edge results (smaller is better)
            nodes_shared = (subj in nodes_in_path) + (obj in nodes_in_path)
            result_size = len(edge_results.get(edge_id, []))

            # Prefer edges with shared nodes, then smaller result sets
            score = nodes_shared * 1000000000 - result_size
            print(score)

            if score > best_score:
                print("setting the new best score")
                best_score = score
                best_edge = edge_id

        join_order.append(best_edge)
        remaining_edges.remove(best_edge)

        # Add new nodes to path
        edge_info = query_graph["edges"][best_edge]
        nodes_in_path.add(edge_info["subject"])
        nodes_in_path.add(edge_info["object"])

    return join_order


def _enrich_path_from_tuples(query_graph, nodes_tuple, edges_tuple, node_cache):
    """
    Convert tuple-based path to enriched format with node/edge properties.

    Args:
        query_graph: Original query graph
        nodes_tuple: ((qnode_id, node_idx), ...)
        edges_tuple: ((qedge_id, subj_idx, predicate, obj_idx), ...)
        node_cache: Dict mapping node_idx -> {id, category, name} for cached lookups

    Returns:
        Enriched path dictionary matching query graph structure
    """
    # Build nodes dict directly from tuple
    nodes = {qnode_id: node_cache[node_idx] for qnode_id, node_idx in nodes_tuple}

    # Build edges dict directly from tuple
    edges = {
        qedge_id: {
            "predicate": predicate,
            "subject": node_cache[subj_idx]["id"],
            "object": node_cache[obj_idx]["id"],
        }
        for qedge_id, subj_idx, predicate, obj_idx in edges_tuple
    }

    return {"nodes": nodes, "edges": edges}
