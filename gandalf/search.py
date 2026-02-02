"""General graph search functions."""

import copy
import gc
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from bmt.toolkit import Toolkit

from gandalf.graph import CSRGraph
from gandalf.query_planner import get_next_qedge, remove_orphaned


def _edge_matches_qualifier_constraints(edge_qualifiers, qualifier_constraints):
    """
    Check if an edge's qualifiers match the query's qualifier constraints.

    Qualifier constraints use OR semantics between qualifier_sets and AND semantics
    within each qualifier_set. An edge matches if it satisfies at least one
    qualifier_set (i.e., has ALL qualifiers in that set).

    Args:
        edge_qualifiers: List of qualifier dicts from the edge, each with
                        'qualifier_type_id' and 'qualifier_value'
        qualifier_constraints: List of constraint dicts from the query, each with
                              a 'qualifier_set' containing qualifiers to match

    Returns:
        True if the edge matches at least one qualifier_set, False otherwise.
        Returns True if qualifier_constraints is None or empty.
    """
    # No constraints means all edges match
    if not qualifier_constraints:
        return True

    # Build a set of (type_id, value) tuples from edge qualifiers for fast lookup
    edge_qualifier_set = set()
    if edge_qualifiers:
        for q in edge_qualifiers:
            type_id = q.get("qualifier_type_id")
            value = q.get("qualifier_value")
            if type_id and value:
                edge_qualifier_set.add((type_id, value))

    # Check if edge satisfies at least one qualifier_set (OR semantics)
    for constraint in qualifier_constraints:
        qualifier_set = constraint.get("qualifier_set", [])
        if not qualifier_set:
            # Empty qualifier_set matches any edge
            return True

        # Check if edge has ALL qualifiers in this set (AND semantics)
        all_match = True
        for required_qualifier in qualifier_set:
            req_type = required_qualifier.get("qualifier_type_id")
            req_value = required_qualifier.get("qualifier_value")
            if (req_type, req_value) not in edge_qualifier_set:
                all_match = False
                break

        if all_match:
            return True

    return False


class GCMonitor:
    """Monitor garbage collection events and log timing information."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.gc_events = []
        self._start_time = None

    def _gc_callback(self, phase, info):
        """Callback invoked by gc module on collection events."""
        if phase == "start":
            self._start_time = time.perf_counter()
        elif phase == "stop" and self._start_time is not None:
            duration = time.perf_counter() - self._start_time
            generation = info.get("generation", "?")
            collected = info.get("collected", 0)
            self.gc_events.append({
                "generation": generation,
                "duration": duration,
                "collected": collected,
            })
            if self.verbose and duration > 0.1:  # Only log slow GC (>100ms)
                print(f"  [GC] Gen {generation}: {duration:.2f}s, collected {collected} objects")
            self._start_time = None

    def start(self):
        """Start monitoring GC events."""
        self.gc_events = []
        gc.callbacks.append(self._gc_callback)

    def stop(self):
        """Stop monitoring and return summary."""
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
        return self.gc_events

    def summary(self):
        """Return summary of GC activity."""
        if not self.gc_events:
            return None
        total_time = sum(e["duration"] for e in self.gc_events)
        total_collected = sum(e["collected"] for e in self.gc_events)
        return {
            "total_collections": len(self.gc_events),
            "total_time": total_time,
            "total_collected": total_collected,
        }


@contextmanager
def gc_disabled():
    """Context manager to temporarily disable GC."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


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


def lookup(graph, query: dict, bmt=None, verbose=True):
    """
    Take an arbitrary Translator query graph and return all matching paths.

    Args:
        graph: CSRGraph instance
        query_graph: Query graph with nodes and edges
        bmt: Biolink Model Toolkit instance (optional, will create if not provided)
        verbose: Print progress information

    Returns:
        List of enriched path dictionaries matching the query graph structure
    """
    t_start = time.perf_counter()

    # Start GC monitoring to track collection events
    gc_monitor = GCMonitor(verbose=verbose)
    gc_monitor.start()

    if bmt is None:
        bmt = Toolkit()
        t_bmt = time.perf_counter()
        if verbose:
            print(f"BMT initialization: {t_bmt - t_start:.2f}s")
    elif verbose:
        print("Using provided BMT instance")

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
            if bmt.get_element(edge_predicate) is not None
            for predicate in bmt.get_descendants(edge_predicate, formatted=True)
        ]
        # allowed_predicates = next_edge["predicates"]

        # Get qualifier constraints for this edge
        qualifier_constraints = next_edge.get("qualifier_constraints", [])

        # Query for matching edges
        edge_matches = _query_edge(
            graph,
            start_node_idxes,
            end_node_idxes,
            start_node.get("categories", []),
            end_node.get("categories", []),
            allowed_predicates,
            qualifier_constraints,
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

    t_post_start = time.perf_counter()

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

    # Group paths by unique node binding combinations
    # Key: tuple of (qnode_id, node_id) pairs sorted by qnode_id
    # Value: list of edge dictionaries from paths with this node combination
    node_binding_groups = defaultdict(list)

    for path in paths:
        # Create a hashable key from node bindings
        node_key = tuple(
            sorted((qnode_id, node["id"]) for qnode_id, node in path["nodes"].items())
        )
        node_binding_groups[node_key].append(path)

    t_grouped = time.perf_counter()
    if verbose:
        print(f"  Grouped into {len(node_binding_groups):,} unique node paths ({t_grouped - t_post_start:.2f}s)")

    # Disable GC during result building to prevent stochastic pauses
    # GC can cause 30+ second delays when triggered during this loop
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        # Build results - one per unique node binding combination
        for node_key, grouped_paths in node_binding_groups.items():
            # Use first path for node info (all paths in group have same nodes)
            first_path = grouped_paths[0]

            result = {
                "node_bindings": {},
                "analyses": [{
                    "resource_id": "infores:gandalf",
                    "edge_bindings": {},
                }],
            }

            # Add node bindings (same for all paths in group)
            for node_id, node in first_path["nodes"].items():
                response["message"]["knowledge_graph"]["nodes"][node["id"]] = node
                result["node_bindings"][node_id] = [
                    {
                        "id": node["id"],
                        "attributes": [],
                    },
                ]

            # Aggregate edge bindings from all paths in group
            # For each query edge, collect all matching edges
            edge_bindings_by_qedge = defaultdict(list)

            for path in grouped_paths:
                for edge_id, edge in path["edges"].items():
                    # Create a unique key for this edge to avoid duplicates
                    edge_key = (edge["subject"], edge["predicate"], edge["object"])

                    # Check if we already have this exact edge
                    existing_keys = [
                        (e["subject"], e["predicate"], e["object"])
                        for e in edge_bindings_by_qedge[edge_id]
                    ]
                    if edge_key not in existing_keys:
                        edge_bindings_by_qedge[edge_id].append(edge)

            # Add edges to knowledge graph and result bindings
            for edge_id, edges in edge_bindings_by_qedge.items():
                result["analyses"][0]["edge_bindings"][edge_id] = []
                for edge in edges:
                    edge_uuid = str(uuid.uuid4())[:8]
                    response["message"]["knowledge_graph"]["edges"][edge_uuid] = edge
                    result["analyses"][0]["edge_bindings"][edge_id].append(
                        {
                            "id": edge_uuid,
                            "attributes": [],
                        }
                    )

            response["message"]["results"].append(result)
    finally:
        # Re-enable GC if it was enabled before
        if gc_was_enabled:
            gc.enable()

    t_built = time.perf_counter()
    if verbose:
        print(f"  Built {len(response['message']['results']):,} results ({t_built - t_grouped:.2f}s)")
        print(f"  Post-processing total: {t_built - t_post_start:.2f}s")

    # Stop GC monitoring and show summary
    gc_monitor.stop()
    gc_summary = gc_monitor.summary()
    if verbose and gc_summary and gc_summary["total_time"] > 0.1:
        print(f"  [GC Summary] {gc_summary['total_collections']} collections, "
              f"{gc_summary['total_time']:.2f}s total, "
              f"{gc_summary['total_collected']} objects collected")

    return response


def _query_edge(
    graph,
    start_idxes,
    end_idxes,
    start_categories,
    end_categories,
    allowed_predicates,
    qualifier_constraints,
    verbose,
):
    """
    Query for a single edge with given constraints.

    Args:
        graph: CSRGraph instance
        start_idxes: List of pinned start node indices, or None if unpinned
        end_idxes: List of pinned end node indices, or None if unpinned
        start_categories: List of allowed categories for start node
        end_categories: List of allowed categories for end node
        allowed_predicates: List of allowed predicate strings
        qualifier_constraints: List of qualifier constraint dicts from query
        verbose: Print progress information

    Returns:
        List of (subject_idx, predicate, object_idx) tuples
    """
    matches = []

    # Case 1: Start pinned, end unpinned
    if start_idxes is not None and end_idxes is None:
        if verbose:
            print(f"  Forward search from {len(start_idxes)} pinned nodes")

        t0 = time.perf_counter()

        total_neighbors = 0
        slow_nodes = []  # Track nodes that take > 0.1s

        for start_idx in start_idxes:
            t_node_start = time.perf_counter()
            node_neighbors = 0

            for obj_idx, predicate, props in graph.neighbors_with_properties(start_idx):
                node_neighbors += 1
                # Check predicate
                if allowed_predicates and predicate not in allowed_predicates:
                    continue

                # Check object categories
                if end_categories:
                    obj_cats = graph.get_node_property(obj_idx, "categories", [])
                    if not any(cat in obj_cats for cat in end_categories):
                        continue

                # Check qualifier constraints
                if qualifier_constraints:
                    edge_qualifiers = props.get("qualifiers", [])
                    if not _edge_matches_qualifier_constraints(
                        edge_qualifiers, qualifier_constraints
                    ):
                        continue

                matches.append((start_idx, predicate, obj_idx))

            t_node_end = time.perf_counter()
            node_time = t_node_end - t_node_start
            total_neighbors += node_neighbors

            if node_time > 0.1:  # Track slow nodes
                slow_nodes.append((start_idx, node_neighbors, node_time))

        t1 = time.perf_counter()
        if verbose:
            print(f"  Traversed {total_neighbors:,} total neighbors")
            if slow_nodes:
                print(f"  Slow nodes (>0.1s): {len(slow_nodes)}")
                for node_idx, neighbors, node_time in slow_nodes[:5]:  # Show top 5
                    print(f"    Node {node_idx}: {neighbors:,} neighbors, {node_time:.2f}s")
            print(f"  Found {len(matches):,} matches in {t1 - t0:.3f}s")

    # Case 2: Start unpinned, end pinned
    elif start_idxes is None and end_idxes is not None:
        if verbose:
            print(f"  Backward search from {len(end_idxes)} pinned nodes")

        t0 = time.perf_counter()

        total_neighbors = 0
        slow_nodes = []  # Track nodes that take > 0.1s

        for i, end_idx in enumerate(end_idxes):
            t_node_start = time.perf_counter()
            node_neighbors = 0

            for subj_idx, predicate, props in graph.incoming_neighbors_with_properties(
                end_idx
            ):
                node_neighbors += 1
                # Check predicate
                if allowed_predicates and predicate not in allowed_predicates:
                    continue

                # Check subject categories
                if start_categories:
                    subj_cats = graph.get_node_property(subj_idx, "categories", [])
                    if not any(cat in subj_cats for cat in start_categories):
                        continue

                # Check qualifier constraints
                if qualifier_constraints:
                    edge_qualifiers = props.get("qualifiers", [])
                    if not _edge_matches_qualifier_constraints(
                        edge_qualifiers, qualifier_constraints
                    ):
                        continue

                matches.append((subj_idx, predicate, end_idx))

            t_node_end = time.perf_counter()
            node_time = t_node_end - t_node_start
            total_neighbors += node_neighbors

            if node_time > 0.1:  # Track slow nodes
                slow_nodes.append((end_idx, node_neighbors, node_time))

        t1 = time.perf_counter()
        if verbose:
            print(f"  Traversed {total_neighbors:,} total incoming neighbors")
            if slow_nodes:
                print(f"  Slow nodes (>0.1s): {len(slow_nodes)}")
                for node_idx, neighbors, node_time in slow_nodes[:5]:  # Show top 5
                    print(f"    Node {node_idx}: {neighbors:,} neighbors, {node_time:.2f}s")
            print(f"  Found {len(matches):,} matches in {t1 - t0:.3f}s")

    # Case 3: Both pinned
    elif start_idxes is not None and end_idxes is not None:
        if verbose:
            print(f"  Both ends pinned: {len(start_idxes)} start, {len(end_idxes)} end")

        t0 = time.perf_counter()

        # Build forward edges with predicates and props for qualifier checking
        # obj_idx -> [(subj_idx, predicate, props), ...]
        forward_edges = defaultdict(list)

        t_neighbors_start = time.perf_counter()
        total_neighbors = 0
        for start_idx in start_idxes:
            for obj_idx, predicate, props in graph.neighbors_with_properties(start_idx):
                total_neighbors += 1
                if allowed_predicates and predicate not in allowed_predicates:
                    continue
                forward_edges[obj_idx].append((start_idx, predicate, props))

        t_neighbors_end = time.perf_counter()
        if verbose:
            print(f"    Neighbor traversal: {t_neighbors_end - t_neighbors_start:.3f}s "
                  f"({total_neighbors:,} neighbors, {len(forward_edges):,} unique targets)")

        # Find intersection with end nodes
        t_intersect_start = time.perf_counter()
        end_set = set(end_idxes)

        for obj_idx in forward_edges.keys():
            if obj_idx in end_set:
                for subj_idx, predicate, props in forward_edges[obj_idx]:
                    # Check qualifier constraints
                    if qualifier_constraints:
                        edge_qualifiers = props.get("qualifiers", [])
                        if not _edge_matches_qualifier_constraints(
                            edge_qualifiers, qualifier_constraints
                        ):
                            continue
                    matches.append((subj_idx, predicate, obj_idx))

        t1 = time.perf_counter()
        if verbose:
            print(f"    Intersection: {t1 - t_intersect_start:.3f}s")
            print(f"  Found {len(matches):,} matches in {t1 - t0:.3f}s")

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    return matches


def _reconstruct_paths(graph, query_graph, edge_results, edge_order, verbose):
    """
    Reconstruct complete paths by iteratively joining edge results.

    Uses NumPy arrays for efficient memory usage and to avoid GC pressure.
    For 15M+ paths, this eliminates Python object creation overhead.

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

    # Build mappings for query structure
    # qnode_id -> column index in node array
    qnode_to_col = {}
    # qedge_id -> column index in predicate array
    qedge_to_col = {eid: i for i, eid in enumerate(join_order)}

    # Build predicate vocabulary: predicate_string -> int
    predicate_to_idx = {}
    idx_to_predicate = []

    def get_pred_idx(pred):
        if pred not in predicate_to_idx:
            predicate_to_idx[pred] = len(idx_to_predicate)
            idx_to_predicate.append(pred)
        return predicate_to_idx[pred]

    # Start with the first edge results
    first_edge_id = join_order[0]
    first_edge = query_graph["edges"][first_edge_id]
    subj_qnode = first_edge["subject"]
    obj_qnode = first_edge["object"]

    # Assign column indices for first edge's nodes
    qnode_to_col[subj_qnode] = 0
    qnode_to_col[obj_qnode] = 1
    num_node_cols = 2

    # Convert first edge results to numpy arrays
    first_results = edge_results[first_edge_id]
    num_paths = len(first_results)

    if num_paths == 0:
        return []

    # Pre-allocate arrays for nodes and predicates
    # nodes: shape (num_paths, max_nodes) - will grow columns as needed
    # preds: shape (num_paths, num_edges)
    max_nodes = len(query_graph["nodes"])
    num_edges = len(join_order)

    paths_nodes = np.zeros((num_paths, max_nodes), dtype=np.int32)
    paths_preds = np.zeros((num_paths, num_edges), dtype=np.int32)

    # Fill in first edge data
    for i, (subj_idx, predicate, obj_idx) in enumerate(first_results):
        paths_nodes[i, 0] = subj_idx
        paths_nodes[i, 1] = obj_idx
        paths_preds[i, 0] = get_pred_idx(predicate)

    if verbose:
        print(f"  Starting with {num_paths:,} paths from edge '{first_edge_id}'")

    # Iteratively join with remaining edges
    for join_idx, edge_id in enumerate(join_order[1:], 1):
        edge = query_graph["edges"][edge_id]
        subj_qnode = edge["subject"]
        obj_qnode = edge["object"]

        if verbose:
            print(
                f"  Join {join_idx}/{len(join_order) - 1}: Adding edge '{edge_id}' ({len(paths_nodes):,} paths)...",
                end="",
            )

        t_join_start = time.perf_counter()

        subj_in_paths = subj_qnode in qnode_to_col
        obj_in_paths = obj_qnode in qnode_to_col

        edge_data = edge_results[edge_id]

        if subj_in_paths and obj_in_paths:
            # Both nodes already in path - validate consistency
            subj_col = qnode_to_col[subj_qnode]
            obj_col = qnode_to_col[obj_qnode]

            # Build index: (subj_idx, obj_idx) -> [pred_idx, ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_data:
                edge_index[(subj_idx, obj_idx)].append(get_pred_idx(predicate))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []

            for path_idx in range(len(paths_nodes)):
                key = (paths_nodes[path_idx, subj_col], paths_nodes[path_idx, obj_col])
                if key in edge_index:
                    for pred_idx in edge_index[key]:
                        new_nodes_list.append(paths_nodes[path_idx].copy())
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)

        elif subj_in_paths:
            # Join on subject node, add object node
            subj_col = qnode_to_col[subj_qnode]

            # Assign column for new object node
            if obj_qnode not in qnode_to_col:
                qnode_to_col[obj_qnode] = num_node_cols
                num_node_cols += 1
            obj_col = qnode_to_col[obj_qnode]

            # Build index: subj_idx -> [(pred_idx, obj_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_data:
                edge_index[subj_idx].append((get_pred_idx(predicate), obj_idx))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []

            for path_idx in range(len(paths_nodes)):
                subj_idx = paths_nodes[path_idx, subj_col]
                if subj_idx in edge_index:
                    for pred_idx, obj_idx in edge_index[subj_idx]:
                        new_nodes = paths_nodes[path_idx].copy()
                        new_nodes[obj_col] = obj_idx
                        new_nodes_list.append(new_nodes)
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)

        elif obj_in_paths:
            # Join on object node, add subject node
            obj_col = qnode_to_col[obj_qnode]

            # Assign column for new subject node
            if subj_qnode not in qnode_to_col:
                qnode_to_col[subj_qnode] = num_node_cols
                num_node_cols += 1
            subj_col = qnode_to_col[subj_qnode]

            # Build index: obj_idx -> [(subj_idx, pred_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx in edge_data:
                edge_index[obj_idx].append((subj_idx, get_pred_idx(predicate)))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []

            for path_idx in range(len(paths_nodes)):
                obj_idx = paths_nodes[path_idx, obj_col]
                if obj_idx in edge_index:
                    for subj_idx, pred_idx in edge_index[obj_idx]:
                        new_nodes = paths_nodes[path_idx].copy()
                        new_nodes[subj_col] = subj_idx
                        new_nodes_list.append(new_nodes)
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)

        else:
            # Neither node in paths - cartesian product
            if verbose:
                print(f"\n    Warning: Cartesian product needed for edge '{edge_id}'")

            # Assign columns for both nodes
            if subj_qnode not in qnode_to_col:
                qnode_to_col[subj_qnode] = num_node_cols
                num_node_cols += 1
            if obj_qnode not in qnode_to_col:
                qnode_to_col[obj_qnode] = num_node_cols
                num_node_cols += 1
            subj_col = qnode_to_col[subj_qnode]
            obj_col = qnode_to_col[obj_qnode]

            new_nodes_list = []
            new_preds_list = []

            for path_idx in range(len(paths_nodes)):
                for subj_idx, predicate, obj_idx in edge_data:
                    new_nodes = paths_nodes[path_idx].copy()
                    new_nodes[subj_col] = subj_idx
                    new_nodes[obj_col] = obj_idx
                    new_nodes_list.append(new_nodes)
                    new_preds = paths_preds[path_idx].copy()
                    new_preds[join_idx] = get_pred_idx(predicate)
                    new_preds_list.append(new_preds)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)

        t_join_end = time.perf_counter()
        if verbose:
            print(f" -> {len(paths_nodes):,} paths ({t_join_end - t_join_start:.2f}s)")

        if len(paths_nodes) == 0:
            if verbose:
                print(f"  No valid paths found after joining edge '{edge_id}'")
            break

    t1 = time.perf_counter()
    if verbose:
        print(f"  Path reconstruction took {t1 - t0:.2f}s")

    num_paths = len(paths_nodes)
    if num_paths == 0:
        return []

    # Build node property cache
    if verbose:
        print(f"  Enriching {num_paths:,} paths...")

    t_cache_start = time.perf_counter()

    # Get unique node indices from numpy array (much faster than Python loops)
    unique_node_indices = np.unique(paths_nodes[:, :num_node_cols])

    node_cache = {}
    for node_idx in unique_node_indices:
        # Start with all stored node properties
        node_props = graph.get_all_node_properties(node_idx).copy()
        # Ensure required fields are present
        node_props["id"] = graph.get_node_id(node_idx)
        if "categories" not in node_props:
            node_props["categories"] = []
        if "attributes" not in node_props:
            node_props["attributes"] = []
        node_cache[node_idx] = node_props

    t_cache_end = time.perf_counter()
    if verbose:
        print(
            f"  Cached properties for {len(unique_node_indices):,} unique nodes "
            f"({t_cache_end - t_cache_start:.2f}s)"
        )

    # Convert numpy arrays to enriched path format
    t_enrich_start = time.perf_counter()

    # Reverse mappings for output
    col_to_qnode = {v: k for k, v in qnode_to_col.items()}
    col_to_qedge = {v: k for k, v in qedge_to_col.items()}

    # Disable GC during enrichment to prevent stochastic pauses
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        enriched_paths = []
        for path_idx in range(num_paths):
            nodes = {}
            for col in range(num_node_cols):
                qnode_id = col_to_qnode[col]
                node_idx = paths_nodes[path_idx, col]
                nodes[qnode_id] = node_cache[node_idx]

            edges = {}
            for col in range(num_edges):
                qedge_id = col_to_qedge[col]
                pred_idx = paths_preds[path_idx, col]
                predicate = idx_to_predicate[pred_idx]

                # Get subject and object from query graph structure
                edge_def = query_graph["edges"][qedge_id]
                subj_qnode = edge_def["subject"]
                obj_qnode = edge_def["object"]
                subj_col = qnode_to_col[subj_qnode]
                obj_col = qnode_to_col[obj_qnode]

                subj_idx = paths_nodes[path_idx, subj_col]
                obj_idx = paths_nodes[path_idx, obj_col]

                # Get all edge properties
                edge_props = graph.get_all_edge_properties(
                    int(subj_idx), int(obj_idx), predicate
                ).copy()

                # Ensure required fields are present
                edge_props["predicate"] = predicate
                edge_props["subject"] = node_cache[subj_idx]["id"]
                edge_props["object"] = node_cache[obj_idx]["id"]

                edges[qedge_id] = edge_props

            enriched_paths.append({"nodes": nodes, "edges": edges})
    finally:
        if gc_was_enabled:
            gc.enable()

    t_enrich_end = time.perf_counter()
    if verbose:
        print(f"  Enrichment took {t_enrich_end - t_enrich_start:.2f}s")

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
            edge = query_graph["edges"][edge_id]
            subj = edge["subject"]
            obj = edge["object"]

            # Score based on:
            # - How many nodes are already in path (higher is better for joining)
            # - Size of edge results (smaller is better)
            nodes_shared = (subj in nodes_in_path) + (obj in nodes_in_path)
            result_size = len(edge_results.get(edge_id, []))

            # Prefer edges with shared nodes, then smaller result sets
            score = nodes_shared * 1000000000 - result_size

            if score > best_score:
                best_score = score
                best_edge = edge_id

        join_order.append(best_edge)
        remaining_edges.remove(best_edge)

        # Add new nodes to path
        edge_info = query_graph["edges"][best_edge]
        nodes_in_path.add(edge_info["subject"])
        nodes_in_path.add(edge_info["object"])

    return join_order
