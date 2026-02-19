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

# When path count exceeds this threshold, skip edge attribute enrichment
# (sources, qualifiers, publications, attributes from LMDB) and only include
# predicates. This avoids expensive per-edge property lookups on large result sets.
LARGE_RESULT_PATH_THRESHOLD = 50_000


class QualifierExpander:
    """
    Handles qualifier value expansion using BMT hierarchy at query time.

    This class caches BMT lookups for performance and provides methods to expand
    qualifier values to include their descendants, following the reasoner-transpiler
    approach.

    For example, if a query specifies qualifier_value "activity_or_abundance",
    this will expand to also match "activity" and "abundance" (the child values).
    """

    def __init__(self, bmt: Toolkit):
        self.bmt = bmt
        self._descendants_cache: dict[tuple[str, str], list[str]] = {}
        self._enum_names: list[str] | None = None

    def _get_enum_names(self) -> list[str]:
        """Get all enum names from the Biolink Model (cached)."""
        if self._enum_names is None:
            try:
                self._enum_names = list(self.bmt.view.all_enums().keys())
            except Exception:
                self._enum_names = []
        return self._enum_names

    def get_value_descendants(self, qualifier_value: str) -> list[str]:
        """
        Get all descendant values for a qualifier value across all Biolink enums.

        Args:
            qualifier_value: The qualifier value to expand

        Returns:
            List of descendant values (including the original value)
        """
        # Check cache with just the value as key (we search all enums)
        cache_key = ("_all_", qualifier_value)
        if cache_key in self._descendants_cache:
            return self._descendants_cache[cache_key]

        descendants = set()
        descendants.add(qualifier_value)  # Always include the original value

        # Search all enums for this value and get descendants
        for enum_name in self._get_enum_names():
            try:
                if self.bmt.is_permissible_value_of_enum(
                    enum_name=enum_name, value=qualifier_value
                ):
                    enum_descendants = self.bmt.get_permissible_value_descendants(
                        permissible_value=qualifier_value, enum_name=enum_name
                    )
                    if enum_descendants:
                        descendants.update(enum_descendants)
            except Exception:
                # If BMT methods fail, continue with other enums
                continue

        result = list(descendants)
        self._descendants_cache[cache_key] = result
        return result

    def expand_qualifier_constraints(
        self, qualifier_constraints: list[dict]
    ) -> list[dict]:
        """
        Expand qualifier constraints to include descendant values.

        This transforms each qualifier in a qualifier_set by expanding its value
        to include descendant values. The result uses a special format where each
        qualifier has "qualifier_values" (plural) containing all acceptable values.

        The matching semantics remain:
        - OR between qualifier_sets (edge matches if ANY set matches)
        - AND within each qualifier_set (edge must match ALL qualifiers in a set)
        - OR between expanded values (edge matches if it has ANY of the descendant values)

        Args:
            qualifier_constraints: List of qualifier constraint dicts, each with
                                   a 'qualifier_set' containing qualifiers to match

        Returns:
            Expanded qualifier constraints with "qualifier_values" lists
        """
        if not qualifier_constraints:
            return qualifier_constraints

        expanded_constraints = []
        for constraint in qualifier_constraints:
            qualifier_set = constraint.get("qualifier_set", [])
            if not qualifier_set:
                # Empty qualifier_set matches any edge, keep as-is
                expanded_constraints.append(constraint)
                continue

            # Expand each qualifier in the set
            expanded_qualifiers = []
            for qualifier in qualifier_set:
                type_id = qualifier.get("qualifier_type_id")
                value = qualifier.get("qualifier_value")

                if not type_id or not value:
                    # Keep original if missing fields
                    expanded_qualifiers.append(qualifier)
                    continue

                # Get descendant values (includes original)
                descendant_values = self.get_value_descendants(value)

                # Create expanded qualifier with list of acceptable values
                expanded_qualifiers.append(
                    {
                        "qualifier_type_id": type_id,
                        "qualifier_values": descendant_values,  # plural - list of values
                    }
                )

            expanded_constraints.append({"qualifier_set": expanded_qualifiers})

        return expanded_constraints


class PredicateExpander:
    """
    Handles predicate expansion for symmetric and inverse predicates at query time.

    This class caches BMT lookups for performance and provides methods to determine
    what predicates should match when traversing edges in different directions.

    Predicate handling follows the reasoner-transpiler rules:
    1. If 'biolink:related_to' is queried, treat as "any predicate"
    2. For each predicate P:
       - Get inverse Q (if exists) → add to inverse predicates
       - If P is symmetric → add P to inverse predicates
    3. Expand both predicates and inverse predicates to descendants
    4. Filter descendants to only those that are canonical OR symmetric

    For a query predicate P:
    - Symmetric predicates: If P is symmetric, an edge A--P-->B also represents B--P-->A
    - Inverse predicates: If P has inverse Q, an edge A--P-->B is equivalent to B--Q-->A

    When traversing:
    - Forward (outgoing edges): Match predicate P directly
    - Backward (incoming edges): Match P if symmetric, or match inverse(P) if it exists
    """

    def __init__(self, bmt: Toolkit):
        self.bmt = bmt
        self._inverse_cache: dict[str, str | None] = {}
        self._symmetric_cache: dict[str, bool] = {}
        self._canonical_cache: dict[str, bool] = {}
        self._descendants_cache: dict[str, list[str]] = {}

    def is_symmetric(self, predicate: str) -> bool:
        """Check if a predicate is symmetric (cached)."""
        if predicate not in self._symmetric_cache:
            try:
                self._symmetric_cache[predicate] = self.bmt.is_symmetric(predicate)
            except Exception:
                self._symmetric_cache[predicate] = False
        return self._symmetric_cache[predicate]

    def is_canonical(self, predicate: str) -> bool:
        """
        Check if a predicate is canonical (cached).

        A predicate is canonical if it has the 'canonical_predicate' annotation
        set to True in the Biolink Model.
        """
        if predicate not in self._canonical_cache:
            try:
                element = self.bmt.get_element(predicate)
                if element is None:
                    self._canonical_cache[predicate] = False
                else:
                    # Check for canonical_predicate annotation
                    annotations = getattr(element, 'annotations', {}) or {}
                    self._canonical_cache[predicate] = bool(
                        annotations.get('canonical_predicate', False)
                    )
            except Exception:
                self._canonical_cache[predicate] = False
        return self._canonical_cache[predicate]

    def is_canonical_or_symmetric(self, predicate: str) -> bool:
        """Check if a predicate is either canonical or symmetric."""
        return self.is_canonical(predicate) or self.is_symmetric(predicate)

    def get_inverse(self, predicate: str) -> str | None:
        """Get the inverse of a predicate if one exists (cached)."""
        if predicate not in self._inverse_cache:
            try:
                if self.bmt.has_inverse(predicate):
                    inverse = self.bmt.get_inverse_predicate(predicate, formatted=True)
                    self._inverse_cache[predicate] = inverse
                else:
                    self._inverse_cache[predicate] = None
            except Exception:
                self._inverse_cache[predicate] = None
        return self._inverse_cache[predicate]

    def get_descendants(self, predicate: str) -> list[str]:
        """Get all descendants of a predicate (cached)."""
        if predicate not in self._descendants_cache:
            try:
                element = self.bmt.get_element(predicate)
                if element is None:
                    self._descendants_cache[predicate] = []
                else:
                    self._descendants_cache[predicate] = self.bmt.get_descendants(
                        predicate, formatted=True
                    )
            except Exception:
                self._descendants_cache[predicate] = []
        return self._descendants_cache[predicate]

    def get_filtered_descendants(self, predicate: str) -> list[str]:
        """
        Get descendants of a predicate, filtered to only canonical OR symmetric.

        This follows the reasoner-transpiler behavior where only predicates that
        are either marked as canonical_predicate or are symmetric are included
        in query expansion.
        """
        descendants = self.get_descendants(predicate)
        return [d for d in descendants if self.is_canonical_or_symmetric(d)]

    def expand_predicates(self, predicates: list[str]) -> tuple[list[str], list[str] | None]:
        """
        Expand predicates following reasoner-transpiler rules.

        This method:
        1. Handles 'biolink:related_to' as "any predicate" (returns empty lists)
        2. For each predicate, gets its inverse and adds to inverse list
        3. For symmetric predicates, adds them to the inverse list too
        4. Expands both lists to descendants
        5. Filters to only canonical OR symmetric predicates

        Args:
            predicates: List of predicate CURIEs from the query

        Returns:
            Tuple of (forward_predicates, inverse_predicates) where:
            - forward_predicates: Predicates to match in the forward direction
              (empty list means match all)
            - inverse_predicates: Predicates to match in the reverse direction
              (empty list means match all, None means don't check inverse)
        """
        # Handle 'related_to' or no predicates as "any predicate" in both directions
        if not predicates or 'biolink:related_to' in predicates:
            return [], []

        # Collect inverse predicates
        inverse_preds = []
        for pred in predicates:
            # Get explicit inverse
            inverse = self.get_inverse(pred)
            if inverse:
                inverse_preds.append(inverse)
            # Symmetric predicates are their own inverse for bidirectional matching
            if self.is_symmetric(pred):
                inverse_preds.append(pred)

        # Expand to descendants and filter to canonical/symmetric
        # Always include the original query predicates (they should always match)
        # Only filter descendants to canonical/symmetric
        forward_expanded = list(predicates)
        for pred in predicates:
            forward_expanded.extend(self.get_filtered_descendants(pred))

        # Always include the original inverse predicates
        inverse_expanded = list(inverse_preds)
        for pred in inverse_preds:
            inverse_expanded.extend(self.get_filtered_descendants(pred))

        # Deduplicate while preserving order
        forward_unique = list(dict.fromkeys(forward_expanded))
        inverse_unique = list(dict.fromkeys(inverse_expanded))

        # Return None for inverse when there are no inverse predicates to check,
        # to distinguish from the empty-list wildcard used by related_to
        return forward_unique, inverse_unique if inverse_unique else None

    def get_predicates_for_incoming_edges(self, predicates: list[str]) -> set[str]:
        """
        Get predicates that should match on incoming edges.

        When we're looking for edges with predicate P pointing TO a node,
        we should also consider:
        - P itself if it's stored in the incoming direction
        - The inverse of P, since an incoming edge with inverse(P) represents
          the same relationship as an outgoing edge with P

        Args:
            predicates: List of predicates we're searching for

        Returns:
            Set of predicates to match on incoming edges
        """
        result = set()
        for pred in predicates:
            # Always include the original predicate for direct matches
            result.add(pred)
            # If P has inverse Q, then an incoming edge with Q is equivalent
            # to an outgoing edge with P from the perspective of the target node
            inverse = self.get_inverse(pred)
            if inverse:
                result.add(inverse)
        return result

    def get_predicates_for_outgoing_edges(self, predicates: list[str]) -> set[str]:
        """
        Get predicates that should match on outgoing edges.

        When we're looking for edges with predicate P pointing FROM a node,
        we should also consider:
        - P itself for direct matches
        - The inverse of P when checking from the object's perspective

        Args:
            predicates: List of predicates we're searching for

        Returns:
            Set of predicates to match on outgoing edges
        """
        result = set()
        for pred in predicates:
            result.add(pred)
            # Include inverse for bidirectional matching
            inverse = self.get_inverse(pred)
            if inverse:
                result.add(inverse)
        return result

    def should_check_reverse_direction(self, predicate: str) -> bool:
        """
        Determine if we should also check the reverse direction for this predicate.

        Returns True if the predicate is symmetric or has an inverse defined.
        """
        return self.is_symmetric(predicate) or self.get_inverse(predicate) is not None

    def get_reverse_predicate(self, predicate: str) -> str | None:
        """
        Get the predicate to use when checking the reverse direction.

        For symmetric predicates, returns the same predicate.
        For predicates with inverses, returns the inverse.
        For other predicates, returns None.
        """
        if self.is_symmetric(predicate):
            return predicate
        return self.get_inverse(predicate)


def _edge_matches_qualifier_constraints(edge_qualifiers, qualifier_constraints):
    """
    Check if an edge's qualifiers match the query's qualifier constraints.

    Qualifier constraints use OR semantics between qualifier_sets and AND semantics
    within each qualifier_set. An edge matches if it satisfies at least one
    qualifier_set (i.e., has ALL qualifiers in that set).

    Supports two formats for constraint qualifiers:
    - Original: {"qualifier_type_id": "...", "qualifier_value": "..."} - exact match
    - Expanded: {"qualifier_type_id": "...", "qualifier_values": [...]} - match any value

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
    # Also build a dict mapping type_id -> set of values for expanded matching
    edge_qualifiers_by_type: dict[str, set[str]] = {}
    if edge_qualifiers:
        for q in edge_qualifiers:
            type_id = q.get("qualifier_type_id")
            value = q.get("qualifier_value")
            if type_id and value:
                edge_qualifier_set.add((type_id, value))
                if type_id not in edge_qualifiers_by_type:
                    edge_qualifiers_by_type[type_id] = set()
                edge_qualifiers_by_type[type_id].add(value)

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

            # Check for expanded format (qualifier_values - plural)
            req_values = required_qualifier.get("qualifier_values")
            if req_values is not None:
                # Expanded format: edge must have this type with ANY of the values
                edge_values = edge_qualifiers_by_type.get(req_type, set())
                if not edge_values.intersection(req_values):
                    all_match = False
                    break
            else:
                # Original format: exact match required
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

    for n1_idx, predicate, _, _ in graph.neighbors_with_properties(start_idx):
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
        for n2_idx, pred_12, _, _ in graph.neighbors_with_properties(n1_idx):
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


def _rewrite_for_subclass(query_graph, subclass_depth=1):
    """Rewrite query graph to add subclass expansion for pinned nodes.

    For each pinned node (one with ``ids``), creates a synthetic superclass
    node that holds the original IDs and a variable-depth ``subclass_of``
    edge connecting the original node to the superclass.  This allows the
    search to match both the exact node and any of its subclass descendants.

    Mirrors the reasoner-transpiler ``match_query`` rewriting logic.

    Args:
        query_graph: The query graph dict (mutated in-place).
        subclass_depth: Maximum number of ``subclass_of`` hops to traverse.
    """
    nodes = query_graph["nodes"]
    edges = query_graph["edges"]

    # Nodes already involved in explicit subclass_of / superclass_of edges
    # should not be rewritten (user specified them intentionally).
    excluded: set[str] = set()
    for edge in edges.values():
        preds = edge.get("predicates", [])
        if "biolink:subclass_of" in preds or "biolink:superclass_of" in preds:
            excluded.add(edge["subject"])
            excluded.add(edge["object"])

    pinned_qnodes = [
        qnode_id
        for qnode_id, qnode in list(nodes.items())
        if qnode.get("ids") and qnode_id not in excluded
    ]

    for qnode_id in pinned_qnodes:
        original = nodes[qnode_id]

        superclass_id = f"{qnode_id}_superclass"
        nodes[superclass_id] = {
            "ids": original.pop("ids"),
            "_superclass": True,
        }
        # Move categories to superclass node if present
        if "categories" in original:
            nodes[superclass_id]["categories"] = original.pop("categories")

        subclass_edge_id = f"{qnode_id}_subclass_edge"
        edges[subclass_edge_id] = {
            "subject": qnode_id,
            "object": superclass_id,
            "predicates": ["biolink:subclass_of"],
            "_subclass": True,
            "_subclass_depth": subclass_depth,
        }


def lookup(graph, query: dict, bmt=None, verbose=True, subclass=True, subclass_depth=1):
    """
    Take an arbitrary Translator query graph and return all matching paths.

    Args:
        graph: CSRGraph instance
        query: Full TRAPI request dict containing message.query_graph
        bmt: Biolink Model Toolkit instance (optional, will create if not provided)
        verbose: Print progress information
        subclass: If True, expand pinned nodes to include subclass descendants
        subclass_depth: Maximum number of subclass_of hops to traverse (default 1)

    Returns:
        TRAPI response dict with message containing results, knowledge_graph, etc.
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

    # Create predicate expander for handling symmetric/inverse predicates at query time
    predicate_expander = PredicateExpander(bmt)

    # Create qualifier expander for handling qualifier value hierarchy at query time
    qualifier_expander = QualifierExpander(bmt)

    original_query_graph = query["message"]["query_graph"]
    query_graph = copy.deepcopy(original_query_graph)
    subqgraph = copy.deepcopy(query_graph)

    # Rewrite query graph for subclass expansion if requested
    if subclass and subqgraph["edges"]:
        if verbose:
            print(f"Rewriting query graph for subclass expansion (depth={subclass_depth})")
        _rewrite_for_subclass(subqgraph, subclass_depth=subclass_depth)
        # Use the rewritten graph as the query graph for the rest of the pipeline
        query_graph = copy.deepcopy(subqgraph)

    # Store results for each edge query
    # edge_id -> list of (subject_idx, predicate, object_idx) tuples
    edge_results = {}

    # Store inverse predicates for each edge (needed for path reconstruction)
    # edge_id -> set of inverse predicates
    edge_inverse_preds = {}

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

        # Handle subclass edges with dedicated traversal
        if next_edge.get("_subclass"):
            subclass_edge_depth = next_edge.get("_subclass_depth", 1)
            edge_matches = _query_subclass_edge(
                graph, start_node_idxes, end_node_idxes, subclass_edge_depth, verbose
            )
            edge_inverse_preds[next_edge_id] = set()
        else:
            # Get allowed predicates using reasoner-transpiler rules:
            # 1. Handle 'related_to' as "any predicate"
            # 2. Expand to descendants filtered to canonical OR symmetric only
            # 3. Also expand inverse predicates for bidirectional matching
            query_predicates = next_edge.get("predicates", [])
            forward_predicates, inverse_predicates = predicate_expander.expand_predicates(
                query_predicates
            )

            # Forward predicates are used for direct edge matching
            # Inverse predicates are used for reverse direction matching
            # Keep them separate to avoid confusion in reverse_pred_map construction
            allowed_predicates = forward_predicates

            if verbose and query_predicates:
                print(f"  Query predicates: {query_predicates}")
                print(f"  Expanded to {len(forward_predicates)} forward, {len(inverse_predicates) if inverse_predicates is not None else 0} inverse predicates")

            # Store inverse predicates for this edge (for path reconstruction)
            edge_inverse_preds[next_edge_id] = set(inverse_predicates) if inverse_predicates is not None else set()

            # Get qualifier constraints for this edge and expand to include descendant values
            qualifier_constraints = next_edge.get("qualifier_constraints", [])
            if qualifier_constraints:
                qualifier_constraints = qualifier_expander.expand_qualifier_constraints(
                    qualifier_constraints
                )

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
                inverse_predicates=inverse_predicates,
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

        for subj_idx, pred, obj_idx, _via_inverse, _fwd_edge_idx in edge_matches:
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
        graph, query_graph, edge_results, original_edges, verbose,
        edge_inverse_preds=edge_inverse_preds
    )

    if verbose:
        print(f"Found {len(paths):,} complete paths")

    t_post_start = time.perf_counter()

    response = {
        "message": {
            "query_graph": original_query_graph,
            "knowledge_graph": {
                "nodes": {},
                "edges": {},
            },
            "results": [],
            "auxiliary_graphs": {},
        }
    }

    # Pre-compute subclass metadata for result building
    superclass_qnodes = {
        qnode_id for qnode_id, qnode in query_graph["nodes"].items()
        if qnode.get("_superclass")
    }
    subclass_qedges = {
        qedge_id for qedge_id, qedge in query_graph["edges"].items()
        if qedge.get("_subclass")
    }
    # Map: original qnode_id -> superclass qnode_id
    qnode_to_superclass = {}
    # Map: original qedge_id -> subclass qedge_id (for finding subclass edges attached to real edges)
    qedge_attached_subclass = defaultdict(list)
    for qedge_id, qedge in query_graph["edges"].items():
        if qedge.get("_subclass"):
            child_qnode = qedge["subject"]   # e.g. "n0"
            parent_qnode = qedge["object"]   # e.g. "n0_superclass"
            qnode_to_superclass[child_qnode] = parent_qnode
    # For each real (non-subclass) edge, find attached subclass edges
    for qedge_id, qedge in query_graph["edges"].items():
        if qedge.get("_subclass"):
            continue
        subj = qedge["subject"]
        obj = qedge["object"]
        if subj in qnode_to_superclass:
            qedge_attached_subclass[qedge_id].append(
                ("subject", f"{subj}_subclass_edge", qnode_to_superclass[subj])
            )
        if obj in qnode_to_superclass:
            qedge_attached_subclass[qedge_id].append(
                ("object", f"{obj}_subclass_edge", qnode_to_superclass[obj])
            )

    # Group paths by unique node binding combinations
    # Key: tuple of (qnode_id, node_id) pairs sorted by qnode_id
    # Value: list of edge dictionaries from paths with this node combination
    node_binding_groups = defaultdict(list)

    for path in paths:
        # Create a hashable key from node bindings — exclude superclass nodes
        node_key = tuple(
            sorted(
                (qnode_id, node["id"])
                for qnode_id, node in path["nodes"].items()
                if qnode_id not in superclass_qnodes
            )
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

            # Add node bindings — skip superclass nodes, substitute IDs
            for qnode_id, node in first_path["nodes"].items():
                if qnode_id in superclass_qnodes:
                    # Don't expose synthetic superclass nodes in bindings
                    continue

                # Add node to knowledge graph
                response["message"]["knowledge_graph"]["nodes"][node["id"]] = node

                # If this qnode has a superclass counterpart with a different result ID,
                # use the superclass ID in the binding (it's the originally queried entity)
                bound_id = node["id"]
                if qnode_id in qnode_to_superclass:
                    superclass_qnode = qnode_to_superclass[qnode_id]
                    superclass_node = first_path["nodes"].get(superclass_qnode)
                    if superclass_node and superclass_node["id"] != node["id"]:
                        bound_id = superclass_node["id"]
                        # Also add the superclass node to the knowledge graph
                        response["message"]["knowledge_graph"]["nodes"][superclass_node["id"]] = superclass_node

                result["node_bindings"][qnode_id] = [
                    {"id": bound_id, "attributes": []},
                ]

            # Aggregate edge bindings from all paths in group.
            # Edges sharing (subject, predicate, object) but with different
            # qualifiers or sources are distinct and must be kept.
            edge_bindings_by_qedge = defaultdict(list)
            edge_seen_keys = defaultdict(set)  # qedge_id -> set of hashable keys

            for path in grouped_paths:
                for edge_id, edge in path["edges"].items():
                    # Build a key that distinguishes edges with different
                    # qualifiers / sources even when (subj, pred, obj) match.
                    quals = edge.get("qualifiers") or []
                    quals_key = tuple(
                        sorted(
                            (q.get("qualifier_type_id", ""), q.get("qualifier_value", ""))
                            for q in quals
                        )
                    )
                    sources = edge.get("sources") or []
                    sources_key = tuple(
                        sorted(
                            (s.get("resource_id", ""), s.get("resource_role", ""))
                            for s in sources
                        )
                    )
                    edge_key = (
                        edge["subject"], edge["predicate"], edge["object"],
                        quals_key, sources_key,
                    )
                    if edge_key not in edge_seen_keys[edge_id]:
                        edge_seen_keys[edge_id].add(edge_key)
                        edge_bindings_by_qedge[edge_id].append(edge)

            # Add edges to knowledge graph and result bindings.
            # Use the original edge ID from the data when available so that
            # the same physical edge is stored once in the KG and referenced
            # by every result that uses it.
            for edge_id, edges in edge_bindings_by_qedge.items():
                # Skip subclass edges from direct bindings
                if edge_id in subclass_qedges:
                    continue

                result["analyses"][0]["edge_bindings"][edge_id] = []

                for edge in edges:
                    edge_kg_id = edge.pop("_edge_id", None) or str(uuid.uuid4())[:8]
                    response["message"]["knowledge_graph"]["edges"][edge_kg_id] = edge

                    # Check if this edge has attached subclass edges
                    attached = qedge_attached_subclass.get(edge_id, [])
                    if attached:
                        # Collect subclass edge IDs from the current path
                        subclass_edge_kg_ids = []
                        superclass_node_overrides = {}

                        for (which_end, sc_edge_id, sc_qnode_id) in attached:
                            sc_edges = edge_bindings_by_qedge.get(sc_edge_id, [])
                            for sc_edge in sc_edges:
                                # Skip self-loops (depth-0, no actual subclass hop)
                                if sc_edge["subject"] == sc_edge["object"]:
                                    continue
                                # Use .get() (not .pop()) because the same sc_edge
                                # dict may be referenced by multiple regular edges.
                                sc_kg_id = sc_edge.get("_edge_id") or str(uuid.uuid4())[:8]
                                response["message"]["knowledge_graph"]["edges"][sc_kg_id] = sc_edge
                                subclass_edge_kg_ids.append(sc_kg_id)

                            # Get the superclass node ID for endpoint override
                            for path in grouped_paths:
                                sc_node = path["nodes"].get(sc_qnode_id)
                                if sc_node:
                                    superclass_node_overrides[which_end] = sc_node["id"]
                                    break

                        if subclass_edge_kg_ids:
                            # Create composite inferred edge
                            composite_edge_ids = [edge_kg_id] + subclass_edge_kg_ids
                            composite_edge_id = "_".join(composite_edge_ids)
                            aux_graph_id = f"aux_{composite_edge_id}"

                            if aux_graph_id not in response["message"]["auxiliary_graphs"]:
                                response["message"]["auxiliary_graphs"][aux_graph_id] = {
                                    "edges": composite_edge_ids,
                                    "attributes": [],
                                }

                            if composite_edge_id not in response["message"]["knowledge_graph"]["edges"]:
                                inferred_edge = {
                                    "subject": superclass_node_overrides.get("subject", edge["subject"]),
                                    "predicate": edge["predicate"],
                                    "object": superclass_node_overrides.get("object", edge["object"]),
                                    "attributes": [
                                        {
                                            "attribute_type_id": "biolink:knowledge_level",
                                            "value": "logical_entailment",
                                        },
                                        {
                                            "attribute_type_id": "biolink:agent_type",
                                            "value": "automated_agent",
                                        },
                                        {
                                            "attribute_type_id": "biolink:support_graphs",
                                            "value": [aux_graph_id],
                                        },
                                    ],
                                    "sources": [
                                        {
                                            "resource_id": "infores:gandalf",
                                            "resource_role": "primary_knowledge_source",
                                        }
                                    ],
                                }
                                response["message"]["knowledge_graph"]["edges"][composite_edge_id] = inferred_edge

                            result["analyses"][0]["edge_bindings"][edge_id].append(
                                {"id": composite_edge_id, "attributes": []}
                            )
                        else:
                            # Subclass edges were all self-loops (depth-0), use normal binding
                            result["analyses"][0]["edge_bindings"][edge_id].append(
                                {"id": edge_kg_id, "attributes": []}
                            )
                    else:
                        result["analyses"][0]["edge_bindings"][edge_id].append(
                            {"id": edge_kg_id, "attributes": []}
                        )

            response["message"]["results"].append(result)
    finally:
        # Re-enable GC if it was enabled before
        if gc_was_enabled:
            gc.enable()

    # Strip internal _edge_id markers from KG edges so they don't leak
    # into the TRAPI response.
    for edge in response["message"]["knowledge_graph"]["edges"].values():
        edge.pop("_edge_id", None)

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


def _query_subclass_edge(graph, start_idxes, end_idxes, depth, verbose):
    """Traverse ``subclass_of`` edges to find subclass relationships.

    The synthetic subclass edge connects a child node (subject) to a
    superclass node (object).  The superclass node holds the original
    pinned IDs, so ``end_idxes`` will be pinned.

    We perform a BFS starting from each pinned end (superclass) node,
    following **incoming** ``subclass_of`` edges up to *depth* hops.
    Depth 0 means the node itself (identity — no hop needed).

    Args:
        graph: CSRGraph instance
        start_idxes: Indices for the child (subject) side, or None
        end_idxes: Indices for the superclass (object) side
        depth: Maximum subclass_of hops
        verbose: Print progress

    Returns:
        List of (child_idx, "biolink:subclass_of", parent_idx, False, fwd_edge_idx) tuples.
        The depth-0 self-match is included with fwd_edge_idx=-1 (no real edge).
    """
    matches = []

    # Resolve the subclass_of predicate index once
    subclass_pred = "biolink:subclass_of"

    if end_idxes is None:
        return matches

    for superclass_idx in end_idxes:
        # BFS: current frontier → next frontier, up to `depth` levels
        # Depth 0 = identity match (the node itself)
        frontier = {superclass_idx}
        visited = {superclass_idx}

        # Always include the depth-0 self-match (no real edge, sentinel -1)
        matches.append((superclass_idx, subclass_pred, superclass_idx, False, -1))

        for _hop in range(depth):
            next_frontier = set()
            for node_idx in frontier:
                # Walk incoming subclass_of edges: child --subclass_of--> node_idx
                for child_idx, predicate, _props, fwd_eidx in graph.incoming_neighbors_with_properties(node_idx):
                    if predicate != subclass_pred:
                        continue
                    if child_idx in visited:
                        continue
                    visited.add(child_idx)
                    next_frontier.add(child_idx)
                    matches.append((child_idx, subclass_pred, superclass_idx, False, fwd_eidx))
            frontier = next_frontier
            if not frontier:
                break

    if verbose:
        print(f"  Subclass traversal: found {len(matches)} matches (depth={depth})")

    return matches


def _query_edge(
    graph,
    start_idxes,
    end_idxes,
    start_categories,
    end_categories,
    allowed_predicates,
    qualifier_constraints,
    verbose,
    inverse_predicates: list[str] = None,
):
    """
    Query for a single edge with given constraints.

    Handles symmetric and inverse predicates at query time by checking both
    edge directions when appropriate. For example, if searching for predicate P
    and P has inverse Q, edges stored as B--Q-->A will be returned as A--P-->B.

    Args:
        graph: CSRGraph instance
        start_idxes: List of pinned start node indices, or None if unpinned
        end_idxes: List of pinned end node indices, or None if unpinned
        start_categories: List of allowed categories for start node
        end_categories: List of allowed categories for end node
        allowed_predicates: List of forward predicate strings (canonical/symmetric descendants)
        qualifier_constraints: List of qualifier constraint dicts from query
        verbose: Print progress information
        inverse_predicates: List of inverse predicate strings for reverse direction
            matching. None means don't check inverse direction. Empty list means
            match all predicates in inverse direction (wildcard).

    Returns:
        List of (subject_idx, predicate, object_idx, via_inverse, fwd_edge_idx) tuples where
        via_inverse indicates if the edge was found through inverse/symmetric lookup and
        fwd_edge_idx is the forward-CSR array position (unique per physical edge).
    """
    matches = []
    seen_edges = set()  # Track (subj, pred, obj, fwd_edge_idx) to avoid duplicates

    # Build set of inverse predicates for quick lookup.
    # None  -> don't check inverse direction at all (default)
    # []    -> match ALL predicates in inverse direction (wildcard, e.g. related_to)
    # [pred]-> match only the listed predicates in inverse direction
    check_inverse = inverse_predicates is not None
    inverse_pred_set = set(inverse_predicates) if inverse_predicates else set()

    def add_match(subj_idx, predicate, obj_idx, fwd_edge_idx, via_inverse=False):
        """Add a match, avoiding duplicates. Includes via_inverse flag.

        Dedup key includes ``fwd_edge_idx`` so that edges with the same
        (subj, pred, obj) but different qualifiers / sources are kept as
        separate matches.
        """
        key = (subj_idx, predicate, obj_idx, fwd_edge_idx)
        if key not in seen_edges:
            seen_edges.add(key)
            matches.append((subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx))

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

            # Check outgoing edges (direct matches)
            for obj_idx, predicate, props, fwd_edge_idx in graph.neighbors_with_properties(start_idx):
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

                add_match(start_idx, predicate, obj_idx, fwd_edge_idx)

            # Check incoming edges for symmetric/inverse predicates
            # An incoming edge with inverse(P) represents an outgoing edge with P
            if check_inverse:
                for other_idx, stored_pred, props, fwd_edge_idx in graph.incoming_neighbors_with_properties(start_idx):
                    node_neighbors += 1

                    # Check if stored predicate is one of our inverse predicates
                    if inverse_pred_set and stored_pred not in inverse_pred_set:
                        continue

                    # Check object categories (the "other" node becomes our object)
                    if end_categories:
                        obj_cats = graph.get_node_property(other_idx, "categories", [])
                        if not any(cat in obj_cats for cat in end_categories):
                            continue

                    # Check qualifier constraints
                    if qualifier_constraints:
                        edge_qualifiers = props.get("qualifiers", [])
                        if not _edge_matches_qualifier_constraints(
                            edge_qualifiers, qualifier_constraints
                        ):
                            continue

                    # Report the actual edge as stored in the graph
                    # The edge is: other_idx --[stored_pred]--> start_idx
                    # Mark as via_inverse since found through inverse lookup
                    add_match(other_idx, stored_pred, start_idx, fwd_edge_idx, via_inverse=True)

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

            # Check incoming edges (direct matches)
            for subj_idx, predicate, props, fwd_edge_idx in graph.incoming_neighbors_with_properties(
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

                add_match(subj_idx, predicate, end_idx, fwd_edge_idx)

            # Check outgoing edges for symmetric/inverse predicates
            # An outgoing edge with inverse(P) represents an incoming edge with P
            if check_inverse:
                for other_idx, stored_pred, props, fwd_edge_idx in graph.neighbors_with_properties(end_idx):
                    node_neighbors += 1

                    # Check if stored predicate is one of our inverse predicates
                    if inverse_pred_set and stored_pred not in inverse_pred_set:
                        continue

                    # Check subject categories (the "other" node becomes our subject)
                    if start_categories:
                        subj_cats = graph.get_node_property(other_idx, "categories", [])
                        if not any(cat in subj_cats for cat in start_categories):
                            continue

                    # Check qualifier constraints
                    if qualifier_constraints:
                        edge_qualifiers = props.get("qualifiers", [])
                        if not _edge_matches_qualifier_constraints(
                            edge_qualifiers, qualifier_constraints
                        ):
                            continue

                    # Report the actual edge as stored in the graph
                    # The edge is: end_idx --[stored_pred]--> other_idx
                    # Mark as via_inverse since found through inverse lookup
                    add_match(end_idx, stored_pred, other_idx, fwd_edge_idx, via_inverse=True)

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
        # obj_idx -> [(subj_idx, predicate, props, fwd_edge_idx), ...]
        forward_edges = defaultdict(list)

        t_neighbors_start = time.perf_counter()
        total_neighbors = 0
        for start_idx in start_idxes:
            for obj_idx, predicate, props, fwd_edge_idx in graph.neighbors_with_properties(start_idx):
                total_neighbors += 1
                if allowed_predicates and predicate not in allowed_predicates:
                    continue
                forward_edges[obj_idx].append((start_idx, predicate, props, fwd_edge_idx))

        # Also check reverse direction for symmetric/inverse predicates
        # Look for edges: end_node --inverse(P)--> start_node
        if check_inverse:
            start_set = set(start_idxes)
            for end_idx in end_idxes:
                for obj_idx, stored_pred, props, fwd_edge_idx in graph.neighbors_with_properties(end_idx):
                    total_neighbors += 1
                    # Only consider if obj_idx is one of our start nodes
                    if obj_idx not in start_set:
                        continue
                    # Check if stored predicate is one of our inverse predicates
                    if inverse_pred_set and stored_pred not in inverse_pred_set:
                        continue
                    # Check qualifier constraints before adding
                    if qualifier_constraints:
                        edge_qualifiers = props.get("qualifiers", [])
                        if not _edge_matches_qualifier_constraints(
                            edge_qualifiers, qualifier_constraints
                        ):
                            continue
                    # Report the actual edge as stored in the graph
                    # The edge is: end_idx --[stored_pred]--> obj_idx
                    # (where obj_idx is a start node)
                    # Mark as via_inverse since found through inverse lookup
                    add_match(end_idx, stored_pred, obj_idx, fwd_edge_idx, via_inverse=True)

        t_neighbors_end = time.perf_counter()
        if verbose:
            print(f"    Neighbor traversal: {t_neighbors_end - t_neighbors_start:.3f}s "
                  f"({total_neighbors:,} neighbors, {len(forward_edges):,} unique targets)")

        # Find intersection with end nodes
        t_intersect_start = time.perf_counter()
        end_set = set(end_idxes)

        for obj_idx in forward_edges.keys():
            if obj_idx in end_set:
                for subj_idx, predicate, props, fwd_edge_idx in forward_edges[obj_idx]:
                    # Check qualifier constraints
                    if qualifier_constraints:
                        edge_qualifiers = props.get("qualifiers", [])
                        if not _edge_matches_qualifier_constraints(
                            edge_qualifiers, qualifier_constraints
                        ):
                            continue
                    add_match(subj_idx, predicate, obj_idx, fwd_edge_idx)

        t1 = time.perf_counter()
        if verbose:
            print(f"    Intersection: {t1 - t_intersect_start:.3f}s")
            print(f"  Found {len(matches):,} matches in {t1 - t0:.3f}s")

    else:
        raise Exception("Both nodes unpinned - bad query planning")

    return matches


def _reconstruct_paths(graph, query_graph, edge_results, edge_order, verbose,
                       edge_inverse_preds=None):
    """
    Reconstruct complete paths by iteratively joining edge results.

    Uses NumPy arrays for efficient memory usage and to avoid GC pressure.
    For 15M+ paths, this eliminates Python object creation overhead.

    Args:
        graph: CSRGraph instance
        query_graph: Original query graph
        edge_results: Dict of edge_id -> [(subj_idx, pred, obj_idx, via_inverse, fwd_edge_idx), ...]
        edge_order: List of edge IDs in original query order
        verbose: Print progress
        edge_inverse_preds: (Deprecated, kept for compatibility) Dict of edge_id -> set of inverse predicates

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

    # Pre-allocate arrays for nodes, predicates, and via_inverse flags
    # nodes: shape (num_paths, max_nodes) - will grow columns as needed
    # preds: shape (num_paths, num_edges)
    # via_inverse: shape (num_paths, num_edges) - tracks if edge was found via inverse lookup
    max_nodes = len(query_graph["nodes"])
    num_edges = len(join_order)

    paths_nodes = np.zeros((num_paths, max_nodes), dtype=np.int32)
    paths_preds = np.zeros((num_paths, num_edges), dtype=np.int32)
    paths_via_inverse = np.zeros((num_paths, num_edges), dtype=np.bool_)
    paths_fwd_edge_idx = np.zeros((num_paths, num_edges), dtype=np.int32)

    # Fill in first edge data
    # For inverse matches, the actual edge has subject/object swapped relative to query
    for i, (subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx) in enumerate(first_results):
        if via_inverse:
            # Inverse match: actual edge is (subj, pred, obj) but query expects reversed
            # subj in actual edge corresponds to query's object
            # obj in actual edge corresponds to query's subject
            paths_nodes[i, 0] = obj_idx   # Query subject column gets actual object
            paths_nodes[i, 1] = subj_idx  # Query object column gets actual subject
        else:
            # Direct match: actual edge matches query direction
            paths_nodes[i, 0] = subj_idx
            paths_nodes[i, 1] = obj_idx
        paths_preds[i, 0] = get_pred_idx(predicate)
        paths_via_inverse[i, 0] = via_inverse
        paths_fwd_edge_idx[i, 0] = fwd_edge_idx

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

        # Normalize edge data to query-aligned direction
        # For inverse matches (via_inverse=True), the actual edge (subj, pred, obj)
        # represents the query direction (obj, pred, subj), so we swap
        # We keep the via_inverse flag and fwd_edge_idx to use during enrichment
        normalized_edge_data = []
        for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
            if via_inverse:
                # Inverse: swap to query-aligned direction
                normalized_edge_data.append((obj_idx, predicate, subj_idx, via_inverse, fwd_edge_idx))
            else:
                normalized_edge_data.append((subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx))
        edge_data = normalized_edge_data

        if subj_in_paths and obj_in_paths:
            # Both nodes already in path - validate consistency
            subj_col = qnode_to_col[subj_qnode]
            obj_col = qnode_to_col[obj_qnode]

            # Build index: (subj_idx, obj_idx) -> [(pred_idx, via_inverse, fwd_edge_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
                edge_index[(subj_idx, obj_idx)].append((get_pred_idx(predicate), via_inverse, fwd_edge_idx))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []
            new_via_inverse_list = []
            new_fwd_edge_idx_list = []

            for path_idx in range(len(paths_nodes)):
                key = (paths_nodes[path_idx, subj_col], paths_nodes[path_idx, obj_col])
                if key in edge_index:
                    for pred_idx, via_inverse, fwd_edge_idx in edge_index[key]:
                        new_nodes_list.append(paths_nodes[path_idx].copy())
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)
                        new_via_inv = paths_via_inverse[path_idx].copy()
                        new_via_inv[join_idx] = via_inverse
                        new_via_inverse_list.append(new_via_inv)
                        new_fwd_eidx = paths_fwd_edge_idx[path_idx].copy()
                        new_fwd_eidx[join_idx] = fwd_edge_idx
                        new_fwd_edge_idx_list.append(new_fwd_eidx)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
                paths_via_inverse = np.array(new_via_inverse_list, dtype=np.bool_)
                paths_fwd_edge_idx = np.array(new_fwd_edge_idx_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)
                paths_via_inverse = np.zeros((0, num_edges), dtype=np.bool_)
                paths_fwd_edge_idx = np.zeros((0, num_edges), dtype=np.int32)

        elif subj_in_paths:
            # Join on subject node, add object node
            subj_col = qnode_to_col[subj_qnode]

            # Assign column for new object node
            if obj_qnode not in qnode_to_col:
                qnode_to_col[obj_qnode] = num_node_cols
                num_node_cols += 1
            obj_col = qnode_to_col[obj_qnode]

            # Build index: subj_idx -> [(pred_idx, obj_idx, via_inverse, fwd_edge_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
                edge_index[subj_idx].append((get_pred_idx(predicate), obj_idx, via_inverse, fwd_edge_idx))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []
            new_via_inverse_list = []
            new_fwd_edge_idx_list = []

            for path_idx in range(len(paths_nodes)):
                subj_idx = paths_nodes[path_idx, subj_col]
                if subj_idx in edge_index:
                    for pred_idx, obj_idx, via_inverse, fwd_edge_idx in edge_index[subj_idx]:
                        new_nodes = paths_nodes[path_idx].copy()
                        new_nodes[obj_col] = obj_idx
                        new_nodes_list.append(new_nodes)
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)
                        new_via_inv = paths_via_inverse[path_idx].copy()
                        new_via_inv[join_idx] = via_inverse
                        new_via_inverse_list.append(new_via_inv)
                        new_fwd_eidx = paths_fwd_edge_idx[path_idx].copy()
                        new_fwd_eidx[join_idx] = fwd_edge_idx
                        new_fwd_edge_idx_list.append(new_fwd_eidx)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
                paths_via_inverse = np.array(new_via_inverse_list, dtype=np.bool_)
                paths_fwd_edge_idx = np.array(new_fwd_edge_idx_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)
                paths_via_inverse = np.zeros((0, num_edges), dtype=np.bool_)
                paths_fwd_edge_idx = np.zeros((0, num_edges), dtype=np.int32)

        elif obj_in_paths:
            # Join on object node, add subject node
            obj_col = qnode_to_col[obj_qnode]

            # Assign column for new subject node
            if subj_qnode not in qnode_to_col:
                qnode_to_col[subj_qnode] = num_node_cols
                num_node_cols += 1
            subj_col = qnode_to_col[subj_qnode]

            # Build index: obj_idx -> [(subj_idx, pred_idx, via_inverse, fwd_edge_idx), ...]
            edge_index = defaultdict(list)
            for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
                edge_index[obj_idx].append((subj_idx, get_pred_idx(predicate), via_inverse, fwd_edge_idx))

            # Find matching paths
            new_nodes_list = []
            new_preds_list = []
            new_via_inverse_list = []
            new_fwd_edge_idx_list = []

            for path_idx in range(len(paths_nodes)):
                obj_idx = paths_nodes[path_idx, obj_col]
                if obj_idx in edge_index:
                    for subj_idx, pred_idx, via_inverse, fwd_edge_idx in edge_index[obj_idx]:
                        new_nodes = paths_nodes[path_idx].copy()
                        new_nodes[subj_col] = subj_idx
                        new_nodes_list.append(new_nodes)
                        new_preds = paths_preds[path_idx].copy()
                        new_preds[join_idx] = pred_idx
                        new_preds_list.append(new_preds)
                        new_via_inv = paths_via_inverse[path_idx].copy()
                        new_via_inv[join_idx] = via_inverse
                        new_via_inverse_list.append(new_via_inv)
                        new_fwd_eidx = paths_fwd_edge_idx[path_idx].copy()
                        new_fwd_eidx[join_idx] = fwd_edge_idx
                        new_fwd_edge_idx_list.append(new_fwd_eidx)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
                paths_via_inverse = np.array(new_via_inverse_list, dtype=np.bool_)
                paths_fwd_edge_idx = np.array(new_fwd_edge_idx_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)
                paths_via_inverse = np.zeros((0, num_edges), dtype=np.bool_)
                paths_fwd_edge_idx = np.zeros((0, num_edges), dtype=np.int32)

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
            new_via_inverse_list = []
            new_fwd_edge_idx_list = []

            for path_idx in range(len(paths_nodes)):
                for subj_idx, predicate, obj_idx, via_inverse, fwd_edge_idx in edge_data:
                    new_nodes = paths_nodes[path_idx].copy()
                    new_nodes[subj_col] = subj_idx
                    new_nodes[obj_col] = obj_idx
                    new_nodes_list.append(new_nodes)
                    new_preds = paths_preds[path_idx].copy()
                    new_preds[join_idx] = get_pred_idx(predicate)
                    new_preds_list.append(new_preds)
                    new_via_inv = paths_via_inverse[path_idx].copy()
                    new_via_inv[join_idx] = via_inverse
                    new_via_inverse_list.append(new_via_inv)
                    new_fwd_eidx = paths_fwd_edge_idx[path_idx].copy()
                    new_fwd_eidx[join_idx] = fwd_edge_idx
                    new_fwd_edge_idx_list.append(new_fwd_eidx)

            if new_nodes_list:
                paths_nodes = np.array(new_nodes_list, dtype=np.int32)
                paths_preds = np.array(new_preds_list, dtype=np.int32)
                paths_via_inverse = np.array(new_via_inverse_list, dtype=np.bool_)
                paths_fwd_edge_idx = np.array(new_fwd_edge_idx_list, dtype=np.int32)
            else:
                paths_nodes = np.zeros((0, max_nodes), dtype=np.int32)
                paths_preds = np.zeros((0, num_edges), dtype=np.int32)
                paths_via_inverse = np.zeros((0, num_edges), dtype=np.bool_)
                paths_fwd_edge_idx = np.zeros((0, num_edges), dtype=np.int32)

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

    # Check if we exceed the large result threshold — if so, skip expensive
    # per-edge property lookups (edge attributes, sources, qualifiers, LMDB
    # publications/attributes) and only include predicates.
    lightweight = num_paths > LARGE_RESULT_PATH_THRESHOLD

    # Build node property cache
    if verbose:
        if lightweight:
            print(
                f"  Enriching {num_paths:,} paths (lightweight mode: "
                f">{LARGE_RESULT_PATH_THRESHOLD:,} paths, skipping edge attributes)..."
            )
        else:
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

                # Get query-aligned node indices
                query_subj_idx = paths_nodes[path_idx, subj_col]
                query_obj_idx = paths_nodes[path_idx, obj_col]

                # Check if this edge was found via inverse lookup
                # If so, the actual edge in the graph has swapped direction
                is_inverse = paths_via_inverse[path_idx, col]

                if is_inverse:
                    # Actual edge: query_obj -> query_subj
                    actual_subj_idx = query_obj_idx
                    actual_obj_idx = query_subj_idx
                else:
                    # Actual edge: query_subj -> query_obj
                    actual_subj_idx = query_subj_idx
                    actual_obj_idx = query_obj_idx

                fwd_eidx = int(paths_fwd_edge_idx[path_idx, col])

                if lightweight:
                    # Only include predicate and structural info — skip
                    # edge attribute lookups (sources, qualifiers, LMDB data)
                    edge_props = {
                        "predicate": predicate,
                        "subject": node_cache[actual_subj_idx]["id"],
                        "object": node_cache[actual_obj_idx]["id"],
                    }
                else:
                    # O(1) property lookup using forward edge index
                    if fwd_eidx < 0:
                        # Synthetic edge (e.g. subclass self-match) with no
                        # real CSR position — build minimal props.
                        edge_props = {}
                    else:
                        edge_props = graph.get_edge_properties_by_index(fwd_eidx).copy()

                    # Ensure required fields are present with actual edge direction
                    edge_props["predicate"] = predicate
                    edge_props["subject"] = node_cache[actual_subj_idx]["id"]
                    edge_props["object"] = node_cache[actual_obj_idx]["id"]

                # Attach the original edge ID (if available) for use
                # during knowledge graph construction.
                if fwd_eidx >= 0:
                    orig_id = graph.get_edge_id(fwd_eidx)
                    if orig_id is not None:
                        edge_props["_edge_id"] = orig_id

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
