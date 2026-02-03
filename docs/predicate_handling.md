# Predicate Handling in Gandalf

This document describes how predicates are processed and matched during query execution in Gandalf's search system.

## Overview

Gandalf handles predicates at **query time** rather than load time. This means:
- The graph stores edges exactly as they appear in the source data (single direction)
- Symmetric and inverse predicate relationships are resolved when queries are executed
- This reduces graph size and allows predicate handling logic to be updated without rebuilding the graph

## Predicate Processing Pipeline

### 1. Descendant Expansion

**Location**: `search.py` in `lookup()` function

**Rule**: For each predicate specified in the query, expand it to include all BMT descendants.

```python
allowed_predicates = [
    predicate
    for edge_predicate in next_edge["predicates"]
    if bmt.get_element(edge_predicate) is not None
    for predicate in bmt.get_descendants(edge_predicate, formatted=True)
]
```

**Example**:

| Query Predicate | Expanded To Include |
|-----------------|---------------------|
| `biolink:related_to` | All predicates (root of hierarchy) |
| `biolink:affects` | `affects`, `regulates`, `disrupts`, `positively_regulates`, `negatively_regulates`, etc. |
| `biolink:treats` | `treats` (plus any descendants) |

### 2. Reverse Predicate Map Construction

**Location**: `search.py` in `_query_edge()` function

**Rule**: Build a mapping from predicates that might be found in the reverse direction to the predicate that should be reported.

```python
reverse_pred_map = {}
for pred in allowed_predicates:
    inverse = predicate_expander.get_inverse(pred)
    if inverse:
        reverse_pred_map[inverse] = pred
    if predicate_expander.is_symmetric(pred):
        reverse_pred_map[pred] = pred
```

**Example**:

| Stored Predicate | Maps To | Reason |
|------------------|---------|--------|
| `biolink:treated_by` | `biolink:treats` | Inverse relationship |
| `biolink:interacts_with` | `biolink:interacts_with` | Symmetric predicate |
| `biolink:expressed_in` | `biolink:expresses` | Inverse relationship |

### 3. Direct Predicate Matching

**Rule**: For edges traversed in their natural direction, check if the stored predicate is in the allowed predicates list.

```python
if allowed_predicates and predicate not in allowed_predicates:
    continue
```

| Traversal Direction | Edge Being Checked | Predicate Requirement |
|---------------------|--------------------|-----------------------|
| Forward (outgoing) | `start --P--> obj` | `P in allowed_predicates` |
| Backward (incoming) | `subj --P--> end` | `P in allowed_predicates` |

### 4. Symmetric Predicate Handling

**Definition**: A symmetric predicate has the same meaning in both directions. If `A --P--> B` exists and P is symmetric, then `B --P--> A` is also semantically true.

**Rule**: When a predicate is symmetric, also check the reverse direction using the same predicate.

**Implementation**: Add `reverse_pred_map[P] = P` for symmetric predicates, then check incoming edges during forward search (and vice versa).

**Examples of Symmetric Predicates**:
- `biolink:interacts_with`
- `biolink:genetically_interacts_with`
- `biolink:physically_interacts_with`
- `biolink:correlated_with`

**Matching Behavior**:

| Query Predicate | Stored Edge | Found Via | Reported As |
|-----------------|-------------|-----------|-------------|
| `interacts_with` | `A --interacts_with--> B` | Direct (outgoing from A) | `A --interacts_with--> B` |
| `interacts_with` | `B --interacts_with--> A` | Reverse (incoming to A) | `A --interacts_with--> B` |

### 5. Inverse Predicate Handling

**Definition**: An inverse predicate expresses the same relationship from the opposite perspective. If `A --P--> B` and P has inverse Q, then `B --Q--> A` represents the same fact.

**Rule**: When checking the reverse direction, look for the inverse predicate and report the match using the originally queried predicate.

**Implementation**: Add `reverse_pred_map[inverse(P)] = P`, then check incoming edges during forward search for the inverse predicate.

**Examples of Inverse Predicate Pairs**:

| Predicate | Inverse |
|-----------|---------|
| `biolink:treats` | `biolink:treated_by` |
| `biolink:expresses` | `biolink:expressed_in` |
| `biolink:has_part` | `biolink:part_of` |
| `biolink:has_participant` | `biolink:participates_in` |
| `biolink:causes` | `biolink:caused_by` |
| `biolink:produces` | `biolink:produced_by` |
| `biolink:enables` | `biolink:enabled_by` |

**Matching Behavior**:

| Query Predicate | Stored Edge | Found Via | Reported As |
|-----------------|-------------|-----------|-------------|
| `treats` | `Drug --treats--> Disease` | Direct | `Drug --treats--> Disease` |
| `treats` | `Disease --treated_by--> Drug` | Reverse (incoming to Drug) | `Drug --treats--> Disease` |

## Search Cases

### Case 1: Start Pinned, End Unpinned (Forward Search)

```
For each start_node:
    1. Check OUTGOING edges: start --P--> obj
       - If P in allowed_predicates:
         - Check category constraints on obj
         - Check qualifier constraints
         - Add match: (start, P, obj)

    2. Check INCOMING edges: other --Q--> start
       - If Q in reverse_pred_map:
         - original_pred = reverse_pred_map[Q]
         - Check category constraints on other
         - Check qualifier constraints
         - Add match: (start, original_pred, other)
```

### Case 2: Start Unpinned, End Pinned (Backward Search)

```
For each end_node:
    1. Check INCOMING edges: subj --P--> end
       - If P in allowed_predicates:
         - Check category constraints on subj
         - Check qualifier constraints
         - Add match: (subj, P, end)

    2. Check OUTGOING edges: end --Q--> other
       - If Q in reverse_pred_map:
         - original_pred = reverse_pred_map[Q]
         - Check category constraints on other
         - Check qualifier constraints
         - Add match: (other, original_pred, end)
```

### Case 3: Both Ends Pinned

```
1. Check OUTGOING edges from start nodes:
   - For each start --P--> obj where P in allowed_predicates:
     - Store in forward_edges[obj]

2. Check reverse direction (OUTGOING from end nodes):
   - For each end --Q--> obj where Q in reverse_pred_map and obj in start_nodes:
     - original_pred = reverse_pred_map[Q]
     - Store in forward_edges[end] as (obj, original_pred, props)

3. Find intersection:
   - For each obj in forward_edges that is also in end_nodes:
     - Check qualifier constraints
     - Add matches
```

## PredicateExpander Class

The `PredicateExpander` class provides cached access to BMT predicate information:

```python
class PredicateExpander:
    def is_symmetric(predicate: str) -> bool
        """Check if predicate is symmetric (cached)."""

    def get_inverse(predicate: str) -> str | None
        """Get inverse predicate if one exists (cached)."""

    def should_check_reverse_direction(predicate: str) -> bool
        """True if predicate is symmetric or has an inverse."""

    def get_reverse_predicate(predicate: str) -> str | None
        """Get predicate to use when checking reverse direction."""
```

## Current Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Descendant expansion | Supported | Via `bmt.get_descendants()` |
| Direct predicate matching | Supported | Standard edge filtering |
| Symmetric predicates | Supported | Via reverse direction check |
| Inverse predicates | Supported | Via `reverse_pred_map` |
| Inverse of descendants | Not supported | If querying `treats` and `kills` is a descendant, we don't automatically check for `killed_by` |
| Canonical normalization | Not supported | Could normalize all results to canonical predicate direction |

## Performance Considerations

1. **Caching**: The `PredicateExpander` caches all BMT lookups to avoid repeated calls
2. **Deduplication**: A `seen_edges` set prevents duplicate matches when an edge is found via both direct and reverse lookups
3. **No graph duplication**: Unlike load-time expansion, query-time handling doesn't increase graph size
4. **BMT initialization**: BMT is initialized once per query session and reused

## References

- [BMT Documentation](https://biolink.github.io/biolink-model-toolkit/)
- [Biolink Model - Inverse Predicates](https://github.com/biolink/biolink-model/issues/57)
- [Best Practices for Inverse Predicates](https://github.com/biolink/biolink-model/issues/440)
