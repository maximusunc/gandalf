# Response Path Analysis: Gandalf vs Automat

## Overview

This analysis compares TRAPI responses from Gandalf and Automat (ground truth) for the
same query graph to identify what kinds of paths Gandalf is missing.

## Query Graph Structure

4 nodes, 3 edges:

```
SN (ChemicalEntity)  <--affects[decreased activity_or_abundance]--  i (BiologicalEntity)
                                                                     |
e  (ChemicalEntity)  <--affects[decreased activity_or_abundance]-----+
       |
       +---treats--->  ON (DiseaseOrPhenotypicFeature, MONDO:0007186 Heartburn)
```

- **edge_0**: `e` --treats--> `ON` (Heartburn)
- **edge_1**: `i` --affects--> `SN` (qualifier: decreased activity_or_abundance)
- **edge_2**: `i` --affects--> `e` (qualifier: decreased activity_or_abundance)

Each result is uniquely identified by the tuple `(SN, e, i)` with `ON` fixed.

## Result Counts

| Metric | Gandalf | Automat |
|--------|---------|---------|
| Total paths | 9 | 36 |
| Unique `e` chemicals (treats Heartburn) | 2 | 4 |
| Unique `i` genes (intermediary) | 6 | 23 |
| Unique `SN` chemicals (target) | 5 | 12 |

## Categories of Missing Paths

### 1. Missing `treats` edges (edge_0) — largest source of missing paths

Gandalf only finds 2 chemicals that treat Heartburn, both from **CTD**:
- Serotonin (CHEBI:28790)
- macrogol (CHEBI:30742)

Automat finds 4 chemicals, all from **text-mining-provider-cooccurrence**:
- macrogol (CHEBI:30742) — shared with Gandalf
- medroxyprogesterone (CHEBI:6716)
- Gemcitabine (CHEBI:175901)
- solution (CHEBI:75958)

**Impact**: The 3 missing `e` chemicals account for ~23 of the 27 missing paths. Every path
through medroxyprogesterone (5 paths), Gemcitabine (13 paths), and solution (5 paths) is
entirely absent from Gandalf's results.

### 2. Missing intermediary biological entities (`i` nodes)

Gandalf finds 6 unique intermediary genes; Automat finds 23.

**Gandalf's `i` genes**: TNF, BCL2, CCL2, VEGFA, VEGFC, TRPV1

**Missing `i` genes** (18): MMP3, MMP8, MMP9, CASP3, BAX, STAT3, MAPT, PECAM1, PDCD1,
SPATA2, RPL17, PLA2G1B, CXCL1, VEGFB, VEGFD, CD80, PCNA, SETD2

Even for the shared `e` chemical macrogol, Gandalf is missing paths through many of these
intermediary genes because it lacks the corresponding `affects` edges.

### 3. Missing `SN` chemicals (target nodes)

Gandalf resolves 5 SN chemicals; Automat resolves 12.

**Missing SN chemicals** (7): Peracetic acid, solution, Oxygen, elemental oxygen,
metal cation, medroxyprogesterone, Gemcitabine

### 4. Gandalf has one unique path not in Automat

Gandalf finds `Dextromethorphan → Serotonin → TNF` via CTD-sourced edges. This path does
not appear in Automat's results, indicating Gandalf has some CTD data that Automat lacks.

## Source and Qualifier Differences

| Aspect | Gandalf | Automat |
|--------|---------|---------|
| `treats` edge sources | CTD | text-mining-provider-cooccurrence |
| `affects` edge sources | text-mining + signor | text-mining only |
| Qualifier values | `decreased` + `downregulated` (via signor) | `decreased` only |

Gandalf's qualifier expansion (via `QualifierExpander` in `gandalf/search.py:22-141`)
correctly expands `decreased` to match `downregulated` using BMT hierarchy. This is
working as intended and allows Gandalf to find paths through signor-sourced edges that
use `downregulated` as the qualifier value.

## Root Cause: Nested qualifier format silently dropped during loading

### The Bug

Commit `5a97186` ("Fix qualifier loading based on real edges") removed support for the
**nested `qualifiers` array format** from `_extract_qualifiers()` in `gandalf/loader.py`.

The original code handled two KGX qualifier formats:
- **Format A** (top-level fields): `"object_aspect_qualifier": "activity"`
- **Format B** (nested array): `"qualifiers": [{"qualifier_type_id": "...", "qualifier_value": "..."}]`

After the commit, only Format A is handled. Edges using Format B have their qualifiers
**silently dropped** — the loader sees the `qualifiers` field but explicitly skips it
(in `_extract_attributes`, `field == "qualifiers" → continue`), and `_extract_qualifiers`
no longer reads from it.

### How This Causes Missing Paths

1. **edge_0** (`treats`, no qualifier constraints): Text-mining `treats` edges are found
   regardless of qualifier format, because this edge has no qualifier constraints.

2. **edge_1 and edge_2** (`affects`, qualifier constraints: `decreased activity_or_abundance`):
   Text-mining `affects` edges using the nested qualifier format are stored with **empty
   qualifier lists**. When the query applies qualifier constraints, these edges fail the
   qualifier check in `_edge_matches_qualifier_constraints()` and are filtered out.

3. Without matching `affects` edges for the intermediary genes, no complete paths can be
   formed through those chemicals, even though the `treats` edges exist.

### The Fix

Restored nested qualifier format support in `_extract_qualifiers()`. The function now:
1. Checks for top-level qualifier fields first (existing behavior)
2. If none found, checks for a nested `qualifiers` array
3. Extracts qualifier dicts from the nested array
4. Returns a flat list of qualifier dicts (consistent with current storage/matching format)

Top-level fields take priority when both formats are present, preserving backward
compatibility.

### Files Changed

- `gandalf/loader.py`: Restored Format B support in `_extract_qualifiers()`
- `tests/test_loader.py`: Added `TestExtractQualifiers` class with tests for both formats

## Codebase Analysis: Query Resolution Pipeline

The query-time code was reviewed in detail and is correct:

- **Predicate expansion** (`PredicateExpander`, `search.py:145-300`): Correctly expands
  `biolink:affects` to descendants filtered to canonical/symmetric.

- **Qualifier matching** (`_edge_matches_qualifier_constraints`, `search.py:374-444`):
  Correctly implements OR between qualifier_sets and AND within each set.

- **Edge querying** (`_query_edge`, `search.py:1462-1757`): All three cases correctly
  handle both forward and inverse predicate directions.

- **Path reconstruction** (`_reconstruct_paths`, `search.py:1760-2215`): Join order
  optimization and path assembly are sound. No path truncation.

- **Data loading** (`loader.py`): Loads ALL edges. No filtering by knowledge source or
  predicate. The only issue was the qualifier extraction format gap (now fixed).
