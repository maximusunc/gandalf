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

## Root Cause: Exact category matching misses nodes with specific categories

### The Bug

`_query_edge()` in `gandalf/search.py` uses **exact string matching** when checking
whether a node's categories satisfy the query's category constraints:

```python
if not any(cat in subj_cats for cat in start_categories):
    continue
```

TRAPI queries typically use broad Biolink Model ancestor categories like
`biolink:ChemicalEntity` or `biolink:BiologicalEntity`. However, node categories in
KGX data use specific descendant types:

| Node | Categories in KGX data | Query category |
|------|----------------------|----------------|
| CHEBI:6716 (medroxyprogesterone) | `["biolink:SmallMolecule"]` | `biolink:ChemicalEntity` |
| NCBIGene:4312 (MMP1) | `["biolink:Gene", "biolink:GeneOrGeneProduct"]` | `biolink:BiologicalEntity` |

Since `"biolink:ChemicalEntity"` is not literally present in `["biolink:SmallMolecule"]`,
the exact-match check fails and the edge is filtered out — even though SmallMolecule **is**
a descendant of ChemicalEntity in the Biolink hierarchy.

### How This Causes Missing Paths

1. **edge_0** (`treats`): Text-mining `treats` edges between chemicals and Heartburn
   are filtered out because the chemical nodes (e.g., medroxyprogesterone, Gemcitabine,
   solution) have specific categories like `SmallMolecule` that don't match the query's
   `ChemicalEntity`.

2. **edge_1 and edge_2** (`affects`): Similarly, intermediary biological entities have
   categories like `Gene` or `GeneOrGeneProduct` that don't match the query's
   `BiologicalEntity`.

3. Without the category match, edges are silently skipped. This means chemicals that
   could form complete paths through intermediary genes are never considered, resulting
   in 27 of 36 expected paths being missing.

### The Fix

Added a `CategoryExpander` class to `gandalf/search.py` (following the same pattern as
the existing `QualifierExpander` and `PredicateExpander`). The expander uses BMT's
`get_descendants()` to expand each query category to include all of its Biolink Model
descendants before the category check in `_query_edge()`.

For example, expanding `biolink:ChemicalEntity` produces a set that includes
`biolink:SmallMolecule`, `biolink:Drug`, `biolink:MolecularMixture`, etc. The expanded
set is passed to `_query_edge()`, so the existing exact-match logic now correctly matches
nodes with specific categories.

The expansion is:
- Performed once per unique category (results are cached)
- Applied at the `lookup()` call site before passing categories to `_query_edge()`
- Consistent with how Automat and other TRAPI services handle category matching

### Files Changed

- `gandalf/search.py`: Added `CategoryExpander` class; integrated into `lookup()`
- `tests/test_search.py`: Added `TestCategoryExpansion` class with unit and integration tests

## Codebase Analysis: Query Resolution Pipeline

The rest of the query-time code was reviewed in detail and is correct:

- **Predicate expansion** (`PredicateExpander`, `search.py:187-345`): Correctly expands
  `biolink:affects` to descendants filtered to canonical/symmetric.

- **Qualifier matching** (`_edge_matches_qualifier_constraints`, `search.py:416-486`):
  Correctly implements OR between qualifier_sets and AND within each set.

- **Qualifier expansion** (`QualifierExpander`, `search.py:22-141`): Correctly expands
  qualifier values (e.g., `decreased` matches `downregulated`) using BMT hierarchy.

- **Edge querying** (`_query_edge`, `search.py:1504-1799`): All three cases correctly
  handle both forward and inverse predicate directions. Category matching now works
  correctly with expanded categories.

- **Path reconstruction** (`_reconstruct_paths`, `search.py:1802-2257`): Join order
  optimization and path assembly are sound. No path truncation.

- **Data loading** (`loader.py`): Loads ALL edges. No filtering by knowledge source or
  predicate. Loading is correct; the issue was query-time category matching.
