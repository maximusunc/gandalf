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

**Root cause**: Gandalf's underlying graph data does not contain the text-mining-sourced
`treats` edges connecting these chemicals to MONDO:0007186 (Heartburn). The loader
(`gandalf/loader.py`) loads ALL edges from input KGX files without filtering, so these edges
were never present in the input data provided to Gandalf.

### 2. Missing intermediary biological entities (`i` nodes)

Gandalf finds 6 unique intermediary genes; Automat finds 23.

**Gandalf's `i` genes**: TNF, BCL2, CCL2, VEGFA, VEGFC, TRPV1

**Missing `i` genes** (18): MMP3, MMP8, MMP9, CASP3, BAX, STAT3, MAPT, PECAM1, PDCD1,
SPATA2, RPL17, PLA2G1B, CXCL1, VEGFB, VEGFD, CD80, PCNA, SETD2

Even for the shared `e` chemical macrogol, Gandalf is missing paths through many of these
intermediary genes because it lacks the corresponding `affects` edges.

**Root cause**: Same as above — the `affects` edges connecting these genes to the relevant
chemicals are not in Gandalf's input graph data.

### 3. Missing `SN` chemicals (target nodes)

Gandalf resolves 5 SN chemicals; Automat resolves 12.

**Missing SN chemicals** (7): Peracetic acid, solution, Oxygen, elemental oxygen,
metal cation, medroxyprogesterone, Gemcitabine

Note that some missing SN chemicals (medroxyprogesterone, Gemcitabine, solution) also
appear as `e` chemicals in Automat but not Gandalf, indicating these chemicals participate
in multiple roles across the query pattern.

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

## Codebase Analysis: Is this a bug or a data issue?

### Query resolution pipeline is correct

After reviewing the query-time code in detail:

- **Predicate expansion** (`PredicateExpander`, `search.py:145-300`): Correctly expands
  `biolink:affects` to descendants filtered to canonical/symmetric. The `affects` predicate
  itself is always included.

- **Qualifier matching** (`_edge_matches_qualifier_constraints`, `search.py:374-444`):
  Correctly implements OR between qualifier_sets and AND within each set. Qualifier value
  expansion via BMT works correctly.

- **Edge querying** (`_query_edge`, `search.py:1462-1757`): All three cases (start pinned,
  end pinned, both pinned) correctly handle both forward and inverse predicate directions.

- **Path reconstruction** (`_reconstruct_paths`, `search.py:1760-2215`): Join order
  optimization and path assembly are sound. No path truncation or result limits that would
  drop valid paths.

- **Data loading** (`loader.py`): Three-pass streaming architecture loads ALL edges from
  input KGX files. No filtering by knowledge source, predicate, or edge properties.

### Conclusion: Data issue, not a code issue

The missing paths are caused by **differences in input data**, not by bugs in Gandalf's
query resolution logic. Specifically:

1. Gandalf's KGX input does not include text-mining-provider-cooccurrence `treats` edges
   for medroxyprogesterone, Gemcitabine, and solution → Heartburn
2. Gandalf's KGX input is missing many `affects` edges between intermediary genes and
   chemicals that Automat has

To resolve the gap, Gandalf's input data pipeline would need to include the text-mining
knowledge source edges that Automat serves.
