# Rust Conversion Plan for GANDALF

## Executive Summary

**Recommendation: Hybrid approach (PyO3), not a full rewrite.**

A full Rust rewrite is not worth the cost right now. GANDALF is ~6K lines of
core Python in alpha (v0.1.8), still evolving rapidly, with a Python-only
dependency (BMT) that would be expensive to replace. However, the codebase has
real performance bottlenecks that Rust can address surgically via PyO3/maturin,
keeping the Python API and ecosystem intact.

---

## Analysis: Where Python Hurts and Where It Doesn't

### Python is genuinely slow here (Rust would help)

1. **Inner traversal loops** (`query_edge.py:170-224`, `query_edge.py:262-313`):
   Pure Python for-loops iterating over every neighbor of pinned nodes, doing
   predicate set membership checks, category lookups, qualifier constraint
   matching, and dedup via `seen_edges` set. For high-degree nodes (10K+
   neighbors), this is millions of Python interpreter overhead calls per query.

2. **GC pressure** (`gc_utils.py`): The codebase dedicates an entire module to
   fighting Python's garbage collector -- freezing objects, disabling GC during
   queries, monitoring collection events. This is a symptom of Python's memory
   model being a poor fit for long-lived, read-heavy data structures. Rust
   eliminates this entire problem category.

3. **Property dict allocation** (`graph.py` `_get_props`): Every edge property
   lookup creates a new Python dict with references to interned pool objects.
   Even though the pool objects are shared, the wrapper dicts are per-call
   allocations that the GC must track. In Rust, this could be a zero-copy
   struct reference.

4. **Path assembly and enrichment** (`search/reconstruct.py`,
   `search/path_finder.py`): Building result lists of dicts with string keys
   is inherently expensive in Python. Rust could return structured data with
   minimal allocation.

### Python is already fast enough here (Rust wouldn't help much)

1. **NumPy vectorized operations** (`path_finder.py:147-206`): The unfiltered
   search uses `np.concatenate`, `np.isin`, `np.column_stack` -- these are
   already C-speed. Rewriting these in Rust gives marginal improvement.

2. **LMDB operations** (`lmdb_store.py`): Already a C library with
   memory-mapped I/O. The Rust `lmdb-rkv` crate is a wrapper around the same
   C library.

3. **BMT ontology lookups** (`search/expanders.py`): Called once per query
   during predicate expansion, not in the hot loop. The overhead is negligible.

4. **FastAPI/HTTP layer** (`server.py`): Network I/O dominates. The difference
   between uvicorn and hyper/actix is noise compared to graph traversal time.

---

## Why NOT a Full Rewrite

1. **BMT dependency**: The Biolink Model Toolkit is Python-only with a complex
   ontology model. Reimplementing it in Rust (or maintaining a Python subprocess
   bridge) would be a large, ongoing maintenance burden for unclear benefit,
   since it's not on the hot path.

2. **Alpha-stage churn**: At v0.1.8 with active development, the API surface
   and data model are still changing. A full rewrite freezes a moving target
   and doubles every future change.

3. **Ecosystem loss**: pytest fixtures, FastAPI's automatic OpenAPI docs,
   Python's TRAPI libraries, Jupyter notebook integration -- these would all
   need replacements or workarounds.

4. **Team velocity**: Python iteration speed matters during active development.
   A Rust rewrite means slower feature development for performance gains that
   can be achieved more surgically.

5. **10K lines isn't that much**: The codebase is small enough that Python's
   overhead is bounded. This isn't a 100K-line system where language overhead
   compounds across layers.

---

## Recommended Plan: PyO3 Hybrid

### Phase 1: Rust Core via PyO3 (highest impact)

Replace the CSR data structure and traversal engine with a Rust extension
module, exposed to Python via PyO3/maturin.

**What moves to Rust:**

```
gandalf-core/  (new Rust crate, built with maturin)
  src/
    lib.rs              # PyO3 module definition
    csr.rs              # CSRGraph struct (fwd/rev offsets, indices, predicates)
    edge_property.rs    # EdgePropertyStore (interned pools, zero-copy access)
    traversal.rs        # neighbor iteration, filtered traversal, binary search
    query_edge.rs       # Forward/backward/both-pinned edge query with filtering
    qualifiers.rs       # Qualifier constraint matching
    path_arrays.rs      # Compact path representation
```

**What stays in Python:**

```
gandalf/
  search/lookup.py      # TRAPI query orchestration (calls Rust traversal)
  search/expanders.py   # BMT-dependent predicate/qualifier expansion
  search/reconstruct.py # Path reconstruction (could move later)
  loader.py             # JSONL parsing (I/O bound, Python is fine)
  server.py             # FastAPI REST API
  enrichment.py         # Response enrichment (LMDB lookups)
  lmdb_store.py         # Cold-path storage (already C-backed)
  query_planner.py      # Query planning logic
  diagnostics.py        # Analysis tools
```

**Expected gains:**
- Eliminate GC module entirely (no more gc_utils.py)
- 10-50x speedup on neighbor traversal inner loops
- Near-zero allocation for edge property lookups
- Memory reduction from compact Rust structs vs Python dicts
- Safe parallelism potential (rayon) without GIL

**Build integration:**
- Use `maturin develop` for local dev, `maturin build` for wheels
- Update `pyproject.toml` to use maturin as build backend
- CI builds wheels for Linux/macOS/Windows

### Phase 2: Parallel Query Execution (medium impact)

Once the core is in Rust, use rayon to parallelize independent edge queries
within a single TRAPI lookup. Currently each qedge is solved sequentially;
with Rust ownership semantics, multiple edges can be queried concurrently on
the shared immutable graph.

### Phase 3: Rust-native Loader (lower impact, optional)

Replace the 3-pass Python loader with a Rust implementation using `serde_json`
for streaming JSONL parsing. This is I/O-bound so the gains are moderate, but
it eliminates the peak memory spike from Python object overhead during loading.

### Phase 4: Full Rust Server (optional, future)

If the project stabilizes and the team gains Rust expertise, consider moving
the HTTP layer to axum/actix-web. This only makes sense if:
- The BMT dependency is removed or stabilized enough to port
- The TRAPI format is stable
- The team is comfortable maintaining Rust long-term

---

## Estimated Effort & Risk

| Phase | Scope | Risk | Prerequisite |
|-------|-------|------|--------------|
| Phase 1 | Rust core + PyO3 bindings | Medium -- PyO3 API is mature but CSR+property store is the heart of the system | None |
| Phase 2 | Parallel traversal | Low -- rayon integration is straightforward once Phase 1 is done | Phase 1 |
| Phase 3 | Rust loader | Low -- mostly mechanical | Phase 1 |
| Phase 4 | Full server | High -- BMT replacement, ecosystem loss | Phases 1-3 + project stability |

## Key Decision Points

1. **Before starting**: Profile real queries on production-sized graphs to
   establish baselines. The GC overhead and traversal loop time should be
   measured, not assumed.

2. **After Phase 1**: Measure actual speedup. If the Python orchestration
   layer (lookup.py, reconstruct.py) becomes the new bottleneck, consider
   moving it to Rust. If not, stop here.

3. **Phase 4 gate**: Only proceed if the project has stabilized (v1.0+),
   the team wants to maintain Rust long-term, and the BMT dependency can be
   addressed.

---

## Alternative Considered: Cython

Cython could speed up the inner loops without leaving the Python ecosystem.
However:
- It doesn't solve the GC problem
- Type annotation coverage in the codebase is already high (good for PyO3)
- Cython's debugging/profiling story is worse than Rust's
- PyO3/maturin has better tooling and community momentum than Cython in 2025+

Cython would be a reasonable intermediate step if the team isn't ready for
Rust, but it caps the performance ceiling lower.
