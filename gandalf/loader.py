"""Load nodes and edges into Gandalf.

Uses a two-pass approach for memory efficiency:
  Pass 1: Scan edges to collect node IDs, predicates, and count edges.
  Pass 2: Re-read edges, converting to integer indices and interning
           properties directly into deduplication pools.

This avoids holding 38M+ individual Python dicts in memory simultaneously.
At 38M edges the old approach peaked at ~100GB; this approach stays under ~5GB.
"""

import gc
import json

import numpy as np

from gandalf.graph import CSRGraph, EdgePropertyStore


def build_graph_from_jsonl(
    edge_jsonl_path,
    node_jsonl_path,
):
    """
    Build a CSR graph from a JSONL file of edges.

    Args:
        edge_jsonl_path: Path to JSONL file where each line has 'subject', 'predicate', and 'object' fields
        node_jsonl_path: Path to JSONL file with node properties

    Returns:
        CSRGraph object with the loaded graph
    """

    # ===================================================================
    # PASS 1 — lightweight scan: collect node IDs, predicates, line count
    # ===================================================================
    print(f"Pass 1: Scanning {edge_jsonl_path}...")

    node_ids = set()
    unique_predicates = set()
    line_count = 0

    with open(edge_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 1_000_000 == 0:
                print(f"  Scanned {line_count:,} lines...")

            data = json.loads(line)
            node_ids.add(data["subject"])
            node_ids.add(data["object"])
            unique_predicates.add(data["predicate"])

    print(f"  {line_count:,} lines, {len(node_ids):,} unique nodes, "
          f"{len(unique_predicates):,} unique predicates")

    # Build mappings
    print("Building node index mapping...")
    node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted(node_ids))}
    num_nodes = len(node_ids)
    del node_ids  # Free ~200MB for 2M nodes

    print("Building predicate vocabulary...")
    predicate_to_idx = {p: i for i, p in enumerate(sorted(unique_predicates))}
    del unique_predicates

    # Load node properties — convert to int-keyed dict immediately
    node_properties = {}
    if node_jsonl_path:
        print(f"Reading node properties from {node_jsonl_path}...")
        with open(node_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                node_data = json.loads(line)
                node_id = node_data.get("id")
                if node_id:
                    idx = node_id_to_idx.get(node_id)
                    if idx is not None:
                        node_properties[idx] = {
                            "id": node_data.get("id", ""),
                            "categories": node_data.get("category", []),
                            "name": node_data.get("name", None),
                            "equivalent_identifiers": node_data.get("equivalent_identifiers", []),
                            "information_content": node_data.get("information_content", 0.0),
                        }
        print(f"  Loaded properties for {len(node_properties):,} nodes")

    # ===================================================================
    # PASS 2 — streaming build: integer indices + property interning
    # ===================================================================
    print(f"Pass 2: Building graph arrays from {edge_jsonl_path}...")

    # Pre-allocate numpy arrays (line_count is upper bound; trimmed later)
    src_array = np.empty(line_count, dtype=np.int32)
    dst_array = np.empty(line_count, dtype=np.int32)
    pred_array = np.empty(line_count, dtype=np.int32)

    # Streaming property interning — one pool + intern dict per field
    make_hashable = EdgePropertyStore._make_hashable

    pubs_intern = {}
    sources_intern = {}
    quals_intern = {}
    pubs_pool = []
    sources_pool = []
    quals_pool = []
    pubs_indices = np.empty(line_count, dtype=np.int32)
    sources_indices = np.empty(line_count, dtype=np.int32)
    quals_indices = np.empty(line_count, dtype=np.int32)

    # Attributes (all "other" fields) are NOT interned: they are typically
    # high-cardinality (unique per edge due to scores, provenance IDs, etc.)
    # which makes interning counter-productive (38M unique intern keys ≈ 100GB).
    # Instead we store a single empty entry and point all edges at it.
    attrs_pool = [[]]
    attrs_indices = np.zeros(line_count, dtype=np.int32)

    # Dedup using integer tuples (much cheaper than string tuples)
    seen_edges = set()

    # Fields consumed by dedicated extraction above (not dumped into attributes)
    core_fields = {
        "id", "category", "subject", "object", "predicate",
        "sources", "publications", "qualifiers",
        # Metadata fields handled via sources fallback or ignored
        "primary_knowledge_source", "knowledge_level", "agent_type",
    }
    qualifier_fields = {
        "qualified_predicate",
        "object_aspect_qualifier",
        "object_direction_qualifier",
        "subject_aspect_qualifier",
        "subject_direction_qualifier",
        "causal_mechanism_qualifier",
        "species_context_qualifier",
    }

    edge_count = 0
    duplicates = 0

    # Disable GC during the hot loop — millions of small allocations make
    # the cyclic GC very expensive; we'll collect once at the end.
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        with open(edge_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                src_idx = node_id_to_idx[data["subject"]]
                dst_idx = node_id_to_idx[data["object"]]
                pred_idx = predicate_to_idx[data["predicate"]]

                edge_key = (src_idx, dst_idx, pred_idx)
                if edge_key in seen_edges:
                    duplicates += 1
                    continue
                seen_edges.add(edge_key)

                src_array[edge_count] = src_idx
                dst_array[edge_count] = dst_idx
                pred_array[edge_count] = pred_idx

                # --- Intern publications ---
                pubs = data.get("publications", [])
                pubs_key = make_hashable(pubs)
                if pubs_key not in pubs_intern:
                    pubs_intern[pubs_key] = len(pubs_pool)
                    pubs_pool.append(pubs)
                pubs_indices[edge_count] = pubs_intern[pubs_key]

                # --- Intern sources ---
                raw_sources = data.get("sources", [])
                if raw_sources:
                    sources = [
                        {
                            "resource_id": s["resource_id"],
                            "resource_role": s["resource_role"],
                            **({"upstream_resource_ids": s["upstream_resource_ids"]}
                               if "upstream_resource_ids" in s else {})
                        }
                        for s in raw_sources
                    ]
                elif "primary_knowledge_source" in data:
                    # Fallback: build sources from top-level fields
                    sources = [{"resource_id": data["primary_knowledge_source"],
                                "resource_role": "primary_knowledge_source"}]
                else:
                    sources = []
                sources_key = make_hashable(sources)
                if sources_key not in sources_intern:
                    sources_intern[sources_key] = len(sources_pool)
                    sources_pool.append(sources)
                sources_indices[edge_count] = sources_intern[sources_key]

                # --- Intern qualifiers ---
                # Support two formats:
                #  1) Pre-built list: "qualifiers": [{"qualifier_type_id":..., "qualifier_value":...}, ...]
                #  2) Individual fields: "object_aspect_qualifier": "activity", ...
                raw_qualifiers = data.get("qualifiers", None)
                if raw_qualifiers and isinstance(raw_qualifiers, list):
                    quals = raw_qualifiers
                else:
                    # Build from individual top-level fields
                    quals = []
                    for field in qualifier_fields:
                        if field in data:
                            quals.append({
                                "qualifier_type_id": f"biolink:{field}",
                                "qualifier_value": data[field],
                            })
                quals_key = make_hashable(quals)
                if quals_key not in quals_intern:
                    quals_intern[quals_key] = len(quals_pool)
                    quals_pool.append(quals)
                quals_indices[edge_count] = quals_intern[quals_key]

                # Attributes (remaining fields) are skipped during loading —
                # too high-cardinality for interning to be effective.

                edge_count += 1
                if edge_count % 1_000_000 == 0:
                    print(f"  Processed {edge_count:,} edges...")

    finally:
        if gc_was_enabled:
            gc.enable()

    # Free intern dicts (pubs_intern can be large if publications are unique)
    del seen_edges, pubs_intern, sources_intern, quals_intern
    gc.collect()

    # Trim arrays to actual size
    if edge_count < line_count:
        src_array = src_array[:edge_count]
        dst_array = dst_array[:edge_count]
        pred_array = pred_array[:edge_count]
        pubs_indices = pubs_indices[:edge_count]
        sources_indices = sources_indices[:edge_count]
        quals_indices = quals_indices[:edge_count]
        attrs_indices = attrs_indices[:edge_count]

    print(f"  {edge_count:,} unique edges ({duplicates:,} duplicates skipped)")
    print(f"  Dedup pools: {len(pubs_pool):,} unique pub lists, "
          f"{len(sources_pool):,} unique source configs, "
          f"{len(quals_pool):,} unique qualifier combos, "
          f"{len(attrs_pool):,} unique attribute sets")

    # Build EdgePropertyStore directly from interned data
    edge_properties = EdgePropertyStore.from_arrays_and_pools(
        pubs_idx=pubs_indices,
        sources_idx=sources_indices,
        quals_idx=quals_indices,
        attrs_idx=attrs_indices,
        pubs_pool=pubs_pool,
        sources_pool=sources_pool,
        quals_pool=quals_pool,
        attrs_pool=attrs_pool,
    )

    # Stack src/dst into Nx2 array for CSRGraph constructor
    edges = np.column_stack((src_array, dst_array))

    # ===================================================================
    # Build CSR structure
    # ===================================================================
    print("Building CSR structure...")
    graph = CSRGraph(
        num_nodes=num_nodes,
        edges=edges,
        edge_predicates=pred_array,
        node_id_to_idx=node_id_to_idx,
        predicate_to_idx=predicate_to_idx,
        edge_properties=edge_properties,
        node_properties=node_properties,
    )

    # Print statistics
    degrees = [graph.degree(i) for i in range(min(1000, graph.num_nodes))]
    if degrees:
        print("\nGraph statistics:")
        print(f"  Nodes: {graph.num_nodes:,}")
        print(f"  Edges: {len(graph.fwd_targets):,}")
        print(f"  Unique predicates: {len(predicate_to_idx):,}")
        print(f"  Avg degree (sampled): {np.mean(degrees):.1f}")
        print(f"  Max degree (sampled): {np.max(degrees)}")
        memory_mb = (
            (
                graph.fwd_targets.nbytes
                + graph.fwd_offsets.nbytes
                + graph.fwd_predicates.nbytes
            )
            / 1024
            / 1024
        )
        print(f"  CSR memory: ~{memory_mb:.1f} MB")

    return graph
