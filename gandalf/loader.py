"""Load nodes and edges into Gandalf.

Single-pass streaming loader that keeps peak memory at ~3-4GB for 38M edges:

Pass 1: Stream JSONL once, building vocabularies incrementally while
        simultaneously converting each edge to integer indices in growing
        Python lists. Interns qualifier/source data into the
        EdgePropertyStoreBuilder and writes attributes to a temporary LMDB
        keyed by original line index.
Pass 2: Convert lists to numpy arrays, sort by (src, dst, pred) via
        np.lexsort. Rewrite the temp LMDB in CSR-sorted order to produce
        the final LMDB where key == CSR edge index (zero indirection at
        query time). Reorder qualifier/source dedup indices to match.
        Build CSR offset arrays.
"""

import array as _array
import shutil
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import orjson

from gandalf.graph import CSRGraph, GrowablePropertyStoreBuilder
from gandalf.lmdb_store import LMDBPropertyStore, _INITIAL_WRITE_MAP_SIZE, _encode_key, _put_with_resize

import lmdb


# Fields that are structural (not stored as properties)
_CORE_FIELDS = {
    "id", "category", "subject", "object", "predicate", "sources",
}

# Node fields that become top-level TRAPI Node properties (not attributes)
_CORE_NODE_FIELDS = {"id", "name", "category"}

# Known qualifier fields that appear as top-level JSONL keys
_QUALIFIER_FIELDS = {
    "qualified_predicate",
    "object_aspect_qualifier",
    "object_direction_qualifier",
    "subject_aspect_qualifier",
    "subject_direction_qualifier",
    "causal_mechanism_qualifier",
    "species_context_qualifier",
}

# Union for fast membership testing during attribute extraction
_SKIP_FIELDS = _CORE_FIELDS | _QUALIFIER_FIELDS | {"qualifiers"}


def _extract_sources(data):
    """Extract normalized source list from edge data.

    Ensures every source has an ``upstream_resource_ids`` list (defaults to
    ``[]``) and prepends an ``infores:gandalf`` aggregator_knowledge_source
    whose upstream points to the top of the existing source chain (i.e. the
    source(s) not referenced in any other source's upstream_resource_ids).
    """
    raw = data.get("sources", [])

    # Normalize: guarantee upstream_resource_ids on every source
    sources = [
        {
            "resource_id": s["resource_id"],
            "resource_role": s["resource_role"],
            "upstream_resource_ids": s.get("upstream_resource_ids", []),
        }
        for s in raw
    ]

    # Find the top of the source chain: sources whose resource_id is NOT
    # referenced in any other source's upstream_resource_ids.  These are the
    # "leaf" providers that no one else aggregates from yet.
    all_upstream = {
        uid
        for s in sources
        for uid in s["upstream_resource_ids"]
    }
    top_ids = [
        s["resource_id"]
        for s in sources
        if s["resource_id"] not in all_upstream
    ]

    # Prepend gandalf as aggregator_knowledge_source
    gandalf_source = {
        "resource_id": "infores:gandalf",
        "resource_role": "aggregator_knowledge_source",
        "upstream_resource_ids": top_ids,
    }

    return [gandalf_source] + sources


def _extract_qualifiers(data):
    """Extract qualifiers.

    Format: Top-level fields (object_aspect_qualifier, etc.)
    """
    qualifiers = []
    for field in _QUALIFIER_FIELDS:
        if field in data:
            qualifiers.append({
                "qualifier_type_id": f"biolink:{field}",
                "qualifier_value": data[field],
            })

    return qualifiers


def _extract_attributes(data):
    """Extract attributes (everything not in core/qualifier/source fields).

    Publications are included as a TRAPI Attribute with
    ``attribute_type_id`` of ``biolink:publications``.
    """
    attributes = []
    for field, value in data.items():
        if field in _SKIP_FIELDS:
            continue
        attributes.append({
            "attribute_type_id": f"biolink:{field}",
            "value": value,
            "original_attribute_name": field,
        })
    return attributes


def _extract_node_attributes(node_data):
    """Extract node attributes as TRAPI Attribute objects.

    Any field not in ``_CORE_NODE_FIELDS`` (id, name, category) is converted
    to a TRAPI-compliant Attribute dict with ``attribute_type_id``, ``value``,
    and ``original_attribute_name``.
    """
    attributes = []
    for field, value in node_data.items():
        if field in _CORE_NODE_FIELDS:
            continue
        attributes.append({
            "attribute_type_id": "biolink:Attribute",
            "value": value,
            "original_attribute_name": field,
        })
    return attributes


def build_graph_from_jsonl(edge_jsonl_path, node_jsonl_path, temp_dir=None):
    """Build a CSR graph from JSONL files using optimized streaming.

    Pass 1 (single pass): Build vocabularies incrementally while collecting
        edge data into growing lists + dedup store + temp LMDB.
    Pass 2: Sort, rewrite LMDB in CSR order, build offsets.

    Args:
        edge_jsonl_path: Path to the edges JSONL file.
        node_jsonl_path: Path to the nodes JSONL file.
        temp_dir: Directory for temporary build files (LMDB scratch space).
            Defaults to the parent directory of edge_jsonl_path.  Set this
            to a partition with sufficient free space (roughly 2x the
            edges.jsonl size).

    Peak memory: ~3-4GB for 38M edges (down from 100GB+).
    """
    edge_jsonl_path = str(edge_jsonl_path)
    node_jsonl_path = str(node_jsonl_path) if node_jsonl_path else None

    # =================================================================
    # Pass 1: Single-pass vocabulary + array building
    # =================================================================
    print(f"Pass 1: Streaming edges from {edge_jsonl_path}...")

    # Vocabularies built incrementally (assign index on first encounter)
    node_id_to_idx = {}
    predicate_to_idx = {}

    # Growing arrays for edge data — array.array('i') uses 4 bytes per int
    # (same as numpy int32), vs 28 bytes per int in a Python list.
    src_arr = _array.array('i')
    dst_arr = _array.array('i')
    pred_arr = _array.array('i')
    edge_ids = []

    # Inline interning of hot-path properties (qualifiers + sources).
    # Only the intern dicts and small pools are kept in memory — each edge
    # contributes just 4+4 = 8 bytes of index storage, not a full dict.
    prop_builder = GrowablePropertyStoreBuilder()

    # Temp LMDB for cold-path properties (publications, attributes)
    # Default to same partition as the input data to avoid filling /tmp.
    build_temp_root = temp_dir or str(Path(edge_jsonl_path).resolve().parent)
    temp_dir_path = tempfile.mkdtemp(prefix="gandalf_build_", dir=build_temp_root)
    temp_lmdb_path = Path(temp_dir_path) / "temp_props.lmdb"
    temp_lmdb_path.mkdir(parents=True, exist_ok=True)

    temp_env = lmdb.open(
        str(temp_lmdb_path),
        map_size=_INITIAL_WRITE_MAP_SIZE,
        readonly=False,
        max_dbs=0,
        readahead=False,
    )

    txn = temp_env.begin(write=True)
    pending = []
    edge_count = 0
    commit_every = 200_000

    try:
        with open(edge_jsonl_path, "rb") as f:
            for line in f:
                data = orjson.loads(line)

                # Assign node indices incrementally
                subj = data["subject"]
                obj = data["object"]
                pred = data["predicate"]

                if subj not in node_id_to_idx:
                    node_id_to_idx[subj] = len(node_id_to_idx)
                if obj not in node_id_to_idx:
                    node_id_to_idx[obj] = len(node_id_to_idx)
                if pred not in predicate_to_idx:
                    predicate_to_idx[pred] = len(predicate_to_idx)

                src_arr.append(node_id_to_idx[subj])
                dst_arr.append(node_id_to_idx[obj])
                pred_arr.append(predicate_to_idx[pred])

                # Capture edge ID from JSONL (if present)
                edge_ids.append(data.get("id"))

                # Extract properties
                sources = _extract_sources(data)
                qualifiers = _extract_qualifiers(data)
                attributes = _extract_attributes(data)

                # Hot path: intern inline (only 8 bytes of index storage per edge)
                prop_builder.append(sources, qualifiers)

                # Cold path: write attributes to temp LMDB
                if attributes:
                    detail = {"attributes": attributes}
                    key = _encode_key(edge_count)
                    val = msgpack.packb(detail, use_bin_type=True)
                    txn = _put_with_resize(temp_env, txn, key, val, pending)

                edge_count += 1

                if edge_count % commit_every == 0:
                    txn.commit()
                    pending.clear()
                    txn = temp_env.begin(write=True)

                if edge_count % 1_000_000 == 0:
                    print(f"  {edge_count:,} edges processed...")

        txn.commit()
    except BaseException:
        txn.abort()
        raise
    finally:
        temp_env.close()

    num_nodes = len(node_id_to_idx)
    print(f"  Found {num_nodes:,} unique nodes, {len(predicate_to_idx):,} predicates, "
          f"{edge_count:,} edges")

    # Convert array.array to numpy (zero-copy view + copy for ownership)
    print("  Converting to numpy arrays...")
    src_indices = np.frombuffer(src_arr, dtype=np.int32).copy()
    dst_indices = np.frombuffer(dst_arr, dtype=np.int32).copy()
    pred_indices = np.frombuffer(pred_arr, dtype=np.int32).copy()
    del src_arr, dst_arr, pred_arr

    # Load node properties
    node_properties = {}
    if node_jsonl_path:
        print(f"Reading node properties from {node_jsonl_path}...")
        with open(node_jsonl_path, "rb") as f:
            for line in f:
                node_data = orjson.loads(line)
                node_id = node_data.get("id")
                if node_id:
                    idx = node_id_to_idx.get(node_id)
                    if idx is not None:
                        node_properties[idx] = {
                            "name": node_data.get("name", None),
                            "categories": node_data.get("category", []),
                            "attributes": _extract_node_attributes(node_data),
                        }
        print(f"  Loaded properties for {len(node_properties):,} nodes")

    print(f"  Arrays and temp LMDB built")

    # =================================================================
    # Pass 2: Sort, rewrite LMDB, build CSR
    # =================================================================
    print("Pass 2: Sorting and building CSR structure...")

    # Sort by (src, dst, pred) using lexsort (last key is primary)
    sort_order = np.lexsort((pred_indices, dst_indices, src_indices))

    # Reorder numpy arrays
    src_sorted = src_indices[sort_order]
    dst_sorted = dst_indices[sort_order]
    pred_sorted = pred_indices[sort_order]

    # Free unsorted arrays
    del src_indices, dst_indices, pred_indices

    # Reorder edge IDs to match CSR sort order (vectorized via numpy)
    edge_ids_arr = np.array(edge_ids, dtype=object)
    edge_ids_sorted = edge_ids_arr[sort_order].tolist()
    del edge_ids, edge_ids_arr

    # Reorder dedup store indices to match CSR order
    prop_builder.reorder(sort_order)
    edge_properties = prop_builder.build()
    del prop_builder

    stats = edge_properties.dedup_stats()
    print(f"  Edge property dedup: {stats['total_edges']:,} edges -> "
          f"{stats['unique_sources']:,} unique source configs, "
          f"{stats['unique_qualifiers']:,} unique qualifier combos")

    # Rewrite temp LMDB in CSR-sorted order → final LMDB
    # This is the expensive build-time step, but ensures query-time
    # LMDB key == CSR edge index with zero indirection.
    final_lmdb_path = Path(temp_dir_path) / "edge_properties.lmdb"
    lmdb_store = LMDBPropertyStore.build_sorted(
        db_path=final_lmdb_path,
        temp_db_path=temp_lmdb_path,
        sort_permutation=sort_order,
        num_edges=edge_count,
    )

    # Clean up temp LMDB
    shutil.rmtree(temp_lmdb_path)
    # NOTE: sort_order is kept alive — needed for rev_to_fwd mapping below.

    # Build CSR offset arrays using searchsorted
    print("  Building CSR offsets...")
    fwd_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    if edge_count > 0:
        boundaries = np.searchsorted(src_sorted, np.arange(num_nodes + 1))
        fwd_offsets = boundaries.astype(np.int64)
    del src_sorted

    # Build reverse CSR: sort edges by (dst, src, pred)
    print("  Building reverse CSR...")
    # Reconstruct per-edge source node IDs from forward CSR offsets (vectorized)
    edge_src = np.repeat(
        np.arange(num_nodes, dtype=np.int32),
        np.diff(fwd_offsets).astype(np.int32),
    )

    rev_order = np.lexsort((pred_sorted, edge_src, dst_sorted))

    rev_dst_sorted = dst_sorted[rev_order]
    rev_sources = edge_src[rev_order]
    rev_predicates = pred_sorted[rev_order]

    rev_offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    if edge_count > 0:
        boundaries = np.searchsorted(rev_dst_sorted, np.arange(num_nodes + 1))
        rev_offsets = boundaries.astype(np.int64)

    # Build rev_to_fwd mapping: for each reverse-CSR position, store the
    # corresponding forward-CSR position.  Forward positions are simply
    # 0..E-1 (the arrays are already in forward-sorted order).  The
    # inverse of sort_order maps original-edge-index → forward position.
    print("  Building rev_to_fwd mapping...")
    fwd_pos = np.empty(edge_count, dtype=np.int32)
    fwd_pos[sort_order] = np.arange(edge_count, dtype=np.int32)
    rev_to_fwd = rev_order.astype(np.int32)
    del fwd_pos, sort_order

    del edge_src, rev_dst_sorted, rev_order

    # Assemble the graph
    print("  Assembling graph...")
    graph = CSRGraph.__new__(CSRGraph)
    graph.num_nodes = num_nodes
    graph.node_id_to_idx = node_id_to_idx
    graph.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
    graph.predicate_to_idx = predicate_to_idx
    graph.id_to_predicate = {idx: pred for pred, idx in predicate_to_idx.items()}
    graph.node_properties = node_properties

    graph.fwd_targets = dst_sorted
    graph.fwd_predicates = pred_sorted
    graph.fwd_offsets = fwd_offsets

    graph.rev_sources = rev_sources
    graph.rev_predicates = rev_predicates
    graph.rev_offsets = rev_offsets
    graph.rev_to_fwd = rev_to_fwd

    graph.edge_properties = edge_properties
    graph.edge_ids = edge_ids_sorted
    graph.lmdb_store = lmdb_store

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
            graph.fwd_targets.nbytes
            + graph.fwd_offsets.nbytes
            + graph.fwd_predicates.nbytes
        ) / 1024 / 1024
        print(f"  CSR memory usage: ~{memory_mb:.1f} MB")

    return graph
