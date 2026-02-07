"""Main Gandalf CSR Graph class."""

import gc
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np


class EdgePropertyStore:
    """Memory-efficient storage for edge properties using deduplication.

    Instead of storing a separate dict per edge, this class deduplicates
    common field values across edges. Each unique publications list, sources
    list, qualifiers list, and attributes list is stored only once in a shared
    pool, and edges reference them by integer index.

    For a graph with 100M edges where there might be only ~50 unique source
    configs, ~10K unique qualifier combos, and ~1M unique publication sets,
    this typically reduces memory from ~30GB to ~2-3GB.

    The 'predicate' key is NOT stored here since it's already in the CSR
    arrays as an integer (fwd_predicates). CSRGraph accessor methods
    synthesize it when needed.
    """

    __slots__ = (
        '_pubs_pool', '_sources_pool', '_quals_pool', '_attrs_pool',
        '_pubs_idx', '_sources_idx', '_quals_idx', '_attrs_idx',
    )

    def __init__(self):
        self._pubs_pool = []
        self._sources_pool = []
        self._quals_pool = []
        self._attrs_pool = []
        self._pubs_idx = None
        self._sources_idx = None
        self._quals_idx = None
        self._attrs_idx = None

    @staticmethod
    def _make_hashable(obj):
        """Convert a JSON-compatible value to a hashable key for interning."""
        if isinstance(obj, dict):
            return tuple(sorted(
                (k, EdgePropertyStore._make_hashable(v)) for k, v in obj.items()
            ))
        elif isinstance(obj, (list, tuple)):
            return tuple(EdgePropertyStore._make_hashable(item) for item in obj)
        return obj

    @classmethod
    def from_arrays_and_pools(cls, pubs_idx, sources_idx, quals_idx, attrs_idx,
                               pubs_pool, sources_pool, quals_pool, attrs_pool):
        """Build store from pre-interned arrays and pools.

        Used by the streaming loader to construct the store directly without
        building intermediate property dicts.
        """
        store = cls()
        store._pubs_idx = pubs_idx
        store._sources_idx = sources_idx
        store._quals_idx = quals_idx
        store._attrs_idx = attrs_idx
        store._pubs_pool = pubs_pool
        store._sources_pool = sources_pool
        store._quals_pool = quals_pool
        store._attrs_pool = attrs_pool
        return store

    @classmethod
    def from_property_list(cls, props_list):
        """Build an EdgePropertyStore from a list of property dicts.

        Args:
            props_list: List of dicts, each with keys like 'publications',
                        'sources', 'qualifiers', 'attributes'. The 'predicate'
                        key is ignored (stored separately in CSR arrays).
        """
        store = cls()
        n = len(props_list)

        pubs_intern = {}
        sources_intern = {}
        quals_intern = {}
        attrs_intern = {}

        pubs_indices = np.empty(n, dtype=np.int32)
        sources_indices = np.empty(n, dtype=np.int32)
        quals_indices = np.empty(n, dtype=np.int32)
        attrs_indices = np.empty(n, dtype=np.int32)

        for i, props in enumerate(props_list):
            # Intern publications
            pubs = props.get("publications", [])
            pubs_key = cls._make_hashable(pubs)
            if pubs_key not in pubs_intern:
                pubs_intern[pubs_key] = len(store._pubs_pool)
                store._pubs_pool.append(pubs)
            pubs_indices[i] = pubs_intern[pubs_key]

            # Intern sources
            sources = props.get("sources", [])
            sources_key = cls._make_hashable(sources)
            if sources_key not in sources_intern:
                sources_intern[sources_key] = len(store._sources_pool)
                store._sources_pool.append(sources)
            sources_indices[i] = sources_intern[sources_key]

            # Intern qualifiers
            quals = props.get("qualifiers", [])
            quals_key = cls._make_hashable(quals)
            if quals_key not in quals_intern:
                quals_intern[quals_key] = len(store._quals_pool)
                store._quals_pool.append(quals)
            quals_indices[i] = quals_intern[quals_key]

            # Intern attributes
            attrs = props.get("attributes", [])
            attrs_key = cls._make_hashable(attrs)
            if attrs_key not in attrs_intern:
                attrs_intern[attrs_key] = len(store._attrs_pool)
                store._attrs_pool.append(attrs)
            attrs_indices[i] = attrs_intern[attrs_key]

        store._pubs_idx = pubs_indices
        store._sources_idx = sources_indices
        store._quals_idx = quals_indices
        store._attrs_idx = attrs_indices

        return store

    def reorder(self, order):
        """Return a new store with entries reordered by the given index array.

        The pools are shared (not copied) since they're immutable after
        construction. Only the per-edge index arrays are reordered.
        """
        store = EdgePropertyStore()
        store._pubs_pool = self._pubs_pool
        store._sources_pool = self._sources_pool
        store._quals_pool = self._quals_pool
        store._attrs_pool = self._attrs_pool
        store._pubs_idx = self._pubs_idx[order]
        store._sources_idx = self._sources_idx[order]
        store._quals_idx = self._quals_idx[order]
        store._attrs_idx = self._attrs_idx[order]
        return store

    def __len__(self):
        if self._pubs_idx is None:
            return 0
        return len(self._pubs_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._get_props(i) for i in indices]
        return self._get_props(key)

    def _get_props(self, idx):
        """Get the full property dict for an edge at the given index."""
        return {
            "publications": self._pubs_pool[self._pubs_idx[idx]],
            "sources": self._sources_pool[self._sources_idx[idx]],
            "qualifiers": self._quals_pool[self._quals_idx[idx]],
            "attributes": self._attrs_pool[self._attrs_idx[idx]],
        }

    def get_field(self, idx, key, default=None):
        """Get a single field value without creating a full dict."""
        if key == "publications":
            return self._pubs_pool[self._pubs_idx[idx]]
        elif key == "sources":
            return self._sources_pool[self._sources_idx[idx]]
        elif key == "qualifiers":
            return self._quals_pool[self._quals_idx[idx]]
        elif key == "attributes":
            return self._attrs_pool[self._attrs_idx[idx]]
        return default

    def dedup_stats(self):
        """Return statistics about deduplication effectiveness."""
        n = len(self)
        return {
            "total_edges": n,
            "unique_publications": len(self._pubs_pool),
            "unique_sources": len(self._sources_pool),
            "unique_qualifiers": len(self._quals_pool),
            "unique_attributes": len(self._attrs_pool),
        }

    def save_mmap(self, directory: Path):
        """Save to directory as mmap-friendly files.

        Writes the per-edge index arrays as .npy files (memory-mappable)
        and the intern pools as a single pickle.
        """
        np.save(directory / "edge_pubs_idx.npy", self._pubs_idx)
        np.save(directory / "edge_sources_idx.npy", self._sources_idx)
        np.save(directory / "edge_quals_idx.npy", self._quals_idx)
        np.save(directory / "edge_attrs_idx.npy", self._attrs_idx)

        pools = {
            "pubs_pool": self._pubs_pool,
            "sources_pool": self._sources_pool,
            "quals_pool": self._quals_pool,
            "attrs_pool": self._attrs_pool,
        }
        with open(directory / "edge_property_pools.pkl", "wb") as f:
            pickle.dump(pools, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_mmap(cls, directory: Path, mmap_mode: str = "r"):
        """Load from directory, memory-mapping the index arrays.

        The per-edge index arrays are loaded via np.load with mmap_mode so
        they stay on disk and are paged in on demand (and shared across
        forked workers via copy-on-write).  The pools are small and loaded
        fully into RAM.
        """
        store = cls()

        store._pubs_idx = np.load(
            directory / "edge_pubs_idx.npy", mmap_mode=mmap_mode
        )
        store._sources_idx = np.load(
            directory / "edge_sources_idx.npy", mmap_mode=mmap_mode
        )
        store._quals_idx = np.load(
            directory / "edge_quals_idx.npy", mmap_mode=mmap_mode
        )

        # Handle backward compat: older graphs may not have attrs
        attrs_path = directory / "edge_attrs_idx.npy"
        if attrs_path.exists():
            store._attrs_idx = np.load(attrs_path, mmap_mode=mmap_mode)
        else:
            store._attrs_idx = np.zeros(len(store._pubs_idx), dtype=np.int32)
            store._attrs_pool = [[]]  # Single empty-list entry at index 0

        with open(directory / "edge_property_pools.pkl", "rb") as f:
            pools = pickle.load(f)

        store._pubs_pool = pools["pubs_pool"]
        store._sources_pool = pools["sources_pool"]
        store._quals_pool = pools["quals_pool"]
        if "attrs_pool" in pools:
            store._attrs_pool = pools["attrs_pool"]

        return store


class CSRGraph:
    """
    Compressed Sparse Row graph representation for fast neighbor lookups.

    Maintains two CSR structures:
    - Forward: node -> outgoing edges (who does this node point to?)
    - Reverse: node -> incoming edges (who points to this node?)
    """

    def __init__(
        self,
        num_nodes,
        edges,
        edge_predicates,
        node_id_to_idx,
        predicate_to_idx,
        edge_properties=None,
        node_properties=None,
    ):
        """
        Build a CSR graph from edge arrays and a pre-built EdgePropertyStore.

        Args:
            num_nodes: Total number of unique nodes
            edges: Nx2 numpy array of (src_idx, dst_idx) or list of tuples
            edge_predicates: numpy array of predicate IDs (parallel to edges)
            node_id_to_idx: Dict mapping original node IDs to integer indices
            predicate_to_idx: Dict mapping predicate strings to integer IDs
            edge_properties: EdgePropertyStore with deduped properties
            node_properties: Dict mapping node_idx -> properties dict
        """
        self.num_nodes = num_nodes
        self.node_id_to_idx = node_id_to_idx
        self.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
        self.node_properties = node_properties or {}

        # Predicate vocabulary
        self.predicate_to_idx = predicate_to_idx
        self.id_to_predicate = {idx: pred for pred, idx in predicate_to_idx.items()}

        # Convert inputs to numpy arrays if needed
        if isinstance(edges, np.ndarray) and edges.ndim == 2:
            src_arr = edges[:, 0].astype(np.int32)
            dst_arr = edges[:, 1].astype(np.int32)
        elif isinstance(edges, np.ndarray) and edges.ndim == 1 and len(edges) == 0:
            src_arr = np.array([], dtype=np.int32)
            dst_arr = np.array([], dtype=np.int32)
        else:
            # list of tuples
            if len(edges) > 0:
                src_arr = np.array([e[0] for e in edges], dtype=np.int32)
                dst_arr = np.array([e[1] for e in edges], dtype=np.int32)
            else:
                src_arr = np.array([], dtype=np.int32)
                dst_arr = np.array([], dtype=np.int32)

        pred_arr = np.asarray(edge_predicates, dtype=np.int32)

        # Build both forward and reverse CSR structures
        print("Building forward CSR...")
        self._build_forward_csr(src_arr, dst_arr, pred_arr, edge_properties)

        print("Building reverse CSR...")
        self._build_reverse_csr(src_arr, dst_arr, pred_arr)

    def _build_forward_csr(self, src, dst, pred, edge_properties):
        """Build forward adjacency list (source -> targets) using numpy sorting."""
        if len(src) == 0:
            self.fwd_targets = np.array([], dtype=np.int32)
            self.fwd_predicates = np.array([], dtype=np.int32)
            self.fwd_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)
            self.edge_properties = edge_properties or EdgePropertyStore()
            return

        # Sort edges by (src, dst, pred) using numpy lexsort
        sort_order = np.lexsort((pred, dst, src))

        self.fwd_targets = dst[sort_order]
        self.fwd_predicates = pred[sort_order]

        # Reorder edge properties to match sorted order
        if edge_properties is not None:
            self.edge_properties = edge_properties.reorder(sort_order)
            stats = self.edge_properties.dedup_stats()
            print(f"  Edge property dedup: {stats['total_edges']:,} edges -> "
                  f"{stats['unique_publications']:,} unique publication lists, "
                  f"{stats['unique_sources']:,} unique source configs, "
                  f"{stats['unique_qualifiers']:,} unique qualifier combos")
        else:
            self.edge_properties = EdgePropertyStore()

        # Build forward offsets using numpy
        src_sorted = src[sort_order]
        self.fwd_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        # Count edges per source node
        unique_src, counts = np.unique(src_sorted, return_counts=True)
        for node_idx, count in zip(unique_src, counts):
            self.fwd_offsets[node_idx + 1] = count
        np.cumsum(self.fwd_offsets, out=self.fwd_offsets)

    def _build_reverse_csr(self, src, dst, pred):
        """Build reverse adjacency list (target -> sources) using numpy sorting."""
        if len(src) == 0:
            self.rev_sources = np.array([], dtype=np.int32)
            self.rev_predicates = np.array([], dtype=np.int32)
            self.rev_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)
            return

        # Sort by (dst, src, pred) for reverse CSR
        sort_order = np.lexsort((pred, src, dst))

        self.rev_sources = src[sort_order]
        self.rev_predicates = pred[sort_order]

        # Build reverse offsets
        dst_sorted = dst[sort_order]
        self.rev_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        unique_dst, counts = np.unique(dst_sorted, return_counts=True)
        for node_idx, count in zip(unique_dst, counts):
            self.rev_offsets[node_idx + 1] = count
        np.cumsum(self.rev_offsets, out=self.rev_offsets)

    def _find_edge_index(self, src_idx, dst_idx, pred_id):
        """Find the position of an edge in the forward CSR arrays.

        Scans the CSR offset range for src_idx to find a matching
        (dst_idx, pred_id) pair. O(degree) but fast for typical node degrees.
        """
        start = int(self.fwd_offsets[src_idx])
        end = int(self.fwd_offsets[src_idx + 1])
        if start == end:
            return None

        targets = self.fwd_targets[start:end]
        preds = self.fwd_predicates[start:end]

        mask = (targets == dst_idx) & (preds == pred_id)
        indices = np.nonzero(mask)[0]
        if len(indices) > 0:
            return start + int(indices[0])
        return None

    def neighbors(self, node_idx, predicate_filter=None):
        """
        Get neighbor indices for a node index (nodes this node points TO)

        Args:
            node_idx: Node index to get neighbors for
            predicate_filter: Optional predicate string to filter edges

        Returns:
            numpy array of neighbor indices
        """
        start = self.fwd_offsets[node_idx]
        end = self.fwd_offsets[node_idx + 1]

        if predicate_filter is None:
            # Return all neighbors
            return self.fwd_targets[start:end]
        else:
            # Filter by predicate
            pred_id = self.predicate_to_idx.get(predicate_filter)
            if pred_id is None:
                return np.array([], dtype=np.int32)

            neighbors = self.fwd_targets[start:end]
            predicates = self.fwd_predicates[start:end]

            # Filter by predicate
            mask = predicates == pred_id
            return neighbors[mask]

    def incoming_neighbors(self, node_idx, predicate_filter=None):
        """
        Get incoming neighbors (nodes that point TO this node).

        Args:
            node_idx: Node index to get incoming neighbors for
            predicate_filter: Optional predicate string to filter edges

        Returns:
            numpy array of source node indices
        """
        start = self.rev_offsets[node_idx]
        end = self.rev_offsets[node_idx + 1]

        if predicate_filter is None:
            return self.rev_sources[start:end]
        else:
            pred_id = self.predicate_to_idx.get(predicate_filter)
            if pred_id is None:
                return np.array([], dtype=np.int32)

            sources = self.rev_sources[start:end]
            predicates = self.rev_predicates[start:end]
            mask = predicates == pred_id
            return sources[mask]

    def neighbors_with_properties(
        self, node_idx: int, predicate_filter: Optional[list] = None
    ):
        """
        Get neighbors with edge properties and predicates

        Args:
            node_idx: Node index to get neighbors for
            predicate_filter: Optional predicate string to filter edges

        Returns:
            List of (neighbor_idx, predicate_str, edge_props) tuples
        """
        start = self.fwd_offsets[node_idx]
        end = self.fwd_offsets[node_idx + 1]

        neighbors = self.fwd_targets[start:end]
        pred_ids = self.fwd_predicates[start:end]
        props = self.edge_properties[start:end]

        # Convert predicate IDs to strings
        predicates = [self.id_to_predicate[int(pid)] for pid in pred_ids]

        result = list(zip(neighbors, predicates, props))

        # Apply predicate filter if specified
        if predicate_filter is not None:
            result = [
                (neighbor, pred, prop)
                for neighbor, pred, prop in result
                if pred in predicate_filter
            ]

        return result

    def incoming_neighbors_with_properties(
        self, node_idx, predicate_filter: Optional[list] = None
    ):
        """
        Get incoming neighbors with edge properties and predicates.

        Returns:
            List of (source_idx, predicate_str, edge_props) tuples
        """
        start = self.rev_offsets[node_idx]
        end = self.rev_offsets[node_idx + 1]

        sources = self.rev_sources[start:end]
        pred_ids = self.rev_predicates[start:end]

        # Get properties from forward edge property list
        result = []
        for src_idx, pred_id in zip(sources, pred_ids):
            predicate = self.id_to_predicate[int(pred_id)]
            # Look up properties from the forward edge
            edge_idx = self._find_edge_index(int(src_idx), node_idx, int(pred_id))
            props = self.edge_properties[edge_idx] if edge_idx is not None else {}

            if predicate_filter is None or predicate in predicate_filter:
                result.append((int(src_idx), predicate, props))

        return result

    def get_edges(self, node_idx):
        """
        Get all edges from a node as (neighbor_idx, predicate_str) tuples

        Args:
            node_idx: Source node index

        Returns:
            List of (neighbor_idx, predicate_str) tuples
        """
        start = self.fwd_offsets[node_idx]
        end = self.fwd_offsets[node_idx + 1]

        neighbors = self.fwd_targets[start:end]
        pred_ids = self.fwd_predicates[start:end]

        return [
            (int(neighbor), self.id_to_predicate[int(pred_id)])
            for neighbor, pred_id in zip(neighbors, pred_ids)
        ]

    def get_incoming_edges(self, node_idx):
        """Get all incoming edges to a node as (source_idx, predicate_str) tuples."""
        start = self.rev_offsets[node_idx]
        end = self.rev_offsets[node_idx + 1]

        sources = self.rev_sources[start:end]
        pred_ids = self.rev_predicates[start:end]

        return [
            (int(source), self.id_to_predicate[int(pred_id)])
            for source, pred_id in zip(sources, pred_ids)
        ]

    def get_edge_property(self, src_idx, dst_idx, predicate, key, default=None):
        """
        Get a specific property for an edge - O(degree) lookup via CSR scan

        Args:
            src_idx: Source node index
            dst_idx: Destination node index
            predicate: Predicate string
            key: Property key to retrieve
            default: Default value if not found

        Returns:
            Property value or default
        """
        pred_id = self.predicate_to_idx.get(predicate)
        if pred_id is None:
            return default

        edge_idx = self._find_edge_index(src_idx, dst_idx, pred_id)
        if edge_idx is None:
            return default

        # Predicate is stored in CSR arrays, not in edge properties
        if key == "predicate":
            return predicate

        # Use efficient single-field access when available
        if isinstance(self.edge_properties, EdgePropertyStore):
            return self.edge_properties.get_field(edge_idx, key, default)
        return self.edge_properties[edge_idx].get(key, default)

    def get_all_edge_properties(self, src_idx, dst_idx, predicate):
        """
        Get all properties for an edge - O(degree) lookup via CSR scan

        Args:
            src_idx: Source node index
            dst_idx: Destination node index
            predicate: Predicate string

        Returns:
            Dict of all edge properties, or empty dict if edge not found.
            Always includes 'predicate' key (synthesized from CSR arrays).
        """
        pred_id = self.predicate_to_idx.get(predicate)
        if pred_id is None:
            return {}

        edge_idx = self._find_edge_index(src_idx, dst_idx, pred_id)
        if edge_idx is None:
            return {}

        # EdgePropertyStore.__getitem__ creates a new dict each time,
        # so it's safe to add predicate to it without mutating the store
        props = self.edge_properties[edge_idx]
        props["predicate"] = predicate
        return props

    def get_all_edges_between(self, src_idx, dst_idx, predicate_filter: Optional[list] = None):
        """
        Get all edges (with different predicates) between two nodes

        Args:
            src_idx: Source node index
            dst_idx: Destination node index

        Returns:
            List of (predicate_str, properties) tuples
        """
        t0 = time.perf_counter()
        start = self.fwd_offsets[src_idx]
        end = self.fwd_offsets[src_idx + 1]

        result = []
        neighbors = self.fwd_targets[start:end]
        pred_ids = self.fwd_predicates[start:end]
        props = self.edge_properties[start:end]

        for neighbor, pred_id, prop in zip(neighbors, pred_ids, props):
            if neighbor == dst_idx:
                predicate_str = self.id_to_predicate[int(pred_id)]
                # Apply predicate filter if specified
                if predicate_filter is None or predicate_str in predicate_filter:
                    result.append((predicate_str, prop))

        t1 = time.perf_counter()
        return result

    def get_node_property(self, node_idx, key, default=None):
        """Get a specific property for a node"""
        return self.node_properties.get(node_idx, {}).get(key, default)

    def get_all_node_properties(self, node_idx):
        """Get all properties for a node as a dict"""
        return self.node_properties.get(node_idx, {})

    def degree(self, node_idx, predicate_filter=None):
        """
        Get degree of a node, optionally filtered by predicate

        Args:
            node_idx: Node index
            predicate_filter: Optional predicate string to count only specific edge types

        Returns:
            Integer degree count
        """
        if predicate_filter is None:
            return self.fwd_offsets[node_idx + 1] - self.fwd_offsets[node_idx]
        else:
            return len(self.neighbors(node_idx, predicate_filter=predicate_filter))

    def get_node_idx(self, node_id):
        """Convert original node ID to internal index"""
        return self.node_id_to_idx.get(node_id)

    def get_node_id(self, node_idx):
        """Convert internal index to original node ID"""
        return self.idx_to_node_id.get(node_idx)

    def get_predicate_stats(self):
        """Get statistics about predicate usage"""
        pred_counts = {}
        for pred_id in self.fwd_predicates:
            pred_str = self.id_to_predicate[int(pred_id)]
            pred_counts[pred_str] = pred_counts.get(pred_str, 0) + 1

        return sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)

    def save(self, filepath):
        """Save graph to disk for fast reloading"""
        print(f"Saving graph to {filepath}...")
        data = {
            "num_nodes": self.num_nodes,
            "node_id_to_idx": self.node_id_to_idx,
            "predicate_to_idx": self.predicate_to_idx,
            # Forward CSR
            "fwd_targets": self.fwd_targets,
            "fwd_predicates": self.fwd_predicates,
            "fwd_offsets": self.fwd_offsets,
            # Reverse CSR
            "rev_sources": self.rev_sources,
            "rev_predicates": self.rev_predicates,
            "rev_offsets": self.rev_offsets,
            # Properties
            "edge_properties": self.edge_properties,
            "node_properties": self.node_properties,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Graph saved!")

    @staticmethod
    def load(filepath):
        """Load graph from disk."""
        print(f"Loading graph from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        graph = CSRGraph.__new__(CSRGraph)
        graph.num_nodes = data["num_nodes"]
        graph.node_id_to_idx = data["node_id_to_idx"]
        graph.idx_to_node_id = {idx: nid for nid, idx in graph.node_id_to_idx.items()}
        graph.predicate_to_idx = data["predicate_to_idx"]
        graph.id_to_predicate = {
            idx: pred for pred, idx in graph.predicate_to_idx.items()
        }

        # Forward CSR
        graph.fwd_targets = data["fwd_targets"]
        graph.fwd_predicates = data["fwd_predicates"]
        graph.fwd_offsets = data["fwd_offsets"]

        # Reverse CSR (with backward compatibility)
        if "rev_sources" in data:
            graph.rev_sources = data["rev_sources"]
            graph.rev_predicates = data["rev_predicates"]
            graph.rev_offsets = data["rev_offsets"]
        else:
            print("Warning: Loaded graph without reverse CSR. Rebuilding...")
            # Rebuild reverse CSR from forward edges
            edges = []
            edge_preds = []
            for src in range(graph.num_nodes):
                start = graph.fwd_offsets[src]
                end = graph.fwd_offsets[src + 1]
                for i in range(start, end):
                    dst = graph.fwd_targets[i]
                    pred = graph.fwd_predicates[i]
                    edges.append((src, dst))
                    edge_preds.append(pred)
            graph._build_reverse_csr(
                np.array([e[0] for e in edges], dtype=np.int32),
                np.array([e[1] for e in edges], dtype=np.int32),
                np.array(edge_preds, dtype=np.int32),
            )

        # Properties
        graph.edge_properties = data["edge_properties"]
        graph.node_properties = data["node_properties"]

        # Convert old list-of-dicts format to EdgePropertyStore
        if isinstance(graph.edge_properties, list):
            print("  Converting edge properties to deduplicated format...")
            graph.edge_properties = EdgePropertyStore.from_property_list(
                graph.edge_properties
            )

        if isinstance(graph.edge_properties, EdgePropertyStore):
            stats = graph.edge_properties.dedup_stats()
            print(f"  Edge property dedup: {stats['total_edges']:,} edges -> "
                  f"{stats['unique_publications']:,} unique publication lists, "
                  f"{stats['unique_sources']:,} unique source configs, "
                  f"{stats['unique_qualifiers']:,} unique qualifier combos")

        print(
            f"Graph loaded! {graph.num_nodes:,} nodes, {len(graph.fwd_targets):,} edges"
        )
        print(f"  Unique predicates: {len(graph.predicate_to_idx):,}")
        return graph

    def save_mmap(self, directory: Union[str, Path]):
        """
        Save graph in memory-mappable format for fast loading.

        Creates a directory with separate files:
        - NumPy arrays as .npy files (can be memory-mapped)
        - Metadata dictionaries as pickle
        - Edge property index arrays as .npy files (memory-mapped)
        - Edge property intern pools as small pickle

        Args:
            directory: Directory path to save graph files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        print(f"Saving graph to {directory} (mmap format)...")
        t0 = time.perf_counter()

        # Save NumPy arrays as .npy files (memory-mappable)
        np.save(directory / "fwd_targets.npy", self.fwd_targets)
        np.save(directory / "fwd_predicates.npy", self.fwd_predicates)
        np.save(directory / "fwd_offsets.npy", self.fwd_offsets)
        np.save(directory / "rev_sources.npy", self.rev_sources)
        np.save(directory / "rev_predicates.npy", self.rev_predicates)
        np.save(directory / "rev_offsets.npy", self.rev_offsets)

        # Save small metadata as pickle
        metadata = {
            "num_nodes": self.num_nodes,
            "node_id_to_idx": self.node_id_to_idx,
            "predicate_to_idx": self.predicate_to_idx,
            "node_properties": self.node_properties,
        }
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save edge properties as mmap-friendly components:
        # - numpy .npy index arrays (memory-mappable)
        # - small pickle with the intern pools
        if isinstance(self.edge_properties, EdgePropertyStore):
            self.edge_properties.save_mmap(directory)
        else:
            # Legacy fallback: pickle the whole thing
            with open(directory / "edge_properties.pkl", "wb") as f:
                pickle.dump(self.edge_properties, f, protocol=pickle.HIGHEST_PROTOCOL)

        t1 = time.perf_counter()
        print(f"Graph saved in {t1 - t0:.2f}s")

        # Print file sizes
        total_size = 0
        for f in directory.iterdir():
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name}: {size / 1024 / 1024:.1f} MB")
        print(f"  Total: {total_size / 1024 / 1024:.1f} MB")

    @staticmethod
    def load_mmap(directory: Union[str, Path], mmap_mode: str = "r"):
        """
        Load graph from memory-mapped format.

        This provides near-instant startup by memory-mapping the large NumPy arrays.
        The OS will page in data on demand, and multiple processes can share
        the same memory pages (great for multi-worker FastAPI).

        Args:
            directory: Directory containing graph files
            mmap_mode: Memory-map mode for NumPy arrays:
                - 'r': Read-only (default, recommended for serving)
                - 'r+': Read-write (allows modification)
                - 'c': Copy-on-write (modifications are private)
                - None: Load fully into memory (no mmap)

        Returns:
            CSRGraph instance with memory-mapped arrays
        """
        directory = Path(directory)
        print(f"Loading graph from {directory} (mmap_mode={mmap_mode})...")
        t0 = time.perf_counter()

        graph = CSRGraph.__new__(CSRGraph)

        # Load NumPy arrays with memory mapping
        graph.fwd_targets = np.load(
            directory / "fwd_targets.npy", mmap_mode=mmap_mode
        )
        graph.fwd_predicates = np.load(
            directory / "fwd_predicates.npy", mmap_mode=mmap_mode
        )
        graph.fwd_offsets = np.load(
            directory / "fwd_offsets.npy", mmap_mode=mmap_mode
        )
        graph.rev_sources = np.load(
            directory / "rev_sources.npy", mmap_mode=mmap_mode
        )
        graph.rev_predicates = np.load(
            directory / "rev_predicates.npy", mmap_mode=mmap_mode
        )
        graph.rev_offsets = np.load(
            directory / "rev_offsets.npy", mmap_mode=mmap_mode
        )

        # Load metadata
        with open(directory / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        graph.num_nodes = metadata["num_nodes"]
        graph.node_id_to_idx = metadata["node_id_to_idx"]
        graph.idx_to_node_id = {idx: nid for nid, idx in graph.node_id_to_idx.items()}
        graph.predicate_to_idx = metadata["predicate_to_idx"]
        graph.id_to_predicate = {
            idx: pred for pred, idx in graph.predicate_to_idx.items()
        }
        graph.node_properties = metadata["node_properties"]

        # Load edge properties - prefer new split mmap format, fall back
        # to legacy single-pickle format
        t_props_start = time.perf_counter()

        if (directory / "edge_property_pools.pkl").exists():
            # New format: mmap'd numpy arrays + small pools pickle
            graph.edge_properties = EdgePropertyStore.load_mmap(
                directory, mmap_mode=mmap_mode
            )
        elif (directory / "edge_properties.pkl").exists():
            # Legacy format: one big pickle (EdgePropertyStore or list)
            with open(directory / "edge_properties.pkl", "rb") as f:
                graph.edge_properties = pickle.load(f)
            if isinstance(graph.edge_properties, list):
                print("  Converting edge properties to deduplicated format...")
                graph.edge_properties = EdgePropertyStore.from_property_list(
                    graph.edge_properties
                )
        else:
            raise FileNotFoundError(
                f"No edge property files found in {directory}"
            )

        t_props_end = time.perf_counter()

        if isinstance(graph.edge_properties, EdgePropertyStore):
            stats = graph.edge_properties.dedup_stats()
            print(f"  Edge property dedup: {stats['total_edges']:,} edges -> "
                  f"{stats['unique_publications']:,} unique publication lists, "
                  f"{stats['unique_sources']:,} unique source configs, "
                  f"{stats['unique_qualifiers']:,} unique qualifier combos")

        t1 = time.perf_counter()
        print(
            f"Graph loaded in {t1 - t0:.2f}s "
            f"(edge_properties: {t_props_end - t_props_start:.2f}s)"
        )
        print(
            f"  {graph.num_nodes:,} nodes, {len(graph.fwd_targets):,} edges, "
            f"{len(graph.predicate_to_idx):,} predicates"
        )

        return graph
