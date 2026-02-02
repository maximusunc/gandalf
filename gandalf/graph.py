"""Main Gandalf CSR Graph class."""

import os
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np


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
        Args:
            num_nodes: Total number of unique nodes
            edges: List of (src_idx, dst_idx) tuples using integer indices
            edge_predicates: List of predicate IDs (parallel to edges list)
            node_id_to_idx: Dict mapping original node IDs to integer indices
            predicate_to_id: Dict mapping predicate strings to integer IDs
            edge_properties: Dict mapping (src_idx, dst_idx, pred_idx) -> properties dict
            node_properties: Dict mapping node_idx -> properties dict
        """
        self.num_nodes = num_nodes
        self.node_id_to_idx = node_id_to_idx
        self.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
        self.node_properties = node_properties or {}

        # Predicate vocabulary
        self.predicate_to_idx = predicate_to_idx
        self.id_to_predicate = {idx: pred for pred, idx in predicate_to_idx.items()}

        # Build both forward and reverse CSR structures
        print("Building forward CSR...")
        self._build_forward_csr(edges, edge_predicates, edge_properties)

        print("Building reverse CSR...")
        self._build_reverse_csr(edges, edge_predicates)

    def _build_forward_csr(self, edges, edge_predicates, edge_properties):
        """Build forward adjacency list (source -> targets)."""
        # Sort edges by source for CSR construction
        edges_with_props = [
            (
                src,
                dst,
                pred_id,
                edge_properties.get((src, dst, pred_id), {}) if edge_properties else {},
            )
            for (src, dst), pred_id in zip(edges, edge_predicates)
        ]
        edges_with_props.sort(key=lambda x: (x[0], x[1], x[2]))

        # Build forward CSR arrays
        self.fwd_targets = np.array(
            [dst for src, dst, _, _ in edges_with_props], dtype=np.int32
        )
        self.fwd_predicates = np.array(
            [pred_id for src, dst, pred_id, _ in edges_with_props], dtype=np.int32
        )
        self.edge_properties = [props for _, _, _, props in edges_with_props]

        # Build edge property index: (src, dst, pred_id) -> index
        self.edge_prop_index = {}
        for i, (src, dst, pred_id, _) in enumerate(edges_with_props):
            self.edge_prop_index[(src, dst, pred_id)] = i

        # Build forward offsets
        self.fwd_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        if len(edges_with_props) > 0:
            current_src = 0
            for i, (src, dst, _, _) in enumerate(edges_with_props):
                while current_src < src:
                    self.fwd_offsets[current_src + 1] = i
                    current_src += 1
                self.fwd_offsets[src + 1] = i + 1

            for i in range(current_src + 1, self.num_nodes + 1):
                self.fwd_offsets[i] = len(edges_with_props)

    def _build_reverse_csr(self, edges, edge_predicates):
        """Build reverse adjacency list (target -> sources)."""
        # Create reverse edges: swap source and destination
        reverse_edges = [
            (dst, src, pred_id) for (src, dst), pred_id in zip(edges, edge_predicates)
        ]
        reverse_edges.sort(key=lambda x: (x[0], x[1], x[2]))

        # Build reverse CSR arrays
        self.rev_sources = np.array(
            [src for dst, src, _ in reverse_edges], dtype=np.int32
        )
        self.rev_predicates = np.array(
            [pred_id for dst, src, pred_id in reverse_edges], dtype=np.int32
        )

        # Build reverse offsets
        self.rev_offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)

        if len(reverse_edges) > 0:
            current_dst = 0
            for i, (dst, src, _) in enumerate(reverse_edges):
                while current_dst < dst:
                    self.rev_offsets[current_dst + 1] = i
                    current_dst += 1
                self.rev_offsets[dst + 1] = i + 1

            for i in range(current_dst + 1, self.num_nodes + 1):
                self.rev_offsets[i] = len(reverse_edges)

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
            edge_idx = self.edge_prop_index.get((int(src_idx), node_idx, int(pred_id)))
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
        Get a specific property for an edge - O(1) lookup

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

        edge_idx = self.edge_prop_index.get((src_idx, dst_idx, pred_id))
        if edge_idx is None:
            return default
        return self.edge_properties[edge_idx].get(key, default)

    def get_all_edge_properties(self, src_idx, dst_idx, predicate):
        """
        Get all properties for an edge - O(1) lookup

        Args:
            src_idx: Source node index
            dst_idx: Destination node index
            predicate: Predicate string

        Returns:
            Dict of all edge properties, or empty dict if edge not found
        """
        pred_id = self.predicate_to_idx.get(predicate)
        if pred_id is None:
            return {}

        edge_idx = self.edge_prop_index.get((src_idx, dst_idx, pred_id))
        if edge_idx is None:
            return {}
        return self.edge_properties[edge_idx]

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
                if predicate_filter is not None and predicate_str in predicate_filter:
                    result.append((predicate_str, prop))

        t1 = time.perf_counter()
        # print("Getting all edges in between took:", t1 - t0)
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
            "edge_prop_index": self.edge_prop_index,
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
            graph._build_reverse_csr(edges, edge_preds)

        # Properties
        graph.edge_properties = data["edge_properties"]
        graph.node_properties = data["node_properties"]
        graph.edge_prop_index = data.get("edge_prop_index", {})

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
        - Edge properties as separate pickle (large, but shared via copy-on-write)

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
            "edge_prop_index": self.edge_prop_index,
        }
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save edge properties separately (large file, shared via copy-on-write)
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
        graph.edge_prop_index = metadata.get("edge_prop_index", {})

        # Load edge properties (large, but will be shared via fork)
        t_props_start = time.perf_counter()
        with open(directory / "edge_properties.pkl", "rb") as f:
            graph.edge_properties = pickle.load(f)
        t_props_end = time.perf_counter()

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
