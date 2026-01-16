"""Main Gandalf CSR Graph class."""
import pickle

import numpy as np


class CSRGraph:
    """Compressed Sparse Row graph representation for fast neighbor lookups"""

    def __init__(
        self,
        num_nodes,
        edges,
        node_id_to_idx,
        edge_properties=None,
        node_properties=None,
    ):
        """
        Args:
            num_nodes: Total number of unique nodes
            edges: List of (src_idx, dst_idx) tuples using integer indices
            node_id_to_idx: Dict mapping original node IDs to integer indices
            edge_properties: Dict mapping (src_idx, dst_idx) -> properties dict
            node_properties: Dict mapping node_idx -> properties dict
        """
        self.num_nodes = num_nodes
        self.node_id_to_idx = node_id_to_idx
        self.idx_to_node_id = {idx: nid for nid, idx in node_id_to_idx.items()}
        self.node_properties = node_properties or {}

        # Sort edges by source for CSR construction
        edges_with_props = [
            (src, dst, edge_properties.get((src, dst), {}) if edge_properties else {})
            for src, dst in edges
        ]
        edges_with_props.sort(key=lambda x: (x[0], x[1]))

        # Build CSR structure
        self.edge_dst = np.array(
            [dst for src, dst, _ in edges_with_props], dtype=np.int32
        )
        self.edge_properties = [
            props for _, _, props in edges_with_props
        ]  # Store properties in same order

        # Build fast edge lookup: (src, dst) -> index in edge_properties
        print("Building edge property index...")
        self.edge_prop_index = {}
        for i, (src, dst, _) in enumerate(edges_with_props):
            self.edge_prop_index[(src, dst)] = i

        # Offsets array: where each node's edges start
        self.offsets = np.zeros(num_nodes + 1, dtype=np.int64)

        if len(edges_with_props) > 0:
            current_src = 0
            for i, (src, dst, _) in enumerate(edges_with_props):
                # Fill gaps for nodes with no outgoing edges
                while current_src < src:
                    self.offsets[current_src + 1] = i
                    current_src += 1
                self.offsets[src + 1] = i + 1

            # Fill remaining offsets for nodes at the end with no edges
            for i in range(current_src + 1, num_nodes + 1):
                self.offsets[i] = len(edges_with_props)

    def neighbors(self, node_idx):
        """Get neighbor indices for a node index - returns numpy array"""
        start = self.offsets[node_idx]
        end = self.offsets[node_idx + 1]
        return self.edge_dst[start:end]

    def neighbors_with_properties(self, node_idx):
        """Get neighbors with edge properties - returns list of (neighbor_idx, edge_props)"""
        start = self.offsets[node_idx]
        end = self.offsets[node_idx + 1]
        neighbors = self.edge_dst[start:end]
        props = self.edge_properties[start:end]
        return list(zip(neighbors, props))

    def get_edge_property(self, src_idx, dst_idx, key, default=None):
        """Get a specific property for an edge - O(1) lookup"""
        edge_idx = self.edge_prop_index.get((src_idx, dst_idx))
        if edge_idx is None:
            return default
        return self.edge_properties[edge_idx].get(key, default)

    def get_node_property(self, node_idx, key, default=None):
        """Get a specific property for a node"""
        return self.node_properties.get(node_idx, {}).get(key, default)

    def degree(self, node_idx):
        """Get degree of a node"""
        return self.offsets[node_idx + 1] - self.offsets[node_idx]

    def get_node_idx(self, node_id):
        """Convert original node ID to internal index"""
        return self.node_id_to_idx.get(node_id)

    def get_node_id(self, node_idx):
        """Convert internal index to original node ID"""
        return self.idx_to_node_id.get(node_idx)

    def save(self, filepath):
        """Save graph to disk for fast reloading"""
        print(f"Saving graph to {filepath}...")
        data = {
            "num_nodes": self.num_nodes,
            "node_id_to_idx": self.node_id_to_idx,
            "edge_dst": self.edge_dst,
            "edge_properties": self.edge_properties,
            "node_properties": self.node_properties,
            "offsets": self.offsets,
            "edge_prop_index": self.edge_prop_index,  # Save the index too
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Graph saved!")

    @staticmethod
    def load(filepath):
        """Load graph from disk"""
        print(f"Loading graph from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Reconstruct graph object
        graph = CSRGraph.__new__(CSRGraph)
        graph.num_nodes = data["num_nodes"]
        graph.node_id_to_idx = data["node_id_to_idx"]
        graph.idx_to_node_id = {idx: nid for nid, idx in graph.node_id_to_idx.items()}
        graph.edge_dst = data["edge_dst"]
        graph.edge_properties = data["edge_properties"]
        graph.node_properties = data["node_properties"]
        graph.offsets = data["offsets"]

        # Rebuild edge property index if not in saved data (for backward compatibility)
        if "edge_prop_index" in data:
            graph.edge_prop_index = data["edge_prop_index"]
        else:
            print("Building edge property index...")
            graph.edge_prop_index = {}
            for i in range(len(graph.edge_dst)):
                # Find which source node this edge belongs to
                src = np.searchsorted(graph.offsets, i, side="right") - 1
                dst = graph.edge_dst[i]
                graph.edge_prop_index[(src, dst)] = i

        print(f"Graph loaded! {graph.num_nodes:,} nodes, {len(graph.edge_dst):,} edges")
        return graph
