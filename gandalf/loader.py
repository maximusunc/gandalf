"""Load nodes and edges into Gandalf."""
import json

import numpy as np

from gandalf.graph import CSRGraph


def build_graph_from_jsonl(
    jsonl_path,
    undirected=True,
    remove_duplicates=True,
    remove_self_loops=True,
    node_jsonl_path=None,
):
    """
    Build a CSR graph from a JSONL file of edges.

    Args:
        jsonl_path: Path to JSONL file where each line has 'subject' and 'object' fields
        undirected: If True, treat graph as undirected (add reverse edges)
        remove_duplicates: If True, remove duplicate edges
        remove_self_loops: If True, remove self-loop edges
        node_jsonl_path: Optional path to JSONL file with node properties

    Returns:
        CSRGraph object with the loaded graph
    """
    print(f"Reading edges from {jsonl_path}...")

    # First pass: collect all unique node IDs and edges with properties
    node_ids = set()
    edge_dict = {}  # (subject, object) -> properties

    line_count = 0
    duplicates = 0
    self_loops = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line_count % 1_000_000 == 0 and line_count > 0:
                print(f"  Processed {line_count:,} edges...")

            data = json.loads(line)
            subject = data["subject"]
            obj = data["object"]

            # Check for self-loops
            if subject == obj:
                self_loops += 1
                if remove_self_loops:
                    line_count += 1
                    continue

            node_ids.add(subject)
            node_ids.add(obj)

            # For undirected graphs, normalize edge representation
            if undirected:
                edge = tuple(sorted([subject, obj]))
            else:
                edge = (subject, obj)

            if edge in edge_dict:
                duplicates += 1
                if remove_duplicates:
                    line_count += 1
                    continue

            # Store edge with its predicate
            edge_dict[edge] = {
                "predicate": data.get("predicate", None),
                "publications": data.get("publications", []),
                "knowledge_source": data.get("primary_knowledge_source", None),
            }

            line_count += 1

    print(f"Found {len(node_ids):,} unique nodes and {len(edge_dict):,} unique edges")
    if duplicates > 0:
        print(f"  Removed {duplicates:,} duplicate edges")
    if self_loops > 0:
        print(f"  Removed {self_loops:,} self-loops")

    # Load node properties if provided
    node_props_by_id = {}
    if node_jsonl_path:
        print(f"Reading node properties from {node_jsonl_path}...")
        with open(node_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                node_data = json.loads(line)
                node_id = node_data.get("id")
                if node_id:
                    node_props_by_id[node_id] = {
                        "category": node_data.get("category", []),
                        "name": node_data.get("name", None),
                    }
        print(f"  Loaded properties for {len(node_props_by_id):,} nodes")

    # Create mapping from node IDs to integer indices
    print("Building node index mapping...")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(node_ids))}

    # Convert node properties to use indices
    node_properties = {}
    for node_id, props in node_props_by_id.items():
        idx = node_id_to_idx.get(node_id)
        if idx is not None:
            node_properties[idx] = props

    # Convert edges to integer indices with properties
    print("Converting edges to indices...")
    edges = []
    edge_properties = {}

    for edge, props in edge_dict.items():
        if undirected:
            subject, obj = edge
        else:
            subject, obj = edge

        src_idx = node_id_to_idx[subject]
        dst_idx = node_id_to_idx[obj]

        edges.append((src_idx, dst_idx))
        edge_properties[(src_idx, dst_idx)] = props

        # Add reverse edge for undirected graph
        if undirected and src_idx != dst_idx:
            edges.append((dst_idx, src_idx))
            # Reverse edge has same properties
            edge_properties[(dst_idx, src_idx)] = props

    print(f"Total edges (with reverse if undirected): {len(edges):,}")

    # Build CSR structure
    print("Building CSR structure...")
    graph = CSRGraph(
        len(node_ids), edges, node_id_to_idx, edge_properties, node_properties
    )

    # Print statistics
    degrees = [graph.degree(i) for i in range(min(1000, graph.num_nodes))]
    if degrees:
        print("\nGraph statistics:")
        print(f"  Nodes: {graph.num_nodes:,}")
        print(f"  Edges: {len(graph.edge_dst):,}")
        print(f"  Avg degree (sampled): {np.mean(degrees):.1f}")
        print(f"  Max degree (sampled): {np.max(degrees)}")
        print(
            f"  Memory usage: ~{(graph.edge_dst.nbytes + graph.offsets.nbytes) / 1024 / 1024:.1f} MB"
        )

    return graph
