"""Load nodes and edges into Gandalf."""

import json

import numpy as np

from gandalf.graph import CSRGraph


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
    print(f"Reading edges from {edge_jsonl_path}...")

    # First pass: collect all unique node IDs and edges with properties
    node_ids = set()
    edge_list = []  # List of (subject, predicate, object, properties) tuples
    edge_set = set()  # For duplicate detection: (subject, predicate, object)

    line_count = 0
    duplicates = 0
    self_loops = 0

    with open(edge_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line_count % 1_000_000 == 0 and line_count > 0:
                print(f"  Processed {line_count:,} edges...")

            data = json.loads(line)
            subject = data["subject"]
            obj = data["object"]
            predicate = data["predicate"]

            node_ids.add(subject)
            node_ids.add(obj)

            # Create edge identifier (subject, predicate, object) - predicates make edges unique
            edge_id = (subject, predicate, obj)

            edge_set.add(edge_id)

            # Store edge with its properties
            edge_props = {
                "predicate": predicate,
                # "category": data.get("category", []),
                "publications": data.get("publications", []),
                "sources": [
                   {
                       "resource_role": "primary_knowledge_source",
                       "resource_id": data.get("primary_knowledge_source", "infores:gandalf"),
                   },
                ],
                # "knowledge_level": data.get("knowledge_level", ""),
                # "agent_type": data.get("agent_type", ""),
                # "original_subject": data.get("original_subject", ""),
                # "original_object": data.get("original_object", ""),
                "qualifiers": data.get("qualifiers", []),
            }

            edge_list.append((subject, predicate, obj, edge_props))

            line_count += 1

    print(f"Found {len(node_ids):,} unique nodes and {len(edge_list):,} edges")

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
                        "id": node_data.get("id", ""),
                        "categories": node_data.get("category", []),
                        "name": node_data.get("name", None),
                        "equivalent_identifiers": node_data.get("equivalent_identifiers", []),
                        "information_content": node_data.get("information_content", 0.0),
                    }
        print(f"  Loaded properties for {len(node_props_by_id):,} nodes")

    # Create mapping from node IDs to integer indices
    print("Building node index mapping...")
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(node_ids))}

    # Build predicate vocabulary
    print("Building predicate vocabulary...")
    unique_predicates = sorted(set(pred for _, pred, _, _ in edge_list))
    predicate_to_idx = {pred: idx for idx, pred in enumerate(unique_predicates)}
    print(f"  Found {len(unique_predicates):,} unique predicates")

    # Convert node properties to use indices
    node_properties = {}
    for node_id, props in node_props_by_id.items():
        idx = node_id_to_idx.get(node_id)
        if idx is not None:
            node_properties[idx] = props

    # Convert edges to integer indices with properties
    print("Converting edges to indices...")
    edges = []  # List of (src_idx, dst_idx) tuples
    edge_predicates = []  # Parallel list of predicate IDs
    edge_properties = {}  # Dict: (src_idx, dst_idx, pred_idx) -> properties

    for subject, predicate, obj, props in edge_list:
        src_idx = node_id_to_idx[subject]
        dst_idx = node_id_to_idx[obj]
        pred_idx = predicate_to_idx[predicate]

        edges.append((src_idx, dst_idx))
        edge_predicates.append(pred_idx)
        edge_properties[(src_idx, dst_idx, pred_idx)] = props

    print(f"Total edges: {len(edges):,}")

    # Build CSR structure
    print("Building CSR structure...")
    graph = CSRGraph(
        num_nodes=len(node_ids),
        edges=edges,
        edge_predicates=edge_predicates,
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
        print(f"  Memory usage: ~{memory_mb:.1f} MB")

    return graph
