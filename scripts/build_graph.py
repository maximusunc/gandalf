#!/usr/bin/env python3
"""
CLI tool to build knowledge graphs from JSONL files.

Example:
    kg-build --edges data/edges.jsonl --output data/graph.pkl
"""

import argparse
import sys
from pathlib import Path

from gandalf import build_graph_from_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  kg-build --edges edges.jsonl --output graph.pkl

  # With node properties
  kg-build --edges edges.jsonl --nodes nodes.jsonl --output graph.pkl

  # Exclude ontology hierarchies
  kg-build --edges edges.jsonl --output graph.pkl \\
      --exclude-predicates biolink:subclass_of biolink:related_to

  # Build directed graph
  kg-build --edges edges.jsonl --output graph.pkl --directed
        """,
    )

    parser.add_argument(
        "--edges", required=True, type=Path, help="Path to edges JSONL file"
    )

    parser.add_argument(
        "--nodes", type=Path, help="Path to nodes JSONL file (optional)"
    )

    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output path for pickled graph"
    )

    parser.add_argument(
        "--exclude-predicates",
        nargs="+",
        help="Predicates to exclude (e.g., biolink:subclass_of)",
    )

    parser.add_argument(
        "--directed",
        action="store_true",
        help="Build directed graph (default: undirected)",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate edges (default: remove)",
    )

    parser.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-loop edges (default: remove)",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.edges.exists():
        print(f"Error: Edge file not found: {args.edges}", file=sys.stderr)
        sys.exit(1)

    if args.nodes and not args.nodes.exists():
        print(f"Error: Node file not found: {args.nodes}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build graph
    print(f"Building graph from {args.edges}")
    if args.nodes:
        print(f"Loading node properties from {args.nodes}")

    try:
        graph = build_graph_from_jsonl(
            jsonl_path=str(args.edges),
            undirected=not args.directed,
            remove_duplicates=not args.keep_duplicates,
            remove_self_loops=not args.keep_self_loops,
            node_jsonl_path=str(args.nodes) if args.nodes else None,
        )

        # Save graph
        print(f"\nSaving graph to {args.output}")
        graph.save(str(args.output))

        print("\n✓ Graph built successfully!")
        print(f"  Nodes: {graph.num_nodes:,}")
        print(f"  Edges: {len(graph.edge_dst):,}")

    except Exception as e:
        print(f"Error building graph: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
