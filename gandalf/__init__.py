"""
Gandalf - Fast 3-hop path finding in large knowledge graphs
"""

__version__ = "0.1.0"

from gandalf.diagnostics import (
    analyze_node_types,
    analyze_predicates,
    diagnose_path_explosion,
)
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.query_planner import get_next_qedge
from gandalf.search import (
    find_3hop_paths_filtered,
    find_3hop_paths_with_properties,
    find_mechanistic_paths,
    lookup,
)

__all__ = [
    # Core classes
    "CSRGraph",
    # Loading
    "build_graph_from_jsonl",
    # Search
    "find_3hop_paths_filtered",
    "find_3hop_paths_with_properties",
    "find_mechanistic_paths",
    "lookup",
    # Diagnostics
    "diagnose_path_explosion",
    "analyze_node_types",
    "analyze_predicates",
    # Query Planner
    "get_next_qedge",
]
