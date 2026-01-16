"""General Diagnostic tools for different path types."""
from collections import Counter, defaultdict

import numpy as np

from gandalf.graph import CSRGraph


def diagnose_path_explosion(graph: CSRGraph, start_id, end_id):
    """
    Diagnose why so many paths exist between two nodes.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    print("=== PATH EXPLOSION DIAGNOSIS ===")
    print(f"Start: {start_id}")
    print(f"End: {end_id}")
    print()

    # 1. Check degrees
    start_deg = graph.degree(start_idx)
    end_deg = graph.degree(end_idx)

    print("1. DEGREE ANALYSIS")
    print(f"   Start node degree: {start_deg:,}")
    print(f"   End node degree: {end_deg:,}")
    print(
        f"   Naive estimate (deg_start × avg_deg × deg_end): ~{start_deg * 20 * end_deg:,} paths"
    )
    print()

    # 2. Analyze 1-hop neighborhoods
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors = graph.neighbors(end_idx)

    print("2. ONE-HOP NEIGHBORHOODS")
    print(f"   Start has {len(start_neighbors):,} direct neighbors")
    print(f"   End has {len(end_neighbors):,} direct neighbors")

    # Check for direct connection (2-hop would be even more explosive)
    if end_idx in start_neighbors:
        print("   ⚠️  WARNING: Direct edge exists between start and end!")
    print()

    # 3. Analyze 2-hop reachability
    print("3. TWO-HOP ANALYSIS FROM START")
    two_hop_nodes = set()
    hop1_to_hop2_count = defaultdict(int)

    for n1_idx in start_neighbors[: min(100, len(start_neighbors))]:  # Sample first 100
        n1_neighbors = graph.neighbors(n1_idx)
        hop1_to_hop2_count[n1_idx] = len(n1_neighbors)
        for n2_idx in n1_neighbors:
            two_hop_nodes.add(n2_idx)

    avg_2hop_fanout = (
        np.mean(list(hop1_to_hop2_count.values())) if hop1_to_hop2_count else 0
    )
    print(f"   Nodes reachable in 2 hops (sampled): {len(two_hop_nodes):,}")
    print(f"   Average fanout at hop 2: {avg_2hop_fanout:.1f}")
    print(
        f"   Estimated full 2-hop reachable: ~{len(start_neighbors) * avg_2hop_fanout:,.0f}"
    )
    print()

    # 4. Check overlap between 2-hop from start and 1-hop from end
    print("4. MIDDLE NODE OVERLAP")
    overlap_count = sum(1 for node in two_hop_nodes if node in end_neighbors)
    overlap_pct = (overlap_count / len(two_hop_nodes) * 100) if two_hop_nodes else 0
    print(f"   Nodes that connect start (2-hop) to end (1-hop): {overlap_count:,}")
    print(f"   Overlap percentage: {overlap_pct:.1f}%")
    print()

    # 5. Path multiplicity analysis
    print("5. PATH MULTIPLICITY")
    print("   Analyzing how many ways to reach middle nodes...")

    # Count how many 1-hop nodes lead to each 2-hop node
    two_hop_incoming = defaultdict(int)
    for n1_idx in start_neighbors[: min(1000, len(start_neighbors))]:
        for n2_idx in graph.neighbors(n1_idx):
            two_hop_incoming[n2_idx] += 1

    multiplicities = []
    if two_hop_incoming:
        multiplicities = list(two_hop_incoming.values())
        print(f"   Average ways to reach a 2-hop node: {np.mean(multiplicities):.1f}")
        print(f"   Max ways to reach a 2-hop node: {np.max(multiplicities)}")
        print(f"   Median: {np.median(multiplicities):.1f}")

        # Show distribution
        mult_dist = Counter(multiplicities)
        print("   Distribution of multiplicities:")
        for mult in sorted(mult_dist.keys())[:10]:
            print(f"      {mult} paths to node: {mult_dist[mult]:,} nodes")
    print()

    # 6. Compute actual path count with formula
    print("6. ACTUAL PATH COUNT CALCULATION")
    actual_paths = compute_path_count_fast(graph, start_idx, end_idx)
    print("   Actual 3-hop paths: {actual_paths:,}")
    print()

    # 7. Find heaviest contributors
    print("7. HEAVIEST MIDDLE NODES (Top 10)")
    middle_node_contributions = defaultdict(int)

    # For each potential middle node (2-hop from start, 1-hop from end)
    end_neighbors_set = set(end_neighbors)

    for n1_idx in start_neighbors[: min(1000, len(start_neighbors))]:
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx in end_neighbors_set:
                middle_node_contributions[n2_idx] += 1

    top_contributors = sorted(
        middle_node_contributions.items(), key=lambda x: x[1], reverse=True
    )[:10]

    for rank, (node_idx, count) in enumerate(top_contributors, 1):
        node_id = graph.get_node_id(node_idx)
        node_deg = graph.degree(node_idx)
        print(f"   {rank}. {node_id}")
        print(f"      Contributes {count:,} paths (degree: {node_deg:,})")
    print()

    # 8. Recommendations
    print("8. RECOMMENDATIONS")
    if start_deg > 1000 or end_deg > 1000:
        print("   ⚠️  High-degree nodes detected!")
        print("      Consider filtering edges by predicate type before querying")

    if avg_2hop_fanout > 50:
        print("   ⚠️  High fanout at hop 2!")
        print("      Many hub nodes in the path - consider constraining middle nodes")

    if overlap_pct > 50:
        print("   ℹ️  High overlap suggests these nodes are well-connected")
        print("      Paths may be redundant - consider deduplication by middle nodes")

    return {
        "start_degree": start_deg,
        "end_degree": end_deg,
        "two_hop_reachable": len(two_hop_nodes),
        "middle_node_overlap": overlap_count,
        "avg_multiplicity": np.mean(multiplicities) if two_hop_incoming else 0,
        "total_paths": actual_paths,
        "top_contributors": top_contributors,
    }


def compute_path_count_fast(graph: CSRGraph, start_idx, end_idx):
    """
    Compute exact 3-hop path count without enumerating them all.
    Much faster than generating all paths.
    """
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    path_count = 0
    for n1_idx in start_neighbors:
        if n1_idx == end_idx:
            continue
        for n2_idx in graph.neighbors(n1_idx):
            if n2_idx == start_idx:
                continue
            if n2_idx in end_neighbors_set:
                path_count += 1

    return path_count


def analyze_node_types(graph: CSRGraph, start_id, end_id, max_sample=1000):
    """
    Analyze what types of nodes appear in the paths.
    Helps understand if certain node types are causing explosion.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    print("=== NODE TYPE ANALYSIS ===")

    # Sample some paths
    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    hop1_categories = []
    hop2_categories = []

    sampled = 0
    for n1_idx in start_neighbors:
        if sampled >= max_sample:
            break
        n1_cat = graph.get_node_property(n1_idx, "category", "")

        for n2_idx in graph.neighbors(n1_idx):
            if sampled >= max_sample:
                break
            if n2_idx == start_idx or n2_idx not in end_neighbors_set:
                continue

            n2_cat = graph.get_node_property(n2_idx, "category", "")

            hop1_categories.append(n1_cat)
            hop2_categories.append(n2_cat)
            sampled += 1

    print(f"Sampled {sampled:,} paths")
    print()

    print("Top categories at HOP 1 (from start):")
    for cat, count in Counter(hop1_categories).most_common(10):
        print(f"  {cat}: {count:,}")
    print()

    print("Top categories at HOP 2 (to end):")
    for cat, count in Counter(hop2_categories).most_common(10):
        print(f"  {cat}: {count:,}")
    print()


def analyze_predicates(graph: CSRGraph, start_id, end_id, max_sample=1000):
    """
    Analyze what predicates appear in the paths.
    Helps identify if certain relationship types dominate.
    """
    start_idx = graph.get_node_idx(start_id)
    end_idx = graph.get_node_idx(end_id)

    print("=== PREDICATE ANALYSIS ===")

    start_neighbors = graph.neighbors(start_idx)
    end_neighbors_set = set(graph.neighbors(end_idx))

    edge1_predicates = []
    edge2_predicates = []
    edge3_predicates = []

    sampled = 0
    for n1_idx in start_neighbors:
        if sampled >= max_sample:
            break

        pred1 = graph.get_edge_property(start_idx, n1_idx, "predicate")

        for n2_idx in graph.neighbors(n1_idx):
            if sampled >= max_sample:
                break
            if n2_idx == start_idx or n2_idx not in end_neighbors_set:
                continue

            pred2 = graph.get_edge_property(n1_idx, n2_idx, "predicate")
            pred3 = graph.get_edge_property(n2_idx, end_idx, "predicate")

            edge1_predicates.append(pred1)
            edge2_predicates.append(pred2)
            edge3_predicates.append(pred3)
            sampled += 1

    print(f"Sampled {sampled:,} paths")
    print()

    print("Top predicates at EDGE 1 (start → hop1):")
    for pred, count in Counter(edge1_predicates).most_common(10):
        print(f"  {pred}: {count:,}")
    print()

    print("Top predicates at EDGE 2 (hop1 → hop2):")
    for pred, count in Counter(edge2_predicates).most_common(10):
        print(f"  {pred}: {count:,}")
    print()

    print("Top predicates at EDGE 3 (hop2 → end):")
    for pred, count in Counter(edge3_predicates).most_common(10):
        print(f"  {pred}: {count:,}")
    print()

    # Predicate combinations
    print("Top predicate PATTERNS (edge1 → edge2 → edge3):")
    patterns = [
        f"{e1} → {e2} → {e3}"
        for e1, e2, e3 in zip(edge1_predicates, edge2_predicates, edge3_predicates)
    ]
    for pattern, count in Counter(patterns).most_common(10):
        print(f"  {pattern}: {count:,}")
