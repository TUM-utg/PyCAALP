"""Operations to measure assembly digraph paths
"""

import networkx as nx


def calculate_num_simple_paths(assem_dig: "AssemblyDigraph") -> int:
    """Calculate all simple paths from layer 0 to the final layer

    Args:
        assem_dig: AssemblyDigraph, including a constructed graph.

    Returns:
        Number of all simple paths from starting node to the final one.
    """
    final_layer = assem_dig.graph.number_of_edges()
    all_simple_paths = nx.all_simple_paths(
        assem_dig.assembly_digraph,
        source="0",
        target=f"{final_layer}_1",
    )
    num_all_simple_paths = 0
    for _ in all_simple_paths:
        num_all_simple_paths += 1
    return num_all_simple_paths


def calculate_sum_of_sh_path_weights(cutsets):
    sh_path = nx.shortest_path(
        cutsets.assembly_digraph,
        source="0_1",
        target=f"{cutsets.get_num_layers-1}_1",
        weight="edge_weight",
    )
    edge_weights = nx.get_edge_attributes(cutsets.assembly_digraph, "edge_weight")
    edge_sum = 0
    for i in range(len(sh_path) - 1):
        edge_sum += edge_weights[(sh_path[i], sh_path[i + 1])]

    return edge_sum
