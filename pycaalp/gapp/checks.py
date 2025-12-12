"""All checks needed for the AssemblyDigraph class"""

from math import factorial
import networkx as nx
import numpy as np


def has_technology_changes(technology_weights: list) -> int:
    """
    Checks if there are technology changes in the given list of weights.

    Args:
        technology_weights: A list of weights representing the
            technology of the edges.

    Returns:
        True if there are technology changes, False otherwise.
    """
    num_tech_changes = len(set(technology_weights))
    return num_tech_changes


def count_num_different_technologies(technology_weights: list) -> int:
    """
    TODO: remove if not needed
    Counts the number of different technologies in the given list of weights.

    Args:
        technology_weights: A list of weights representing the
            technology of the edges.

    Returns:
        The number of different technologies.
    """
    tech_list = [technology_weights[0]]
    for value in technology_weights:
        if value not in tech_list:
            tech_list.append(value)
    return len(tech_list) - 1


def create_tech_list_with_occurences(technology_values: list):
    tech_occur = {}
    for value in technology_values:
        if value not in tech_occur.keys():
            tech_occur[value] = 1
        else:
            tech_occur[value] += 1

    return tech_occur


def find_min_diff(technology_occur: dict):
    max_key = max(technology_occur, key=technology_occur.get)
    min_diff = float("-inf")
    for tech, occurs in technology_occur.items():
        if tech != max_key:
            curr_min_diff = technology_occur[max_key] - occurs
            min_diff = min(curr_min_diff, min_diff)
    if min_diff == float("-inf"):
        return 0
    return min_diff


def check_technology_changes(technology_values: dict, new_edge_technology: str):
    tech_occur = create_tech_list_with_occurences(list(technology_values.values()))
    min_diff_prev = find_min_diff(tech_occur)
    if new_edge_technology not in tech_occur:
        tech_occur[new_edge_technology] = 1
    else:
        tech_occur[new_edge_technology] += 1
    min_diff_after = find_min_diff(tech_occur)
    if min_diff_after < min_diff_prev:
        return 1
    return 0


def check_common_node(graph_edges: list) -> bool:
    """
    Checks if there is a common node between the edges of a graph.

    Args:
        graph_edges: A list with the edges of a graph.

    Returns:
        True if there is a common node, False otherwise.
    """
    # Get a list of all edges in the graph
    edges = list(graph_edges)

    # Find the common nodes on all edges
    common_nodes = set(edges[0])  # Initialize with the nodes of the first edge
    for edge in edges[1:]:
        common_nodes.intersection_update(edge)
    return len(common_nodes) > 0


def check_num_subgraphs_and_one_assembly_policy(
    init_graph: nx.Graph, temp_graph: nx.Graph, check_subgraphs: bool = True
) -> bool:
    """
    Checks if the number of subgraphs is equal to the number of subgraphs of the
    initial graph + 1 and if there is only one assembly policy.

    Args:
        init_graph: The initial graph.
        temp_graph: The graph to be checked.
        check_subgraphs: Whether to check the number of subgraphs.
            Defaults to True.

    Returns:
        True if the number of subgraphs is equal to the number of subgraphs of the
            initial graph + 1 and if there is only one assembly policy, False otherwise.
    """
    # Important: first check if there are layer+1 subgraphs meaning that the
    # right amount of parts is disassembled
    sub_graphs = np.array(list(nx.connected_components(temp_graph)))
    _num_subgraphs = len(sub_graphs)

    # One assembly policy check
    sub_graphs_lengths = np.vectorize(len)(sub_graphs)
    num_connected_subgraphs = len(np.where(sub_graphs_lengths > 1)[0])

    if check_subgraphs:
        num_init_graph_subgraphs = len(
            np.array(list(nx.connected_components(init_graph)))
        )
        return (
            _num_subgraphs == num_init_graph_subgraphs + 1
            and num_connected_subgraphs == 1
        )
    return num_connected_subgraphs == 1


def check_one_assembly_policy(graph: nx.Graph) -> bool:
    """
    Checks if there is only one assembly policy in the graph.

    Args:
        graph: The graph to be checked.

    Returns:
        True if there is only one assembly policy, False otherwise.
    """
    sub_graphs = np.array(list(nx.connected_components(graph)))
    sub_graphs_lengths = np.vectorize(len)(sub_graphs)
    num_connected_subgraphs = len(np.where(sub_graphs_lengths > 1)[0])
    return num_connected_subgraphs in [0, 1]


def find_max_edges_connected_per_node(graph: nx.Graph) -> int:
    """
    Finds the maximum number of edges connected to a node in the graph.

    Args:
        graph: The graph to be checked.

    Returns:
        The maximum number of edges connected to a node in the graph.
    """
    _max = -1
    for node in graph.nodes:
        _max = max(_max, len(graph.edges(node)))
    return _max


def print_assembly_states_length_per_layer(assembly_state: list) -> None:
    """
    Prints the length of the assembly states per layer.

    Args:
        assembly_state: A list with the assembly states.
    """
    print("Assembly states length per layer:")
    for key, val in assembly_state.items():
        print(f"Layer {key}: {len(val)}")
    print()


def dict_assembly_digraph_length_per_layer(
    assembly_digraph: nx.DiGraph, print_results: bool = False
) -> dict:
    """
    Prints the length of the assembly digraph per layer.

    Args:
        assembly_digraph: A directed graph with the assembly states.
        print_results: If True, prints the total number of nodes and edges.

    Returns:
        A dictionary with nodes per layer.
    """
    layer_dict = {}
    for node in assembly_digraph.nodes():
        layer = int(node.split("_")[0])
        if not layer_dict.get(layer):
            layer_dict[layer] = 0
        else:
            layer_dict[layer] += 1

    if print_results:
        print("Assembly digraph length per layer:")
        print(f"Total number of nodes: {len(assembly_digraph.nodes())}")
        print("Toal number of nodes per layer:")
        print(layer_dict)
        print(f"Total number of edges: {len(assembly_digraph.edges())}")
    return layer_dict


def dict_assembly_digraph_nodes_per_layer(assembly_digraph: nx.DiGraph) -> dict:
    """
    Returns a dictionary with the nodes per layer.

    Args:
        assembly_digraph: A directed graph with the assembly states.

    Returns:
        A dictionary with the node names per layer of the assembly digraph.
    """
    node_dict = {}
    for node in assembly_digraph.nodes():
        layer = int(node.split("_")[0])
        if not node_dict.get(layer):
            node_dict[layer] = [node]
        else:
            node_dict[layer].append(node)
    return node_dict


def binomial_coeff(y: int, x: int):
    """Using the formula C(y,x)= y!/x!(y-x)!

    Args:
        y: set size (all elements)
        x: element combinations

    Returns:
        binomial coefficient
    """
    # Check if x is within the valid range
    if x < 0 or x > y:
        return "Invalid input: x should be between 0 and y."

    num_combinations = factorial(y) // (factorial(x) * factorial(y - x))
    return num_combinations


def all_cutset_combs(total_edges):
    """Give all the possible combinations for given assembly digraph edges"""
    max_nodes = 0
    max_edge_num = 0
    print("Max possible nodes in layer:")
    for i in range(0, total_edges + 1):
        b_coeff = binomial_coeff(total_edges, i)
        print(f"{i}: {b_coeff}")
        max_nodes += b_coeff
        if i < total_edges:
            max_edge_num += b_coeff * (total_edges - i)

    print(f"Max nodes in assembly digraph: {max_nodes:.3g}")
    print(f"Max edges in assembly digraph: {max_edge_num:.3g}")
    return max_edge_num
