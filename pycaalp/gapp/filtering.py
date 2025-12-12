"""Filtering functions for the time balancing algorithm."""

import random
import re
import pandas as pd
import numpy as np
import networkx as nx


def pick_random_percentage(
    input_list: list, percentage: int, protected_elements: list = None
) -> list:
    """Pick a percentage of elements randomly from a list.

    Args:
        input_list: The list to pick elements from.
        percentage: The percentage of elements to pick.
        protected_elements: Elements to protect, i.e., can't be picked

    Returns:
        A list containing the randomly picked elements.
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Iterate on list and remove the sets which first element is in the protected elements
    if protected_elements is not None:
        for i in range(len(input_list) - 1, -1, -1):
            if input_list[i][0] in protected_elements:
                del input_list[i]

    num_elements_to_pick = int(len(input_list) * (percentage / 100))
    return random.sample(input_list, num_elements_to_pick)


def filter_assembly_digraph_edges(
    assembly_digraph: nx.DiGraph,
    filter_percentage: int,
    num_layers: int,
    unique_nodes_dict: dict = None,
) -> nx.DiGraph:
    """Filter the assembly digraph edges by removing a percentage of the edges of each layer.

    Args:
        assembly_digraph: A directed graph representing the assembly states.
        filter_percentage: The percentage of elements to pick.
        num_layers: Total number of layers in the assembly digraph.
        unique_nodes_dict: All protected nodes, the ones not to filter.

    Returns:
        The filtered directed graph.
    """
    assert 0 <= filter_percentage < 99, (
        "The filter percentage must be between 1 and 90. "
        f"Given value: {filter_percentage}"
    )

    curr_layer = num_layers - 2
    edge_list = []
    for node in assembly_digraph.nodes():
        # TODO: protect second layer
        if int(node.split("_")[0]) in [0, 1, num_layers - 1]:
            continue

        # Filter the nodes of the current layer
        # when the first node of the next layer is reached
        if int(node.split("_")[0]) != curr_layer:

            # In case of unique nodes filter the important nodes
            if unique_nodes_dict is not None:
                rand_edges_to_remove = pick_random_percentage(
                    edge_list, filter_percentage, unique_nodes_dict[curr_layer]
                )
            else:
                rand_edges_to_remove = pick_random_percentage(
                    edge_list, filter_percentage
                )
            # Remove the random edges
            for u, v in rand_edges_to_remove:
                assembly_digraph.remove_edge(u, v)

            # Empty the edge list for the next layer
            curr_layer -= 1
            edge_list = []

        edge_list.extend(list(assembly_digraph.out_edges(node)))

    # Remove all the successors
    nodes_to_remove = []
    all_removed = False
    while not all_removed:
        for node in assembly_digraph.nodes():
            if len(list(assembly_digraph.successors(node))) == 0:
                nodes_to_remove.append(node)

        # Remove the final node since it's the last successor
        nodes_to_remove.remove(f"{num_layers-1}_1")

        for node in nodes_to_remove:
            assembly_digraph.remove_node(node)

        if len(nodes_to_remove) == 0:
            all_removed = True
        nodes_to_remove = []
    return assembly_digraph


def filter_assembly_diagraph_nodes(
    assembly_digraph: nx.DiGraph, filter_percentage: int, num_layers: int
) -> nx.DiGraph:
    """Filter the assembly digraph nodes by removing a percentage of the nodes of each layer.

    Args:
        assembly_digraph: A directed graph representing the assembly states.
        filter_percentage: The percentage of elements to pick.
        num_layers: Total number of layers in the assembly digraph.

    Returns:
        The filtered directed graph.
    """
    assert 0 <= filter_percentage < 99, (
        "The filter percentage must be between 1 and 100. "
        f"Given value: {filter_percentage}"
    )

    # NOTE: Do not filter layer 1, and num_layers - 1, layers [0, num_layers]
    curr_layer = 1
    temp_node_list = []
    remove_node_list = []
    for node in assembly_digraph.nodes():
        if int(node.split("_")[0]) == 0:
            continue

        if int(node.split("_")[0]) == num_layers - 1:
            break

        if int(node.split("_")[0]) != curr_layer:
            # Filter the nodes of the current layer
            # when the first node of the next layer is reached
            num_nodes = len(temp_node_list)
            num_nodes_to_remove = int(num_nodes * (filter_percentage / 100))
            rand_nodes_indices = random.sample(range(num_nodes), num_nodes_to_remove)
            rand_nodes_indices.sort(reverse=True)

            # Empty the node list for the next layer
            curr_layer += 1
            for index in rand_nodes_indices:
                remove_node_list.append(temp_node_list[index])
            temp_node_list = []

        temp_node_list.append(node)

    for node in remove_node_list:
        assembly_digraph.remove_node(node)
    return assembly_digraph


def find_unique_nodes_from_df(filename: str, unique_nodes_dict: dict = None) -> dict:
    """Find the unique nodes of the assembly digraph.

    Args:
        filename: The filename of the shortest path weight results.
            Consists of short paths for various weights.
        unique_nodes_dict: All protected nodes, the ones not to filter.
            If exists, it will be updated.

    Returns:
        A dictionary containing the unique nodes of each layer.
    """
    # Read the results from the shortest path weight analysis
    df = pd.read_csv(filename, sep=",")

    shortest_paths = df["shortest_path"]

    # Declare a dictionary to store the unique nodes
    if not unique_nodes_dict:
        unique_nodes_dict = {}
        num_layers = int(re.findall(r"'(.*?)_", list(shortest_paths)[0])[0])
        for i in range(num_layers):
            unique_nodes_dict[i] = []

    for path in shortest_paths.items():
        for i, node_id_str in enumerate(
            path[0]
        ):  # ignore the first and last node path[1]
            temp_node = re.findall(r"'(.*?)'", node_id_str)[0]
            if temp_node not in unique_nodes_dict[i]:
                unique_nodes_dict[i].append(temp_node)
    return unique_nodes_dict


def find_unique_nodes_from_short_path(
    all_shortest_paths_gen: nx.all_shortest_paths,
) -> dict:
    """Use the generator from networkx.all_shortest_paths to find all the nodes
    included in all these paths.

    Args:
        all_shortest_paths_gen: A generator of all shortest paths

    Returs:
        A dictionary containing the unique nodes of each layer
    """
    unique_nodes_dict = {}
    for paths in all_shortest_paths_gen:
        for layer, node in enumerate(paths):
            if layer not in unique_nodes_dict:
                unique_nodes_dict[layer] = [node]
            else:
                if node not in unique_nodes_dict[layer]:
                    unique_nodes_dict[layer].append(node)
    return unique_nodes_dict


def keep_only_unique_nodes(
    assembly_digraph: nx.DiGraph,
    unique_nodes_dict: dict,
    last_layer_index: int,
) -> nx.DiGraph:
    """Keep only the unique nodes of each layer.

    Args:
        assembly_digraph: The assembly digraph.
        unique_nodes_dict: A dictionary containing the unique nodes of each layer.
        last_layer_index: Index of the last layer.

    Returns:
        The assembly digraph with only the unique nodes.
    """
    remove_nodes = []
    for node in assembly_digraph.nodes():
        if int(node.split("_")[0]) in [0, last_layer_index]:
            continue
        layer = int(node.split("_")[0])
        if node not in unique_nodes_dict[layer]:
            remove_nodes.append(node)

    for node in remove_nodes:
        assembly_digraph.remove_node(node)
    return assembly_digraph


def find_all_shortest_paths(
    assembly_digraph: nx.DiGraph, main_graph_num_edges: int, method: str = None
) -> dict:
    if not method:
        method = "dijkstra"
    all_shortest_paths = nx.all_shortest_paths(
        assembly_digraph,
        source="0_1",
        target=f"{main_graph_num_edges}_1",
        weight="edge_weight",
        method=method,
    )
    return find_unique_nodes_from_short_path(all_shortest_paths)


def minmax_norm(attr_vals: dict) -> list:
    """Calculate the minmax norm of a vector

    Args:
        vec_2_norm: Unnormalized values of a vector.

    Returns:
        Normalized vector.
    """
    if not attr_vals:
        raise ValueError("vec_2_norm list is empty")
    minx = min(attr_vals.values())
    maxx = max(attr_vals.values())
    for key, value in attr_vals.items():
        if abs(minx - maxx) < 1e-3:
            attr_vals[key] = 0
        else:
            attr_vals[key] = (value - minx) / (maxx - minx)
    return attr_vals


def scale_to(attr_vals: dict, div: str = "mean") -> list:
    """Scales the vector by dividing with its mean/median
    Args:
        vec_2_norm: Unnormalized values of a vector.
        div: Attribute to scale the vector, "mean" or "median".

    Returns:
        Scaled vector.
    """
    if not attr_vals:
        raise ValueError("vec_2_norm list is empty")
    np_vals = np.array(list(attr_vals.values()))
    div_val = 1
    if div == "mean":
        div_val = np.mean(np_vals)
    elif div == "median":
        div_val = np.median(np_vals)
    else:
        raise ValueError(f"div sould be either 'mean' or 'median', not {div} ")
    for key, value in attr_vals.items():
        attr_vals[key] = value / div_val
    return attr_vals


def normalize_attributes(graph: nx.Graph) -> nx.Graph:
    """Normalize the assembly attributes (edges, nodes).

    Attributes to normalize:
        Tolerance
        Sensibility
        Handling
        Time

    Args:
        graph: Constructed assembly graph.

    Returns:
        Graph with normalized attributes (weights, time)
    """
    tol = "tolerance"
    hand = "handling"
    ti = "time"

    tolerance = nx.get_edge_attributes(graph, tol)
    node_handling = nx.get_node_attributes(graph, hand)
    edge_handling = nx.get_edge_attributes(graph, hand)
    time = nx.get_edge_attributes(graph, ti)

    # First, make a copy of the absolute values
    # to be use later in the results
    nx.set_edge_attributes(graph, tolerance, name="absolute_tolerance")
    nx.set_node_attributes(graph, node_handling, name="absolute_handling")
    nx.set_edge_attributes(graph, edge_handling, name="absolute_handling")
    nx.set_edge_attributes(graph, time, name="absolute_time")

    nx.set_edge_attributes(graph, minmax_norm(tolerance), tol)
    nx.set_node_attributes(graph, minmax_norm(node_handling), hand)
    nx.set_edge_attributes(graph, minmax_norm(edge_handling), hand)
    nx.set_edge_attributes(graph, scale_to(time, div="median"), ti)
    return graph
