"""Add and remove operations for the AssemblyDigraph class.

Addition of node and edge attributes.
Removal of duplicate nodes.

"""

import networkx as nx
import numpy as np

from pycaalp.gapp.freedom_matrices import complete_collision_check


def extend_dict(cutset_dict: dict, new_values: dict) -> dict:
    """
    Extends the values of a dictionary with the values of another dictionary.

    Args:
        cutset_dict: The dictionary to be extended.
        new_values: The dictionary with the new values.

    Returns:
        The extended dictionary.
    """
    if isinstance(new_values, nx.Graph):
        cutset_dict[0].append(new_values)
    elif isinstance(new_values, dict):
        for key, value in new_values.items():
            if key not in cutset_dict:
                cutset_dict[key] = []
            cutset_dict[key].extend(value)
    return cutset_dict


def add_node_with_attributes(
    digraph: nx.DiGraph,
    name: str,
    graph: nx.Graph,
    attributes_dict: dict,
    *additional_attributes: str,
) -> None:
    """
    Adds a node to a digraph with the graph as an attribute and further attributes
    as attributes of the node.
    Possible additional attributes are: 'acc_weight', 'max_sensibility'

    Args:
        digraph: The digraph to be modified.
        name: The name of the node.
        graph: The graph to be added as an attribute.
        attributes_dict: A dictionary with the attributes to be added to the node with
            their coressponding values.
        *additional_attributes: The additional attributes to be added to the node
            taken from the graph.

    Returns:
        nx.Graph: The modified digraph.
    """
    # Add the attributes of the dict to the node
    for key, value in attributes_dict.items():
        digraph.nodes[name][key] = value

    # Add the additional attributes to the node
    conn_comp = np.array(list(nx.connected_components(graph)))
    conn_comp_mask = np.vectorize(__set_length_greater_than_one)(conn_comp)
    conn_comp = conn_comp[conn_comp_mask]
    # Add the connected components indices and ids to the digraph
    if conn_comp.size == 0:
        digraph.nodes[name]["conn_comp_index"] = []
        digraph.nodes[name]["conn_comp_id"] = []
    else:
        digraph.nodes[name]["conn_comp_index"] = list(conn_comp[0])
        part_no_ids = nx.get_node_attributes(graph, "part_no")
        digraph.nodes[name]["conn_comp_id"] = [part_no_ids[key] for key in conn_comp[0]]

    # Get weights, sensibilities
    weights = nx.get_node_attributes(graph, "weight")
    sensibilities = nx.get_node_attributes(graph, "sensibility")

    for attr in additional_attributes:
        if attr == "acc_weight":
            if conn_comp.size == 0:
                acc_weight = 0
            else:
                acc_weight = sum(
                    [weights[key] for key in conn_comp[0] if key in weights]
                )

            digraph.nodes[name][attr] = acc_weight
        elif attr == "max_sensibility":
            if conn_comp.size == 0:
                max_sensb = 0
            else:
                max_sensb = max(
                    [sensibilities[key] for key in conn_comp[0] if key in sensibilities]
                )
            digraph.nodes[name][attr] = max_sensb
        else:
            print(f"Attribute {attr} not supported")
            break


def add_edge_with_attributes(
    main_graph: nx.Graph,
    digraph: nx.DiGraph,
    from_graph: nx.Graph,
    to_graph: nx.Graph,
    from_name: str,
    to_name: str,
    attributes_dict: dict,
    *attributes: str,
) -> None:
    """
    Adds an edge to a digraph with the graph as an attribute and further attributes
    as attributes of the edge.
    Possible additional attributes are: 'technology', 'sensibility', 'part_a', 'part_b'

    Args:
        main_graph: The initial graph.
        digraph: The digraph to be modified.
        from_graph: The graph of the source node.
        to_graph: The graph of the target node.
        from_name: The name of the source node.
        to_name: The name of the target node.
        attributes_dict: A dictionary with the attributes to be added to the edge with
            their coressponding values.
        *attributes: The additional attributes to be added to the edge
            taken from the graph.
    """
    # Access part a and part b
    # Find the parts to be connected
    init_graph_data = main_graph.nodes.data()
    source_graph_edges = from_graph.edges()
    target_graph_edges = to_graph.edges()

    for edge in target_graph_edges:
        if edge not in source_graph_edges:
            new_edge = edge
            break

    # Check if there is a new node addition only when freedom matrices are used
    if attributes_dict["freedom_matrix"]:
        # Check node addition first
        if not (from_graph.edges(new_edge[0]) and from_graph.edges(new_edge[1])):
            new_part = new_edge[1] if from_graph.edges(new_edge[0]) else new_edge[0]
            if complete_collision_check(main_graph, from_graph, new_edge, new_part):
                return

    #  Add the edge to the digraph if no collision is detected
    digraph.add_edge(from_name, to_name)

    # Access the initial graph attributes
    tech_init_graph = (
        nx.get_edge_attributes(main_graph, "technology")
        if "technology" in attributes
        else None
    )
    sensb_init_graph = (
        nx.get_node_attributes(main_graph, "sensibility")
        if "sensibility" in attributes
        else None
    )

    # Add the dict attributes to the edge
    for key, value in attributes_dict.items():
        digraph.edges[from_name, to_name][key] = value

    # Add the additional attributes to the edge
    for attr in attributes:
        if attr == "technology":
            digraph.edges[from_name, to_name][attr] = tech_init_graph.get(new_edge)
        elif attr == "sensibility":
            digraph.edges[from_name, to_name]["max_" + attr] = max(
                sensb_init_graph.get(new_edge[0]), sensb_init_graph.get(new_edge[1])
            )
        elif attr == "part_a":
            digraph.edges[from_name, to_name][attr + "_index"] = new_edge[0]
            digraph.edges[from_name, to_name][attr + "_id"] = init_graph_data[
                new_edge[0]
            ]["part_no"]
        elif attr == "part_b":
            digraph.edges[from_name, to_name][attr + "_index"] = new_edge[1]
            digraph.edges[from_name, to_name][attr + "_id"] = init_graph_data[
                new_edge[1]
            ]["part_no"]
        else:
            print(f"Attribute {attr} not supported")
            break


def remove_duplicate_equivalent_graphs(temp_cutset: list) -> list:
    """
    Removes duplicate and equivalent graphs from a list of disassembly states.

    Parameters:
        disassembly_states: A list of NetworkX Graph objects.

    Returns:
        A list of unique graphs after removing duplicate and equivalent duplicates.

    """
    if len(temp_cutset) > 0:
        temp_cutset = __remove_duplicate_graphs(temp_cutset)
        temp_cutset = __remove_equivalent_graphs(temp_cutset)
    return temp_cutset


def __remove_duplicate_graphs(disassembly_states: list) -> list:
    """
    Removes equal graphs from a list of disassembly states.

    Parameters:
        disassembly_states: A list of NetworkX Graph objects.

    Returns:
        A list of unique graphs after removing equal duplicates.
    """
    unique_graphs = []
    unique_graphs.append(disassembly_states[0])
    # this can be maybe parallized
    for graph in disassembly_states:
        is_duplicate = False
        for unique_graph in unique_graphs:
            if graph == unique_graph:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_graphs.append(graph)
    return unique_graphs


def __remove_equivalent_graphs(disassembly_states: list) -> list:
    """
    Removes equivalent graphs from a list of disassembly states.

    Parameters:
        disassembly_states: A list of NetworkX Graph objects.

    Returns:
        A list of unique graphs after removing equivalent duplicates.
    """
    # First reorder the states in descending order
    graphs_sorted = sorted(
        disassembly_states,
        key=lambda g: nx.number_of_edges(g),
        reverse=True,
    )

    unique_graphs = []
    unique_graphs.append(graphs_sorted[0])
    for graph in graphs_sorted:
        is_duplicate = False
        for unique_graph in unique_graphs:
            # Check if all the edges of graph are included in each unique_graph
            if set(graph.edges()).issubset(set(unique_graph.edges())):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_graphs.append(graph)
    return unique_graphs


def __set_length_greater_than_one(s: set) -> bool:
    """
    Checks if a set has more than one elements.

    Args:
        s: The set to be checked.

    Returns:
        True if the set has more than one elements, False otherwise.
    """
    return len(s) > 1
