"""Filtering functions for the time balancing algorithm."""

import math
import random
import re
import pandas as pd
import numpy as np
import networkx as nx
from loguru import logger

from pycaalp.gapp.paths import k_shortest_paths


def pick_random_percentage(
    input_list: list, percentage: int, protected_elements: set | None = None
) -> list:
    """Pick a percentage of elements randomly from a list.

    Args:
        input_list: The list to pick elements from.
        percentage: The percentage of elements to pick.
        protected_elements: Elements to protect, i.e., can't be picked.
            Checked by membership of the element itself (for edge lists,
            the protected elements are (u, v) edge tuples).

    Returns:
        A list containing the randomly picked elements.
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Drop the protected elements so they can never be picked
    if protected_elements is not None:
        for i in range(len(input_list) - 1, -1, -1):
            if input_list[i] in protected_elements:
                del input_list[i]

    num_elements_to_pick = int(len(input_list) * (percentage / 100))
    return random.sample(input_list, num_elements_to_pick)


def filter_assembly_digraph_edges(
    assembly_digraph: nx.DiGraph,
    filter_percentage: int,
    num_layers: int,
    protected_edges: set | None = None,
) -> nx.DiGraph:
    """Filter the assembly digraph edges by removing a percentage of the edges of each layer.

    Args:
        assembly_digraph: A directed graph representing the assembly states.
        filter_percentage: The percentage of elements to pick.
        num_layers: Total number of layers in the assembly digraph.
        protected_edges: (u, v) edge tuples that must never be removed, e.g.
            from find_adaptive_protected_edges (or edges_from_protected_nodes
            for the legacy node-based protection).

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
        if int(node.split("_")[0]) in [0, num_layers - 1]:
            continue

        # Filter the nodes of the current layer
        # when the first node of the next layer is reached
        if int(node.split("_")[0]) != curr_layer:

            rand_edges_to_remove = pick_random_percentage(
                edge_list, filter_percentage, protected_edges
            )
            # Remove the random edges
            for u, v in rand_edges_to_remove:
                assembly_digraph.remove_edge(u, v)

            # Empty the edge list for the next layer
            curr_layer -= 1
            edge_list = []

        edge_list.extend(list(assembly_digraph.out_edges(node)))

    # The loop ends without triggering the filter for the last accumulated layer
    # (layer 1), because layer-0 nodes are skipped via `continue`. Process it now.
    if edge_list and curr_layer not in [0, num_layers - 1]:
        rand_edges_to_remove = pick_random_percentage(
            edge_list, filter_percentage, protected_edges
        )
        for u, v in rand_edges_to_remove:
            assembly_digraph.remove_edge(u, v)

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
    assembly_digraph: nx.DiGraph,
    main_graph_num_edges: int,
    weight_attr: str = "edge_weight",
) -> dict:
    logger.debug("Runnning all shortest paths")
    all_shortest_paths = k_shortest_paths(
        assembly_digraph,
        source="0_1",
        target=f"{main_graph_num_edges}_1",
        k=2000,
        weight=weight_attr,
    )
    return find_unique_nodes_from_short_path(all_shortest_paths)


def edges_from_protected_nodes(
    assembly_digraph: nx.DiGraph, unique_nodes_dict: dict
) -> set:
    """Express the legacy node protection as an edge set.

    The old scheme protected every out-edge of a protected node (see
    pick_random_percentage before the edge-protection change). This helper
    reproduces exactly that semantics for the edge-based
    filter_assembly_digraph_edges, so the node scheme stays available as a
    baseline.

    Args:
        assembly_digraph: A directed graph representing the assembly states.
        unique_nodes_dict: Per-layer protected node names, e.g. from
            find_all_shortest_paths or find_unique_nodes_from_df.

    Returns:
        The set of (u, v) edges whose tail node is protected.
    """
    protected_nodes = set()
    for nodes in unique_nodes_dict.values():
        protected_nodes.update(nodes)
    return {
        edge
        for node in protected_nodes
        if node in assembly_digraph
        for edge in assembly_digraph.out_edges(node)
    }


def find_adaptive_protected_edges(
    assembly_digraph_obj,
    stop_perc_graph: float = 10.0,
    blends: tuple = (0.0, 0.5, 1.0),
    lam: float = 0.5,
    penalty: float = 0.5,
    stall_rounds: int = 2,
    diverse_stall: int = 25,
    base_attr: str = "_protect",
) -> set:
    """Collect the edges to protect from reduction by adaptive path enumeration.

    Alternative to find_all_shortest_paths (Yen with a magic k=2000 on the
    single edge_weight ranking): mirrors build_adaptive_subgraph — bl-union
    over the blend grid until its edge growth stalls, then diverse penalised
    re-routing at λ — but the budget is a **subgraph percentage** instead of a
    path count k, so the protected-set size transfers across assemblies (the
    k→edges map does not; see experiments/bl_union_convergence).

    Every returned edge lies on at least one enumerated source→sink path, so a
    reduction that keeps these edges keeps the digraph source→sink connected
    and contains the adaptive subgraph: the MIP objective on the reduced
    digraph is bounded by the adaptive subgraph's objective.

    Args:
        assembly_digraph_obj: AssemblyDigraph instance (needs
            set_blended_weights, so `time` attributes and num_phases).
        stop_perc_graph: Stop once the edge set reaches this percentage of the
            assembly digraph's edges (enumeration may stall earlier).
        blends: Blend grid for the bl-union phase (0 = pure edge_weight
            ranking, 1 = pure continuous balance ranking).
        lam: λ for the diverse re-routing phase (typically the true w_balanced).
        penalty: Additive re-routing penalty (see diverse_shortest_paths).
        stall_rounds: Consecutive duplicate bl-union rounds before switching
            to diverse re-routing.
        diverse_stall: Consecutive duplicate diverse paths before giving up
            (the re-routing needs a few rounds of penalties to escape a rut,
            so this is larger than stall_rounds).
        base_attr: Prefix for the private edge attributes written on the digraph.

    Returns:
        The set of (u, v) edges to protect.
    """
    digraph = assembly_digraph_obj.assembly_digraph
    src, tgt = "0_1", f"{assembly_digraph_obj.graph.number_of_edges()}_1"
    target_edges = math.ceil(stop_perc_graph / 100 * digraph.number_of_edges())

    # Phase 1 (bl-union): one Yen generator per blend, advanced round-robin.
    gens = []
    for i, blend in enumerate(blends):
        attr = f"{base_attr}_blend_{i}"
        assembly_digraph_obj.set_blended_weights(blend, out_attr=attr)
        gens.append(nx.shortest_simple_paths(digraph, src, tgt, weight=attr))

    edge_set = set()
    zero_streak = 0  # consecutive rounds that added no new edge
    active = list(range(len(gens)))
    while len(edge_set) < target_edges and active:
        before = len(edge_set)
        for gi in list(active):
            try:
                path = next(gens[gi])
            except StopIteration:
                active.remove(gi)
                continue
            edge_set.update(zip(path, path[1:]))
        if len(edge_set) == before:
            zero_streak += 1
            if zero_streak >= stall_rounds:
                break  # bl-union stalled — only duplicate paths left
        else:
            zero_streak = 0

    # Phase 2 (diverse): penalised re-routing at λ until the target is reached.
    # Inlined rather than calling diverse_shortest_paths so the loop can stop
    # exactly at target_edges (the helper resets its penalties on every call,
    # so chunked calls would regenerate the same paths forever).
    if len(edge_set) < target_edges:
        wattr = f"{base_attr}_lam"
        pen_attr = f"{base_attr}_pen"
        assembly_digraph_obj.set_blended_weights(lam, out_attr=wattr)
        for _, _, data in digraph.edges(data=True):
            data[pen_attr] = data[wattr]
        zero_streak = 0
        while len(edge_set) < target_edges:
            try:
                path = nx.shortest_path(digraph, src, tgt, weight=pen_attr)
            except nx.NetworkXNoPath:
                break
            before = len(edge_set)
            edge_set.update(zip(path, path[1:]))
            for u, v in zip(path, path[1:]):
                digraph[u][v][pen_attr] += penalty
            if len(edge_set) == before:
                zero_streak += 1
                if zero_streak >= diverse_stall:
                    break
            else:
                zero_streak = 0

    logger.debug(
        f"Protected {len(edge_set)} edges "
        f"({100 * len(edge_set) / digraph.number_of_edges():.1f}% of the digraph, "
        f"target {stop_perc_graph}%)"
    )
    return edge_set


def minmax_norm(attr_vals: dict) -> dict:
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


def scale_to(attr_vals: dict, div: str = "mean") -> dict:
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
    tol_s = "tolerance"
    hand_s = "handling"
    ti_s = "time"
    mass_s = "mass"  # avoid confusion with attribute/graph weights

    tolerance = nx.get_edge_attributes(graph, tol_s)
    node_handling = nx.get_node_attributes(graph, hand_s)
    edge_handling = nx.get_edge_attributes(graph, hand_s)
    time = nx.get_edge_attributes(graph, ti_s)
    mass = nx.get_edge_attributes(graph, mass_s)

    # First, make a copy of the absolute values
    # to be use later in the results
    nx.set_edge_attributes(graph, tolerance, name="absolute_tolerance")
    nx.set_node_attributes(graph, node_handling, name="absolute_handling")
    nx.set_edge_attributes(graph, edge_handling, name="absolute_handling")
    nx.set_edge_attributes(graph, time, name="absolute_time")
    nx.set_edge_attributes(graph, mass, name="absolute_mass")

    nx.set_edge_attributes(graph, minmax_norm(tolerance), tol_s)
    nx.set_node_attributes(graph, minmax_norm(node_handling), hand_s)
    nx.set_edge_attributes(graph, minmax_norm(edge_handling), hand_s)
    nx.set_edge_attributes(graph, minmax_norm(mass), mass_s)
    # scale_to results in better time balancing compared to minmax_norm
    # For assembly 1: max phase time for num phases 1 to 5.
    # minmax_norm: [2156.85, 1222.81, 878.68, 807.90, 462.92]
    # scale_to(median): [2156.85, 1100.04, 753.4, 567.5, 449.12]
    nx.set_edge_attributes(graph, scale_to(time, div="median"), ti_s)
    return graph
