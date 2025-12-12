"""Create functions for the AssemblyDigraph class"""

from itertools import combinations
import networkx as nx

from pycaalp.gapp.checks import check_num_subgraphs_and_one_assembly_policy


def create_all_cutset_combinations_from_new_edges(
    init_graph: nx.Graph, edges_to_remove: list
) -> dict[int, list]:
    """
    Creates all possible combinations of edges to be removed from a graph.

    Args:
        init_graph: The initial graph.
        edges_to_remove: A list with the edges to be removed.

    Returns:
        A dictionary with graphs with all possible edges removed.
    """
    # Initialize the cutset dictionary with num keys and empty lists as values
    new_cutset_dict = {i: [] for i in range(0, len(edges_to_remove) - 1)}

    # Creates all combinations of edges to be removed
    # Skip last iteration since its the final graph where all edges are removed
    # and has been implemented outside the function
    for i in range(1, len(edges_to_remove)):
        for edges in list(combinations(edges_to_remove, i)):
            # Add the combinations to the dictionary
            temp_graph = init_graph.copy()
            temp_graph.remove_edges_from(edges)

            # No need to check for subgraphs since we checked initially for the
            # final graph
            if check_num_subgraphs_and_one_assembly_policy(
                None, temp_graph, check_subgraphs=False
            ):
                new_cutset_dict[i - 1].append(temp_graph)

    # Remove empty lists from dict
    new_cutset_dict = {k: v for k, v in new_cutset_dict.items() if v}
    return new_cutset_dict
