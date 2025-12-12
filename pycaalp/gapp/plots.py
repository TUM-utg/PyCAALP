"""Plot functions"""

import os
import matplotlib.pyplot as plt
import networkx as nx

TUM_BLUE = (0, 101 / 255, 189 / 255)
GRAY = (153 / 255, 153 / 255, 153 / 255)


def sort_name_ind(val: list[str]):
    sort_elem_list = [None] * len(val)
    for i in range(1, len(val) + 1):
        for j, v in enumerate(val):
            if int(v.replace("J", "")) == i:
                sort_elem_list[i - 1] = j
    return sort_elem_list


def sort_from_index_list(org_val: list[any], sort_elem_list: list[int]):
    sort_val = [None] * len(org_val)
    for i, ind in enumerate(sort_elem_list):
        sort_val[i] = org_val[ind]
    return sort_val


def plot_edge_attributes_w_table(
    graph: nx.Graph,
    pos: dict = None,
    plot_dir: str = None,
    part_name: str = None,
    save_format: str = "svg",
):
    """Plot edge attributes of an assembly in a table.

    Args:
        graph : graph to be plotted.
        pos : position of graph's nodes and edges.
        plot_dir : Plot file directory.
        part_name : Assembly part name.
        save_format : Save format ('png', 'svg'...).
    """
    if not pos:
        pos = nx.spring_layout(graph)

    plot_title = "joint attributes"
    if part_name:
        plot_title = f"{part_name} " + plot_title
    plt.title(plot_title, fontsize=16)
    # Basic plot
    nx.draw(
        graph,
        pos,
        node_color="black",
        edge_color=TUM_BLUE,
        font_size=7,
        width=1.5,
        node_size=50,
    )

    # Draw edge labels (weights)
    edge_short_names = {}
    edge_original_names = nx.get_edge_attributes(graph, "name")
    for joint, original_eges_names in edge_original_names.items():
        edge_short_names[joint] = f"J{original_eges_names.replace("joint","")}"
    # For table representation case
    sort_edge_names_indices = sort_name_ind(list(edge_short_names.values()))

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_short_names,
        font_size=8,
        font_color="black",
        font_weight="bold",
    )
    edge_short_names = sort_from_index_list(
        list(edge_short_names.values()), sort_edge_names_indices
    )
    time_labels = list(nx.get_edge_attributes(graph, "absolute_time").values())
    tol_labels = list(nx.get_edge_attributes(graph, "absolute_tolerance").values())
    tech_labels = list(nx.get_edge_attributes(graph, "technology").values())

    time_labels = sort_from_index_list(time_labels, sort_edge_names_indices)
    tol_labels = sort_from_index_list(tol_labels, sort_edge_names_indices)
    tech_labels = sort_from_index_list(tech_labels, sort_edge_names_indices)

    edge_table_data = [time_labels] + [tol_labels] + [tech_labels]
    row_labels = ["w_len", "tol", "tech"]
    table = plt.table(
        edge_table_data,
        colLabels=edge_short_names,
        colColours=[TUM_BLUE] * len(edge_short_names),
        rowLabels=row_labels,
        rowColours=[GRAY] * len(row_labels),
    )
    # https://matplotlib.org/stable/api/table_api.html#matplotlib.table.Cell.set_text_props
    for i in range(len(edge_short_names)):
        table[0, i].set_text_props(
            color="w", weight="bold", size="x-large", style="italic"
        )
    for i in range(len(row_labels)):
        table[i + 1, -1].set_text_props(
            color="w", weight="bold", size="x-large", style="italic"
        )

    plot_fname = f"{part_name.replace(" ", "")}_edge_attributes.{save_format}"
    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_fname = os.path.join(plot_dir, plot_fname)

    plt.savefig(plot_fname, format=save_format, bbox_inches="tight")
    plt.close()


def plot_node_attributes_w_table(
    graph: nx.Graph,
    pos: dict = None,
    plot_dir: str = None,
    part_name: str = None,
    save_format: str = "svg",
):
    """Plot node attributes of an assembly in a table.

    Args:
        graph : graph to be plotted.
        pos : position of graph's nodes and edges.
        plot_dir : Plot file directory.
        part_name : Assembly part name.
        save_format : Save format ('png', 'svg'...).
    """
    if not pos:
        pos = nx.spring_layout(graph)

    plot_title = "part attributes"
    if part_name:
        plot_title = f"{part_name} " + plot_title
    plt.title(plot_title, fontsize=16)
    # Basic plot
    nx.draw(
        graph,
        pos,
        node_color=TUM_BLUE,
        edge_color="black",
        font_size=4,
        node_size=250,
    )

    # Draw node labels (weights)
    node_short_names = {}
    for i, part in enumerate(list(graph.nodes())):
        node_short_names[part] = f"P{i+1}"
    nx.draw_networkx_labels(
        graph, pos, node_short_names, font_size=8, font_weight="bold", font_color="w"
    )
    mass_label = list(nx.get_node_attributes(graph, "weight").values())
    handling_label = list(nx.get_node_attributes(graph, "absolute_handling").values())

    # Parts-Nodes table
    part_row = list(node_short_names.values())
    node_table_data = [mass_label] + [handling_label]
    row_labels = ["mass", "hand"]
    table = plt.table(
        node_table_data,
        colLabels=part_row,
        colColours=[TUM_BLUE] * len(part_row),
        rowLabels=row_labels,
        rowColours=[GRAY] * len(row_labels),
    )

    for i in range(len(part_row)):
        table[0, i].set_text_props(
            color="w", weight="bold", size="x-large", style="italic"
        )
    for i in range(len(row_labels)):
        table[i + 1, -1].set_text_props(
            color="w", weight="bold", size="x-large", style="italic"
        )

    plot_fname = f"{part_name.replace(" ", "")}_node_attributes.{save_format}"
    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_fname = os.path.join(plot_dir, plot_fname)

    plt.savefig(plot_fname, format=save_format, bbox_inches="tight")
    plt.close()


def save_plot_assembly_graphs_w_table(graph: nx.Graph, plot_dir=None, part_name=None):
    """Create 2 plots with node and edge attributes for a given graph.

    Args:
        graph : The graph to be plotted.
        plot_dir : Plot file directory.
        part_name : Assembly part name.
    """

    # pos = nx.spring_layout(graph)  # Layout for positioning nodes
    pos = nx.kamada_kawai_layout(graph)  # Layout for positioning nodes
    plot_edge_attributes_w_table(graph, pos, plot_dir=plot_dir, part_name=part_name)
    plot_node_attributes_w_table(graph, pos, plot_dir=plot_dir, part_name=part_name)
