"""Operations to read and write part data from cvs or xlsx files"""

import networkx as nx
import numpy as np
import pandas as pd
import json


def read_graph_from_file_complete(
    file_name: str,
    with_time: bool,
    only_technology: bool,
    file_type: str = "xlsx",
    with_weights: bool = True,
) -> nx.Graph:
    """
    Reads a graph from an Excel file and returns a NetworkX graph object.
    NOTE: this function is hardcoded based on the data format of the CSV.

    Parameters:
        file_name: The path to the CSV or Excel file.
        with_time: True if the file contains time data.
        file_type: 'xlsx' or 'csv'.
        with_weights: True if the file contains weights and sensibilities.

    Returns:
        A NetworkX graph object.
    """
    # Main csv file format
    # 0: Part A
    # 1: weight A
    # 2: Handlingsens for Part A - used as handling for the edge
    # 3: Part B
    # 4: weight B
    # 5: Handlingsens for Part B - used as handling for the edge
    # 6: Desc.Part A
    # 7: Desc.Part B
    # 8: Technology
    # 9: Time
    # 10: Tolerance

    # Read the file
    data_frame = None
    if file_type == "xlsx":
        data_frame = pd.read_excel(file_name)
    elif file_type == "csv":
        data_frame = pd.read_csv(file_name)
    else:
        raise ValueError("Data file should be either in CSV or XLSX formats")

    graph = nx.Graph()
    all_weights = []
    all_sensb = []
    tolerance = []
    time = []

    if not with_weights:
        print("Reading file with no weights and sensibilities...")
        # Access from_id, to_id and technology columns
        # NOTE: for now there is only technology - retrieve manually
        technology = data_frame.iloc[:, 5].to_numpy()

        # Create the nodes indices based on the node names
        from_names = data_frame.iloc[:, 3].astype(str).to_numpy()
        to_names = data_frame.iloc[:, 4].astype(str).to_numpy()

        # Create the id indices based on the node ids
        from_ids = data_frame.iloc[:, 1].astype(str).to_numpy()
        to_ids = data_frame.iloc[:, 2].astype(str).to_numpy()

    else:
        # Access from_id, to_id and technology columns
        technology = data_frame.iloc[:, 9].to_numpy()
        if with_time:
            time = data_frame.iloc[:, 10].to_numpy()

        if not only_technology:
            tolerance = data_frame.iloc[:, 11].to_numpy()

        # Create the nodes indices based on the node names
        from_names = data_frame.iloc[:, 7].astype(str).to_numpy()
        to_names = data_frame.iloc[:, 8].astype(str).to_numpy()

        from_ids = data_frame.iloc[:, 1].astype(str).to_numpy()
        to_ids = data_frame.iloc[:, 4].astype(str).to_numpy()

        # Access weights and sensibilities
        from_weights = data_frame.iloc[:, 2].astype(str).to_numpy()
        to_weights = data_frame.iloc[:, 5].astype(str).to_numpy()
        from_sensb = data_frame.iloc[:, 3].astype(str).to_numpy()
        to_sensb = data_frame.iloc[:, 6].astype(str).to_numpy()

        # Concatenate weights and sensibilities
        all_weights = np.concatenate([from_weights, to_weights])
        all_sensb = np.concatenate([from_sensb, to_sensb])

    # Filter the ids  of from and to and remove '\xa0' from the end
    for i, _id in enumerate(from_ids):
        if _id.endswith("\xa0"):
            from_ids[i] = _id.replace("\xa0", "")
        if to_ids[i].endswith("\xa0"):
            to_ids[i] = to_ids[i].replace("\xa0", "")

    # concatenate the from-to names and ids
    all_names = np.concatenate([from_names, to_names])
    all_ids = np.concatenate([from_ids, to_ids])

    # Create a dictionary with the indices as keys and the names and ids as values
    # This will be saved in the main class - added 2 more dictionaries for the names and ids
    # NOTE: check if needed - performance reasons
    _id = 0
    full_parts_dict = {}
    part_name_to_index_dict = {}
    part_id_to_index_dict = {}
    weight_list = {}
    sensb_list = {}

    used_parts_list = []
    for i, part_id in enumerate(all_ids):
        if part_id not in used_parts_list and part_id not in ["nan", "Null"]:
            # Update the 3 dictionaries
            full_parts_dict[_id] = (all_names[i], all_ids[i])
            part_name_to_index_dict[all_names[i]] = _id
            part_id_to_index_dict[part_id] = _id
            used_parts_list.append(part_id)

            if with_weights:
                weight_list[_id] = float(all_weights[i])
                sensb_list[_id] = int(float(all_sensb[i]))

            _id += 1

    # Create the edges list to to be used for graph.add_edges_from
    edges_list = []
    technology_list = []
    time_list = []
    tolercance_list = []
    sensibility_list = []

    # Added indices for each method
    # NOTE: check how to implement for more methods
    technology_indices = {"MAG": 1}
    for i, from_id in enumerate(from_ids):
        if from_id not in ["nan", "Null"] and to_ids[i] not in ["nan", "Null"]:
            edges_list.append(
                (part_id_to_index_dict[from_id], part_id_to_index_dict[to_ids[i]])
            )
            # Update technology-tolerance-time lists
            technology_list.append(technology_indices[technology[i]])
            if not only_technology:
                tolercance_list.append(
                    int(tolerance[i])
                )  # NOTE: add max sensibilities???
                sensibility_list.append(
                    max(
                        sensb_list[part_id_to_index_dict[from_id]],
                        sensb_list[part_id_to_index_dict[to_ids[i]]],
                    )
                )
            if with_time:
                time_list.append(int(time[i]))

    # Add the edges to the graph and technology-tolerance-time as an attribute
    graph.add_edges_from(edges_list)
    nx.set_edge_attributes(graph, dict(zip(edges_list, technology_list)), "technology")
    if not only_technology:
        nx.set_edge_attributes(
            graph, dict(zip(edges_list, tolercance_list)), "tolerance"
        )
        nx.set_edge_attributes(
            graph, dict(zip(edges_list, sensibility_list)), "handling"
        )
    if with_time:
        nx.set_edge_attributes(graph, dict(zip(edges_list, time_list)), "time")

    # Add part_name and part_id as attributes to each node
    part_name_to_index_dict_labels = {
        i: part for i, part in enumerate(part_name_to_index_dict)
    }
    part_id_to_index_dict_labels = {
        i: part for i, part in enumerate(part_id_to_index_dict)
    }

    if not with_weights:
        nx.set_node_attributes(graph, part_name_to_index_dict_labels, "part_name")
        nx.set_node_attributes(graph, part_id_to_index_dict_labels, "part_id")

    else:
        nx.set_node_attributes(graph, part_id_to_index_dict_labels, "part_no")

        # Add weights and sensibility as attributes to each node
        nx.set_node_attributes(graph, weight_list, "weight")
        nx.set_node_attributes(graph, sensb_list, "sensibility")

    graph.parts_dict = (
        full_parts_dict  # Used in freedom matrices init and MIP output print
    )
    return graph


def read_graph_from_json(file_name: str) -> nx.Graph:
    """Read part data from json file and create graph.

    Args:
        file_name: JSON file with part data.

    Returns:
        Complete graph with all part joints and attributes.
    """
    with open(file_name, mode="r", encoding="utf-8") as f_data:
        part_data = json.load(f_data)

    parts = part_data["parts"]
    joints = part_data["joints"]
    graph = nx.Graph()

    # Add nodes
    for part in parts:
        curr_attrs = parts[part]
        graph.add_node(
            part, weight=curr_attrs["weight"], handling=curr_attrs["handling"]
        )
    # Add edges
    for joint in joints:
        curr_attrs = joints[joint]
        curr_parts = curr_attrs["parts"]
        max_handling = max(
            parts[curr_parts[0]]["handling"], parts[curr_parts[1]]["handling"]
        )
        graph.add_edge(
            curr_parts[0],
            curr_parts[1],
            name=joint,
            technology=curr_attrs["technology"],
            time=curr_attrs["time"],
            tolerance=curr_attrs["tolerance"],
            handling=max_handling,
        )

    assert len(list(parts)) == len(
        graph.nodes
    ), f"Number of nodes in graph is not equal to the number of parts in {file_name}"
    assert len(list(joints)) == len(
        graph.edges
    ), f"Number of edges in graph is not equal to the number of joints in {file_name}"
    return graph
