"""The main class for the Assemlby Digraph API.

All the functions used for the computation of the assembly digraph.
"""

import sys
import time
import json
import math
import itertools
import networkx as nx
from loguru import logger


from pycaalp.gapp.checks import check_one_assembly_policy, check_technology_changes
from pycaalp.gapp.read_write import read_graph_from_json
from pycaalp.gapp.file_formats import assembly_digraph_to_dict, save_to_pkl

from pycaalp.gapp.freedom_matrices import (
    assign_fm_cs_to_main_graph,
    init_trafo_mats,
    complete_collision_check,
)

from pycaalp.gapp.paths import calculate_num_simple_paths

from pycaalp.gapp.filtering import (
    normalize_attributes,
    find_all_shortest_paths,
    filter_assembly_digraph_edges,
)

from pycaalp.gapp.paths import calculate_sum_of_sh_path_weights


class AssemblyDigraph:
    """Main class for the Assembly Digraph API.
    This class contains all the functions needed to compute the assembly digraph with the
    option of using deegree of freedom matrices.
    """

    def __init__(
        self,
        file_name: str = None,
        graph: nx.Graph = None,
        dfm_file=None,
        w_tech=0.3333,
        w_hand=0.3333,
        w_tol=0.3333,
        reduction_percentage=0,
        pkl_save_format: str = "dict",
        log_format: str = None,
    ):
        """Create an nx graph from a file or from a given graph, otherwise graph is None.

        Args:
            file_name: JSON file name with data of assembly parts and possibly
                some attributes, e.g., technology, sensibility.
            graph: NetworkX graph containing part to be assembled.
            dfm_file: JSON file name with DFM and coordinate system per joint.
            w_tech: Technology weight.
            w_hand: Handling weight.
            w_tol: Tolerance weight.
            reduction_percentage: Edge reduction (%) for the assembly directed graph reduction.
            pkl_save_format: Format of the pkl file: "dict" or "class".
            log_format: Loger format: "SET_OUT"(already set before the class creation),
                or loguru "INFO", "DEBUG".
        """
        if file_name.endswith(".json") or not isinstance(file_name, str):
            graph = read_graph_from_json(file_name)
        else:
            raise ValueError("File should be a string in JSON format")

        # Check if the graph is empty, fully connected and obeys the one assembly policy
        if graph is None:
            raise ValueError("No graph found")
        if not nx.is_connected(graph):
            raise ValueError("The graph is not connected.")
        if not check_one_assembly_policy(graph):
            raise ValueError("The graph consists of multiple subassemblies.")

        # Normalize graph attributes
        graph = normalize_attributes(graph)
        # Class attributes
        self.graph = graph
        self.graph_file_name = file_name
        self.assembly_digraph = None
        self.w_tech = w_tech
        self.w_hand = w_hand
        self.w_tol = w_tol
        self.reduction_percentage = reduction_percentage
        self.freedom_matrices = False
        self.pkl_save_format = pkl_save_format
        self.sum_of_sh_path_weights = None

        assert all(
            w >= 0.0 for w in [self.w_tech, self.w_hand, self.w_tol]
        ), f"All attribute coefficients should be positive\n. w_tech:{self.w_tech}, w_hand:{self.w_hand}, w_tol:{self.w_tol}"
        assert math.isclose(
            w_tech + w_hand + w_tol, 1.0, abs_tol=1e-3
        ), f"The attribute coefficients should sum to 1\n. w_tech:{self.w_tech}, w_hand:{self.w_hand}, w_tol:{self.w_tol}"

        # Main graph attributes
        self.node_handling = nx.get_node_attributes(graph, "handling")
        self.edge_tolerance = nx.get_edge_attributes(graph, "tolerance")
        self.edge_technology = nx.get_edge_attributes(graph, "technology")

        if dfm_file:
            if not isinstance(dfm_file, str):
                raise ValueError(f"DFM file should be a str. {dfm_file} was given.")
            if not dfm_file.endswith(".json"):
                raise ValueError(f"DFM file {dfm_file} should be in JSON format")

            self.freedom_matrices = True
            self.init_dfm_trafo_from_json(dfm_file)

        # Set logging mode
        if log_format is None:  # Defaults to INFO
            logger.remove()
            logger.add(sys.stdout, level="INFO")
        elif log_format == "SET_OUT":  # When is already set
            logger.info("Logger was set outside of the Assembly Digraph scope")
        else:
            logger.remove()
            logger.add(sys.stdout, level=log_format)

    def init_dfm_trafo_from_json(self, dfm_filename):
        """Initialize the freedom matrices and the coordinate systems for the given graph."""
        # Read file
        with open(dfm_filename, mode="r", encoding="utf-8") as jf:
            dfm_data_dict = json.load(jf)

        # Assign freedom matrices and coordinate system.
        assign_fm_cs_to_main_graph(main_graph=self.graph, dfm_dict=dfm_data_dict)
        # Add the transformation matrices.
        init_trafo_mats(main_graph=self.graph)

    def calculate_di_edge_weight(
        self,
        new_edge: tuple[int, int],
        layer: int,
        to_node_out_edges_operations: list,
    ) -> float:
        """Calculate the edge weight of the directed edge in the assembly digraph.

        Args:
            new_edge: edge to be added.
            prev_edges: previous edges in the assembly.
            layer: current layer of the assembly digraph.
            temp_graph_edge_tech: cuttent graphs edge weights.

        Returns:
            Calculated edge weight of the assembly digraph.
        """
        edge_weight = 0.0
        # Handling, Tolerance
        max_hand = self.graph[new_edge[0]][new_edge[1]]["handling"]
        # ΝΟΤΕ: This works since each layer adds only one edge
        edge_weight += (
            self.edge_tolerance.get(new_edge) * self.w_tol + max_hand * self.w_hand
        ) / layer

        # Technology
        if len(to_node_out_edges_operations) > 0:
            new_technology = self.edge_technology[new_edge]
            tech_change_count = 0
            for _, _, operation in to_node_out_edges_operations:
                if self.edge_technology[operation] != new_technology:
                    tech_change_count += 1
            tech_change_fraction = tech_change_count / len(to_node_out_edges_operations)
            edge_weight += self.w_tech * tech_change_fraction / layer

        return edge_weight

    def create_assembly_digraph(self) -> nx.DiGraph:
        """
        Computes the assembly digraph of the given graph.
        Fully fused approach: The assembly digraph is computed in place while
        the dissasembly states are computed.

        Returns:
            nx.DiGraph: A directed graph representing the assembly states.
        """
        disassembly_states = {0: [list(self.graph.edges())]}
        edges = list(self.graph.edges)
        total_num_layers = self.graph.number_of_edges()
        digraph = nx.DiGraph()
        # Add node layer if needed (it is already included on the node name)
        # digraph.add_node(f"{total_num_layers}_1", layer=total_num_layers)

        temp_graph = self.graph.copy()
        # Iterate over the number of layers and remove (num layers + 1) edges from
        # the fully connected graph
        for layer in range(1, total_num_layers + 1):
            disassembly_states[layer] = []
            curr_edge_index = 0

            # Create all combinations of edges to remove
            for comb in itertools.combinations(range(total_num_layers), layer):
                edges_to_remove = [edges[k] for k in comb]
                temp_graph.remove_edges_from(edges_to_remove)

                # Keep only the combinations that satisfy the one assembly policy
                if check_one_assembly_policy(temp_graph):
                    disassembly_states[layer].append(list(temp_graph.edges()))
                    curr_edge_index += 1

                    # Add connections to the previous layer of the assembly digraph
                    # NOTE: from -> temp_graph, to -> prev_graph(i.e., prev_edges)
                    for prev_edg_index, prev_edges in enumerate(
                        disassembly_states[layer - 1]
                    ):
                        # Skip empty lists, filtered in the previous loop
                        if not prev_edges:
                            continue

                        if set(temp_graph.edges()).issubset(set(prev_edges)):

                            new_edge = (prev_edges - temp_graph.edges()).pop()

                            # DFM check
                            if self.freedom_matrices:
                                # NOTE: Check dfm only in new node additions
                                # node addition for a new edge: not edges(partA) OR not edges(partB)
                                # equivalent to not (edges(partA) and edges(partB)) - DM Law
                                if not (
                                    temp_graph.edges(new_edge[0])
                                    and temp_graph.edges(new_edge[1])
                                ):
                                    new_part = (
                                        new_edge[1]
                                        if temp_graph.edges(new_edge[0])
                                        else new_edge[0]
                                    )
                                    if complete_collision_check(
                                        self.graph, temp_graph, new_edge, new_part
                                    ):
                                        continue

                            from_name = (
                                str(total_num_layers - layer)
                                + "_"
                                + str(curr_edge_index)
                            )

                            to_name = (
                                str(total_num_layers - layer + 1)
                                + "_"
                                + str(prev_edg_index + 1)
                            )

                            edge_weight = self.calculate_di_edge_weight(
                                new_edge,
                                total_num_layers
                                - layer
                                + 1,  # NOTE: use layer of digraph, not assembly state
                                list(digraph.out_edges(to_name, "operation")),
                            )

                            digraph.add_edge(
                                from_name,
                                to_name,
                                operation=new_edge,
                                edge_weight=edge_weight,
                            )

                temp_graph.add_edges_from(edges_to_remove)

            logger.debug(
                f"Finished layer {layer} out of {self.graph.number_of_edges()} "
            )
            logger.debug(f"Number of initial states: {len(disassembly_states[layer])}")

            # Try to remove all the nodes without succesor in the current layer of the digraph
            # to avoid checking for them in the next layer
            # Usefull only when dmf
            if self.freedom_matrices:
                num_deleted_nodes = 0
                for i in range(len(disassembly_states[layer])):
                    node_name_to_di_name = (
                        str(total_num_layers - layer) + "_" + str(i + 1)
                    )
                    if node_name_to_di_name not in digraph.nodes:
                        # print(f"Excluding digraph node {node_name_to_di_name}")
                        # Use element since name of digraph nodes starts from 1
                        disassembly_states[layer][i] = None
                        num_deleted_nodes += 1

                logger.debug(
                    f"Deleted nodes: {num_deleted_nodes} out of {len(disassembly_states[layer])}"
                )

        self.assembly_digraph = digraph
        return digraph

    def compute_assembly_digraph_complete(self) -> nx.DiGraph:
        """Computes the assebmly digraph of the given graph.

        Returns:
            nx.DiGraph: A directed graph representing the assebmly digraph.
        """
        # Check if there is no graph
        if self.graph is None:
            raise ValueError("No graph provided")
        start = time.time()
        logger.info("Computing assembly digraph...")
        # Run the complete assembly digraph code.
        self.create_assembly_digraph()
        # Reduce the graph
        if self.reduction_percentage:
            unique_nodes_dict = find_all_shortest_paths(
                self.assembly_digraph, self.graph.number_of_edges()
            )
            self.assembly_digraph = filter_assembly_digraph_edges(
                self.assembly_digraph,
                self.reduction_percentage,
                self.get_num_layers,
                unique_nodes_dict,
            )

        # Independent calculation, i.e., run shortest path again
        self.sum_of_sh_path_weights = calculate_sum_of_sh_path_weights(self)

        stop_comp_cuts_ev = time.time() - start
        # Log the total time
        logger.debug(f"Time to compute assembly digraph: {stop_comp_cuts_ev} sec")
        return self.assembly_digraph

    def save_class_to_pickle(self, file_name: str = "assembly_digraph.pkl") -> None:
        """Saves the AssemblyDigraph class to a pickle file by converting it to a dictionary.

        Args:
            file_name: File name of the PKL output to be generated.
        """
        if self.pkl_save_format == "dict":
            graph_to_dict = assembly_digraph_to_dict(self)
            save_to_pkl(graph_to_dict, file_name=file_name)
            # elif self.pkl_save_format == "class":
            save_to_pkl(self, file_name=file_name)
        else:
            raise ValueError("Unknown pickle save format")
        logger.info(f"Saved assembly digraph class to pickle file {file_name}")

    def generate_assembly_digraph_file_complete(
        self, file_name="assemlby_digraph.pkl"
    ) -> None:
        """Generates an assembly digraph file by computing the assebmly digraph
        and saving the assembly digraph class to a pickle file.

        Args:
            file_name: File name of the PKL output to be generated.
        """
        if self.freedom_matrices:
            print(
                f"Generating assembly digraph file from {self.graph_file_name}"
                "using freedom matrices"
            )
        else:
            print(
                f"Generating assembly digraph file from {self.graph_file_name}"
                " (no freedom matrices)"
            )
        self.compute_assembly_digraph_complete()
        self.save_class_to_pickle(file_name=file_name)

    def get_init_graph(self) -> nx.Graph:
        """Get the main assembly graph"""
        return self.graph

    def get_num_simple_paths(self) -> int:
        """Get the number of the simple paths of an assembly digraph"""
        if not self.assembly_digraph:
            raise ValueError("No assembly digraph exist.")
        return calculate_num_simple_paths(self)

    @property
    def get_num_layers(self) -> int:
        """Get the number of layers of a assebmly digraph"""
        # NOTE: compute indirectly the number of layers but efficient
        if not self.assembly_digraph:
            raise ValueError("No assembly digraph exists")
        return self.graph.number_of_edges() + 1
