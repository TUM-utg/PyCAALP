"""Random cutset reduction of a assebmly digraph - assembly digraph.
This will be used to reduce the complexity of the MILP problem.
"""

import os
import pickle
from loguru import logger

from pycaalp.gapp.product_class import Product
from pycaalp.gapp.filtering import (
    filter_assembly_digraph_edges,
    filter_assembly_diagraph_nodes,
    find_unique_nodes_from_df,
)
from pycaalp.time_balancing.sp_var_weigh_constants_mix import create_unique_folder


def create_pkls_with_filtered_assembly_digraph(
    parts_filename: str,
    res_dir: str,
    shortest_paths_filename: str,
    unique_nodes_dict: dict = None,
    start_perc: int = 0,
    end_perc: int = 90,
    step: int = 2,
    save_pickle: bool = True,
    filter_method: str = "edges",
    dfm_pack: tuple = (None),
    **kwargs,
) -> str:
    """Filter the assembly digraph edges or nodes to reduce the complexity of the MILP problem.

    Args:
        parts_filename: Parts data file.
        res_dir: Results' directory.
        shortest_paths_filename: File with shortest path results.
        unique_nodes_dict: Unique nodes gathered from previous shortest path results.
        start_perc: Initial edge reduction percentage.
        end_perc: Final edge reduction percentage.
        step: Step for each reduction run.
        save_pickle: True if results of the reduced assembly digraph
            will be saved to a PKL file.
        filter_method: Filter method; "edge" or "node".
            Edge reduction is recommended.
        dfm_pack: DFM information, i.e., config.py instances (dfm, dfm_filename, coordysy_filename)

    Raises:
        ValueError: If weight constants are not provided.

    Returns:
        Full path of the new folder with PKL files of reduces digraphs.
    """

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Check if weight constants are provided
    w_tech, w_hand, w_tol = ""
    for key, value in kwargs.items():
        if key == "w_tech":
            w_tech = value
        elif key == "w_hand":
            w_hand = value
        elif key == "w_tol":
            w_tol = value
        else:
            raise ValueError("Invalid keyword argument")
    # Create a unique folder for the current run
    _run_dir = create_unique_folder(res_dir)
    # Create a product class assebmly digraph
    if parts_filename.endswith(".pkl"):
        with open(parts_filename, "rb") as f:
            prod_class = pickle.load(f)
    elif parts_filename.endswith(".csv"):
        if dfm_pack[0]:
            prod_class = Product(
                file_name=parts_filename,
                with_time=True,
                freedom_matrices=True,
                freedom_matrices_filename=dfm_pack[1],
                coordsys_filename=dfm_pack[2],
            )
        else:
            prod_class = Product(file_name=parts_filename, with_time=True)
        prod_class.w_tech = w_tech
        prod_class.w_hand = w_hand
        prod_class.w_tol = w_tol
        prod_class.compute_assembly_digraph_complete()
        # Save Product class as pickle
        if save_pickle:
            pickle_filename = parts_filename[:-4] + "_post_red_perc_0.pkl"
            prod_class.save_class_to_pickle(
                file_name=_run_dir + pickle_filename.split("/")[-1]
            )
    prod_class.product_format_to_pkl = "dict"  # Important to run for MILP

    if shortest_paths_filename is not None:
        unique_nodes_dict = find_unique_nodes_from_df(shortest_paths_filename)

    initial_assembly_digraph = prod_class.assembly_digraph.copy()
    initial_num_nodes = initial_assembly_digraph.number_of_nodes()
    initial_num_edges = initial_assembly_digraph.number_of_edges()
    for i in range(start_perc, end_perc, step):
        # ===================================================================== #
        #  APPLY POST PROCESSING METHODS TO REDUCE THE COMPLEXITY OF THE MODEL  #
        # ===================================================================== #
        # Filter the assembly digraph edges
        logger.info(
            f"Filtering {i} % of the assembly digaph with method: remove {filter_method}"
        )

        prod_class.filter_percentage = i
        if filter_method == "edges":
            prod_class.assembly_digraph = filter_assembly_digraph_edges(
                prod_class.assembly_digraph,
                prod_class.filter_percentage,
                prod_class.get_num_layers,
                unique_nodes_dict,
            )
        elif filter_method == "nodes":
            prod_class.assembly_digraph = filter_assembly_diagraph_nodes(
                prod_class.assembly_digraph,
                prod_class.filter_percentage,
                prod_class.get_num_layers,
            )
        curr_num_nodes = prod_class.assembly_digraph.number_of_nodes()
        node_reduction = 100 * (initial_num_nodes - curr_num_nodes) / initial_num_nodes
        logger.info(f"Num nodes: {curr_num_nodes}, {node_reduction:.2f}% decrease")
        curr_num_edges = prod_class.assembly_digraph.number_of_edges()
        edge_reduction = 100 * (initial_num_edges - curr_num_edges) / initial_num_edges
        logger.info(f"Num edges: {curr_num_edges}, {edge_reduction:.2f}% decrease")
        # Save product class as pickle
        if save_pickle:
            logger.info(f"Saving pickle for {i} %")
            pickle_filename = parts_filename[:-4] + f"_post_red_perc_{i}.pkl"
            prod_class.save_class_to_pickle(
                file_name=_run_dir + pickle_filename.split("/")[-1]
            )
        # Restore the assebmly digraph to its initial state
        prod_class.assembly_digraph = initial_assembly_digraph.copy()
    return _run_dir
