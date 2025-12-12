"""Adjust the assembly diagraph weights and calculate the shortest paths.
Weights are wtech, wtol and whand.
"""

# pylint: disable=C0103, C0116
import os
import time
import random
from itertools import permutations

import networkx as nx
from loguru import logger
from pycaalp.gapp.product_class import Product


def create_unique_folder(results_dir: str) -> str:
    """Create a unque folder to store the simulation results.

    Args:
        results_dir (str): Directory to create the folder.

    Returns:
        str: Full path the new folder.
    """
    # Get the current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Create a unique folder name using the timestamp
    folder_name = results_dir + f"run_{timestamp}/"
    # Create the folder
    os.makedirs(folder_name)
    return folder_name


def random_weight_generator() -> list[int]:
    """Generate three random weight numbers used for the multiple shortest
    path method.

    The numbers are between 0 and 1.

    Returns:
        The technology, handling and tolerance weights.
    """
    # Generate random numbers between 0 and 1
    num1 = random.uniform(0, 1)
    num2 = random.uniform(0, 1 - num1)
    num3 = 1 - num1 - num2
    # Ensure all numbers are between 0 and 1
    num1 = max(0, min(1, num1))
    num2 = max(0, min(1, num2))
    num3 = max(0, min(1, num3))
    # Print the result
    return [num1, num2, num3]


def calc_sp_random_weights(
    prod_class: "Product", results_filename: str, num_rand_runs: int
) -> None:
    """Run shortest path algorithm for an assembly digraph with multiple
    random weights.

    Args:
        prod_class: The Product class with assembly digraph.
        results_filename: The filename to print the results to.
        num_rand_runs: Number of shortest path calculations.
    """
    for _ in range(num_rand_runs):
        rand_weights = random_weight_generator()
        for w_tech, w_hand, w_tol in permutations(rand_weights):
            logger.debug(f"w_tech: {w_tech}, w_hand: {w_hand}, w_tol: {w_tol}")
            assert (
                w_tech + w_hand + w_tol - 1 < 1e-5
            ), "The attribute coefficient should sum to 1"
            # Create assebmly digraph
            prod_class.w_tech = w_tech
            prod_class.w_hand = w_hand
            prod_class.w_tol = w_tol
            prod_class.create_assembly_digraph()
            # Calculate the shortest path
            shortest_path = nx.shortest_path(
                prod_class.assembly_digraph,
                source="0_1",
                target=f"{prod_class.graph.number_of_edges()}_1",
                weight="edge_weight",
                method="dijkstra",
            )
            # Write the results to the csv
            with open(results_filename, "a", encoding="utf8") as f:
                f.write(f"{shortest_path}, {w_tech:.4f},{w_hand:.4f},{w_tol:.4f}\n")


def calc_sp_manual_weights(prod_class: "Product", results_filename: str) -> None:
    """Run shortest path algorithm for an assembly digraph with multiple
    constant weights.

    The constant weights are specified in the function.

    Args:
        prod_class: The Product class with assembly digraph.
        results_filename: The filename to print the results to.
    """
    # Constant weights
    weights_dict = {
        1: [1, 0, 0],
        2: [0.5, 0.5, 0],
        3: [0.5, 0.25, 0.25],
        4: [0.5, 0.3, 0.2],
    }
    # Run for all unique permutations of each weights_dict combination.
    for i in range(0, len(weights_dict)):
        for w_tech, w_hand, w_tol in set(permutations(weights_dict[i + 1])):
            assert (
                w_tech + w_hand + w_tol - 1 < 1e-5
            ), "The attribute coefficient should sum to 1"
            # Create assebmly digraph
            prod_class.w_tech = w_tech
            prod_class.w_hand = w_hand
            prod_class.w_tol = w_tol
            prod_class.create_assembly_digraph()
            # Calculate the shortest path
            shortest_path = nx.shortest_path(
                prod_class.assembly_digraph,
                source="0_1",
                target=f"{prod_class.graph.number_of_edges()}_1",
                weight="edge_weight",
                method="dijkstra",
            )
            # Write the results to the csv
            with open(results_filename, "a", encoding="utf8") as f:
                f.write(f"{shortest_path}, {w_tech:.4f},{w_hand:.4f},{w_tol:.4f}\n")


def calculate_sp_many_weight_constants(
    file_name: str,
    res_dir: str,
    num_rand_runs: int = 5,
    log_format: str = None,
    dfm_pack: tuple = (None),
) -> str:
    """Run shortest path multiple times with random or manual edge weight constants.

    Args:
        file_name: Assembly part file name.
        res_dir: The directory to store the results.
        num_rand_runs: Number of runs for random weight shortest path calculation.
        log_format: Loger format: "SET_OUT"(already set before the class creation),
                or loguru "INFO", "DEBUG".
        dfm_pack: DFM information, i.e., config.py instances (dfm, dfm_filename, coordysy_filename)

    Returns:
        str: Full path of the new filename with the shortest path results per weight constant set.
    """
    # Create a unique folder for the current run
    _run_dir = create_unique_folder(res_dir)

    # Create a csv to add the shortest path for each weight set
    results_filename = _run_dir + "shortest_path_multiple_weights.csv"
    with open(results_filename, "w", encoding="utf8") as f:
        f.write("shortest_path,w_tech,w_hand,w_tol\n")

    if dfm_pack[0]:
        prod_class = Product(
            file_name=file_name,
            log_format=log_format,
            with_time=True,
            freedom_matrices=True,
            freedom_matrices_filename=dfm_pack[1],
            coordsys_filename=dfm_pack[2],
        )
    else:
        prod_class = Product(file_name=file_name, log_format=log_format, with_time=True)

    calc_sp_random_weights(
        prod_class=prod_class,
        num_rand_runs=num_rand_runs,
        results_filename=results_filename,
    )
    calc_sp_manual_weights(prod_class=prod_class, results_filename=results_filename)

    prod_class.save_class_to_pickle(file_name=_run_dir + file_name.split("/")[-1][:-4])

    print(f"The results are saved in {results_filename}")
    return results_filename


# NOTE: Uncomment to run example (OUTDATED)
# if __name__ == "__main__":
#     _file_name = "parts_data/fuegepartner_full.csv"
#     _res_dir = "results/milp_cutset_reduction_pre/filter_short_p/"

#     calculate_sp_many_weight_constants(
#         file_name=_file_name, res_dir=_res_dir, num_rand_runs=5
#     )
