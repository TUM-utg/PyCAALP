"""Basic run of assembly digraph.

Create an assembly digraph from an assembly file.
"""

import time
from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_milp


def create_pkl(
    _filename_parts: str,
    _filename_pickle: str = "temp.pkl",
    dfm_json_file: str = None,
    # _log_format: str = "DEBUG",
):
    """Create a pickle file from the assembly digraph."""
    start = time.time()
    if dfm_json_file:
        assembly_digraph = AssemblyDigraph(
            file_name=_filename_parts,
            dfm_file=dfm_json_file,
            # log_format=_log_format,
        )
    else:
        assembly_digraph = AssemblyDigraph(
            file_name=_filename_parts,
            reduction_percentage=0,
            # log_format=_log_format,
            # edge_weight_constants=(w_tech, w_hand, w_tol),
        )

    assembly_digraph.generate_assembly_digraph_file_complete(file_name=_filename_pickle)
    end = time.time()
    print(f"Time to generate assembly digraph: {end - start}")

    return assembly_digraph


if __name__ == "__main__":

    # Create PKL file with assembly digraph
    # Part Data
    FILENAME_PARTS = "data/assembly_1/assembly_1_2_tech_parts.json"

    # Output file
    FILENAME_PICKLE = "assembly_1.pkl"

    # Freedom matrices files
    create_pkl(
        FILENAME_PARTS,
        FILENAME_PICKLE,
        # dfm_json_file=FILENAME_DFM,
    )

    # Run MILP using the PKL file. Use default settings.
    # Write results in "balancing_results" folder
    model, res_dict = run_milp(pickle_filename=FILENAME_PICKLE, w_balanced=10)
    print(res_dict)
