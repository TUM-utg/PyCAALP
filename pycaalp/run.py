from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_milp


def create_assembly_digraph(**kwargs) -> AssemblyDigraph:
    """
    Accepted arguments:
    - file_name: Parts data file name (JSON)
    - dfm_file: DFM file name (JSON)
    - w_tech: Techology weight. Defaults to 0.3333.
    - w_hand: Handling weight. Defaults to 0.3333.
    - w_tol: Tolerance weight. Defaults to 0.3333.
    - reduction_percentage: Edge reduction percentage

    Returns:
        AssemblyDigraph: Fully constructed assembly digraph
    """
    _assem_digr = AssemblyDigraph(**kwargs)
    _assem_digr.compute_assembly_digraph_complete()
    return _assem_digr


def optimize(**kwargs):
    """
    Accepted arguments:
    - assem_digr: Assembly digraph.
    - num_phases: Number of phases for MIP, defaults to 3.
    - w_balanced: Time balancing weight, defaults to 0.5. Range [0,1]
    - relative_gap: Relative gap for the MIP model solution. Range [0,1]
    - full_result_output: True|False

    Returns:
        [optional] result: all results per operation or phase
        best path: part connections per phase
    """
    return run_milp(**kwargs)
