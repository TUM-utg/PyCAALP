# pylint: disable=C0103, C0116
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


if __name__ == "__main__":
    # Assembly digraph options
    file_name = "data/assembly_1/assembly_1_2_tech_parts.json"

    w_tech = 1.0
    w_hand = 0.0
    w_tol = 0.0
    edge_reduction_percentage = 0

    # MIP options
    num_phases = 3
    w_balanced = 0.5
    relative_gap = 0.0

    # 1st create the assembly graph
    assembly_digraph = create_assembly_digraph(
        file_name=file_name,
        w_tech=w_tech,
        w_hand=w_hand,
        w_tol=w_tol,
        # reduction_percentage=edge_reduction_percentage,
        # dfm_file=dfm_file,
        # log_format="DEBUG",
    )

    # 2nd run MIP
    result, best_path = optimize(
        assembly_digraph=assembly_digraph,
        relative_gap=relative_gap,
        num_phases=num_phases,
        w_balanced=w_balanced,
        full_result_output=True,
        hide_output=False,
    )
    print(best_path)
    print(f"{result["absolute_time_per_phase"]=}")
