import numpy as np

from loguru import logger
from pycaalp.gapp.assembly_digraph import AssemblyDigraph
from pycaalp.time_balancing.model import run_mip


def run_assembly_line_planning(
    w_tech: float,
    w_hand: float,
    w_tol: float,
    w_mass: float,
    w_balanced: float,
    test_assembly_fname,
    num_phases,
    rel_gap,
    red_perc,
    dfm_fname: str = None,
):
    """Run a full assembly and line planning problem with specific weights

    Args: TODO update
            assembly_fname: assembly parts directory
            w_tech: technology weight constant
            w_hand: handling weight constant
            w_tol: tolerance weight constant
            w_balanced: time balancing weight constant

    Returns: Full operations results
    """
    assembly_digraph = AssemblyDigraph(
        file_name=test_assembly_fname,
        w_tech=w_tech,
        w_hand=w_hand,
        w_tol=w_tol,
        w_mass=w_mass,
        reduction_percentage=red_perc,
        dfm_file=dfm_fname,
    )
    assembly_digraph.compute_assembly_digraph_complete()
    model_results = run_mip(
        assembly_digraph=assembly_digraph,
        num_phases=num_phases,
        w_balanced=w_balanced,
        relative_gap=rel_gap,
        return_model=True,
    )
    return model_results


def run_full_edge_reduction_change(
    w_edge_reduction: list,
    w_tech: float,
    w_hand: float,
    w_tol: float,
    w_bal: float,
    w_mass: float,
    test_assembly_fname: str,
    num_runs: int = 3,
    dfm_fname: str = None,
    num_phases: int = 3,
    rel_gap: float = 0.0,
):
    # check edge reduction
    if len(w_edge_reduction) == 3:
        loop_range = range(
            w_edge_reduction[0], w_edge_reduction[2] + 1, w_edge_reduction[1]
        )
    else:
        loop_range = w_edge_reduction

    results = {}  # {[edge_reduction]: (model.getSolvingTime, model.getObjVal) }
    for red_perc in loop_range:
        _num_runs = num_runs
        if red_perc == 0:
            _num_runs = 1
        logger.info(f"Running test with: {red_perc=}")
        temp_solving_time = np.zeros(_num_runs)
        temp_objective_value = np.zeros(_num_runs)
        temp_max_phase_time = np.zeros(_num_runs)
        for n in range(_num_runs):
            res = run_assembly_line_planning(
                red_perc=red_perc,
                w_tech=w_tech,
                w_hand=w_hand,
                w_tol=w_tol,
                w_mass=w_mass,
                w_balanced=w_bal,
                test_assembly_fname=test_assembly_fname,
                dfm_fname=dfm_fname,
                num_phases=num_phases,
                rel_gap=rel_gap,
            )
            model_result = res[0]
            temp_solving_time[n] = model_result.getSolvingTime()
            temp_objective_value[n] = model_result.getObjVal()
            temp_max_phase_time[n] = max(res[1]["absolute_time_per_phase"].values())

        results[red_perc] = (
            np.mean(temp_solving_time),
            np.mean(temp_objective_value),
            np.mean(temp_max_phase_time),
            np.std(temp_solving_time),
            np.std(temp_objective_value),
            np.std(temp_max_phase_time),
        )

    return results
