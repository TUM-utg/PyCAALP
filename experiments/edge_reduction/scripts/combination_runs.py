import time

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

    Returns:
        (model, operations_dict, build_s, total_s) where build_s is the
        digraph build (incl. adaptive protection + reduction) wall time and
        total_s is build + MIP construction + solve wall time.
    """
    # Wall clock: build_s covers the digraph build incl. the adaptive-protection
    # enumeration + reduction filtering; total_s covers build + MIP construction
    # + solve. The MIP's own getSolvingTime() (pure solve) is read by the caller.
    # These matter because reduction TRADES build cost for solve cost — solve
    # time alone hides the protection overhead reduction adds.
    t0 = time.perf_counter()
    assembly_digraph = AssemblyDigraph(
        file_name=test_assembly_fname,
        w_tech=w_tech,
        w_hand=w_hand,
        w_tol=w_tol,
        w_mass=w_mass,
        reduction_percentage=red_perc,
        dfm_file=dfm_fname,
        # The adaptive edge protection blends at lambda_balance and cuts phase
        # boundaries at num_phases — they must match what the MIP solves with
        lambda_balance=w_balanced,
        num_phases=num_phases,
    )
    assembly_digraph.compute_assembly_digraph_complete()
    build_s = time.perf_counter() - t0
    model_results = run_mip(
        assembly_digraph=assembly_digraph,
        num_phases=num_phases,
        w_balanced=w_balanced,
        relative_gap=rel_gap,
        return_model=True,
    )
    total_s = time.perf_counter() - t0
    return model_results[0], model_results[1], build_s, total_s


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

    # results[red_perc] = (
    #   0 mean_solve_s, 1 mean_obj, 2 mean_max_phase,
    #   3 std_solve_s,  4 std_obj,  5 std_max_phase,
    #   6 mean_total_s, 7 std_total_s, 8 mean_build_s, 9 std_build_s)
    results = {}
    for red_perc in loop_range:
        _num_runs = num_runs
        if red_perc == 0:
            _num_runs = 1
        logger.info(f"Running test with: {red_perc=}")
        temp_solving_time = np.zeros(_num_runs)
        temp_objective_value = np.zeros(_num_runs)
        temp_max_phase_time = np.zeros(_num_runs)
        temp_total_time = np.zeros(_num_runs)
        temp_build_time = np.zeros(_num_runs)
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
            temp_build_time[n] = res[2]
            temp_total_time[n] = res[3]

        results[red_perc] = (
            np.mean(temp_solving_time),
            np.mean(temp_objective_value),
            np.mean(temp_max_phase_time),
            np.std(temp_solving_time),
            np.std(temp_objective_value),
            np.std(temp_max_phase_time),
            # Appended so the existing positional consumers (indices 0-5) keep
            # working: total wall time (build+solve) and build-only time.
            np.mean(temp_total_time),
            np.std(temp_total_time),
            np.mean(temp_build_time),
            np.std(temp_build_time),
        )

    return results
