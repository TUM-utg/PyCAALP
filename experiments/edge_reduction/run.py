"""Main run file for edge reduction tests

check  README.md for more
"""

import os
import argparse
import importlib.util
from loguru import logger

from experiments.edge_reduction.scripts.combination_runs import (
    run_full_edge_reduction_change,
)

from experiments.edge_reduction.scripts.plots import (
    plot_objective_value,
    plot_total_time,
    plot_wall_time,
    plot_speedup,
    plot_average_max_time,
    plot_speedup_vs_qual_lost,
    plot_speedup_vs_max_length_lost,
)
from pycaalp.gapp.file_formats import save_to_pkl, save_all_res_to_json


def import_config_dynamically(config_dir: str = "multi_attr_runs/config.py"):
    """Import configuration file dynamically given the test's directory path.

    Args:
        config_dir : Configuration file path.

    Returns:
        Imported configuration file.
    """
    if not config_dir.endswith(".py"):
        raise ValueError("configuration should be a Python file")
    # Load the module from the given path
    spec = importlib.util.spec_from_file_location("config_data", config_dir)
    _config_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_config_data)
    return _config_data


def multiple_reduction_runs(
    w_edge_reduction: list,
    w_tech: float,
    w_hand: float,
    w_tol: float,
    w_mass: float,
    w_bal: float,
    res_fname: str,
    test_assembly_fname: str,
    dfm_fname: str = None,
    num_phases: int = 3,
    rel_gap: float = 0.0,
    num_rep_runs: int = 3,
):
    """Run multiple tests with varying the 4 main weights/options.

    Args:
        w_balanced: Balancing weight values.
        w_tech: Technology weight values.
        w_hand: Handling weight values.
        w_tol: Tolerance weight values.
        w_mass: Mass weight values.
        assem_fname: Assembly parts/joints filename.
        num_phases: MIP solver number of phases.
        res_fname: Result file.
        dfm_fname: DFM matrices for the assembly.
        rel_gap: MIP solver relative gap.
    """
    res = run_full_edge_reduction_change(
        w_edge_reduction=w_edge_reduction,
        w_tech=w_tech,
        w_hand=w_hand,
        w_tol=w_tol,
        w_bal=w_bal,
        w_mass=w_mass,
        test_assembly_fname=test_assembly_fname,
        dfm_fname=dfm_fname,
        num_phases=num_phases,
        rel_gap=rel_gap,
        num_runs=num_rep_runs,
    )

    # Save the results in JSON and PKL format
    if not os.path.exists(os.path.dirname(res_fname)):
        os.makedirs(os.path.dirname(res_fname))
    save_to_pkl(res, res_fname)
    # Just to have the results easily accessible
    save_all_res_to_json(res, res_fname.replace("pkl", "json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument(
        "--config-file",
        dest="config",
        type=str,
        required=False,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--only-plots",
        dest="only_plots",
        action="store_true",
        required=False,
        help="Whether to run plots code only",
    )
    # Overrides so one base config can drive a parallel λ / P sweep
    # (see run_sweep.sh) without hand-copying a config file per point.
    parser.add_argument(
        "--w-balanced",
        dest="w_balanced",
        type=float,
        default=None,
        help="Override config's w_balanced (λ)",
    )
    parser.add_argument(
        "--num-phases",
        dest="num_phases",
        type=int,
        default=None,
        help="Override config's num_phases",
    )
    parser.add_argument(
        "--res-fname",
        dest="res_fname",
        type=str,
        default=None,
        help="Override config's res_fname (plots land next to it)",
    )
    CONFIG_DIR = "edge_reduction/config.py"
    args = parser.parse_args()
    if args.config:
        CONFIG_DIR = args.config
    config_data = import_config_dynamically(CONFIG_DIR).config_data
    RUN_RED_PERCENTAGE = True
    if args.only_plots:
        RUN_RED_PERCENTAGE = False

    # Set default values
    DFM_FNAME = None
    W_TECH = 0.25
    W_HAND = 0.25
    W_TOL = 0.25
    W_MASS = 0.25
    W_BALANCED = 0.5
    RELATIVE_GAP = 0.0
    NUM_REP_RUNS = 3

    # Unpack config data
    for key, val in config_data.items():
        match key:
            case "assembly_fname":
                ASSEMBLY_FNAME = val
            case "dfm_fname":
                DFM_FNAME = val
            case "res_fname":
                RES_FNAME = val
            case "w_tech":
                W_TECH = val
            case "w_hand":
                W_HAND = val
            case "w_tol":
                W_TOL = val
            case "w_mass":
                W_MASS = val
            case "w_balanced":
                W_BALANCED = val
            case "w_edge_reduction":
                W_EDGE_REDUCTION = val
            case "num_phases":
                NUM_PHASES = val
            case "num_rep_runs":
                NUM_REP_RUNS = val
            case "relative_gap":
                RELATIVE_GAP = val
            case "plots":
                PLOTS = val
            case _:
                raise AttributeError(f"{key} is not a valid config attribute")

    # CLI overrides win over the config so the sweep can reuse one base config
    if args.w_balanced is not None:
        W_BALANCED = args.w_balanced
    if args.num_phases is not None:
        NUM_PHASES = args.num_phases
    if args.res_fname is not None:
        RES_FNAME = args.res_fname

    logger.info(f"Running {__file__}")
    logger.info(f"{ASSEMBLY_FNAME=}")
    if DFM_FNAME:
        logger.info(f"{DFM_FNAME=}")
    logger.info(f"{NUM_PHASES=}")
    logger.info(f"{RES_FNAME=}")
    # 1. Multiple weights tests
    if RUN_RED_PERCENTAGE:
        multiple_reduction_runs(
            w_edge_reduction=W_EDGE_REDUCTION,
            w_tech=W_TECH,
            w_hand=W_HAND,
            w_tol=W_TOL,
            w_mass=W_MASS,
            w_bal=W_BALANCED,
            res_fname=RES_FNAME,
            test_assembly_fname=ASSEMBLY_FNAME,
            dfm_fname=DFM_FNAME,
            num_phases=NUM_PHASES,
            rel_gap=RELATIVE_GAP,
            num_rep_runs=NUM_REP_RUNS,
        )
    else:
        logger.info("Skipped multiple_reduction_runs")

    # 2. Plots — land next to the results (so each sweep point is self-contained
    # when --res-fname is overridden; identical to the old location otherwise,
    # since existing configs keep res.pkl in the config's own folder).
    if args.res_fname is not None:
        PLOT_DIR = os.path.join(os.path.dirname(RES_FNAME), "plots")
    else:
        PLOT_DIR = os.path.join(os.path.dirname(CONFIG_DIR), "plots")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    # Check for the results path
    if not os.path.exists(RES_FNAME):
        raise FileNotFoundError(
            f"The result file '{RES_FNAME}' does not exist. No results have been generated"
        )

    for plot_type in PLOTS:
        match plot_type:
            case "objective_value":
                logger.info("Running plot_objective_value")
                plot_objective_value(RES_FNAME, PLOT_DIR, NUM_PHASES)
            case "total_time":
                logger.info("Running plot_total_time")
                plot_total_time(RES_FNAME, PLOT_DIR, NUM_PHASES)
            case "wall_time":
                logger.info("Running plot_wall_time")
                plot_wall_time(RES_FNAME, PLOT_DIR, NUM_PHASES)
            case "speedup":
                # logger.info("Running plot_speedup")
                # plot_speedup(RES_FNAME, PLOT_DIR, NUM_PHASES)
                # logger.info("Running plot_speedup_vs_qual_lost")
                # plot_speedup_vs_qual_lost(RES_FNAME, PLOT_DIR, NUM_PHASES)
                logger.info("Running plot_speedup_vs_max_length_lost")
                plot_speedup_vs_max_length_lost(RES_FNAME, PLOT_DIR, NUM_PHASES)
            case "avg_max_time":
                logger.info("Running plot_average_max_time")
                plot_average_max_time(RES_FNAME, PLOT_DIR, NUM_PHASES)
