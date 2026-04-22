"""Main run file for mutliple attributes tests

Basic usage:
    $ python -m multiple_attributes.run

For further example tests:
    Create a new test case folder e.g., multiple_attributes/test_assembly_X
    Create a config file based on multiple_attributes/template_config.py and save it in the new folder
    Then run:
    $ python -m multiple_attributes.run --config-file <new-config-dir>

Only plots run:
    For occasions where the results are already generated and only the plots are needed.
    $ python -m multiple_attributes.run --config-file <config-dir> --only-plots
"""

import os
import argparse
import importlib.util
from loguru import logger

from experiments.multiple_attributes.scripts.combination_runs import (
    run_full_run_attr_change,
)
from experiments.multiple_attributes.scripts.visualize_attribute_changes import (
    plot_attribute_changes_combined,
)
from experiments.multiple_attributes.scripts.visualize_attribute_development import (
    plot_attribute_development_combined,
)
from experiments.multiple_attributes.scripts.visualize_time_per_phase import (
    plot_time_per_phase_combined,
)
from experiments.multiple_attributes.scripts.visualize_time_v_wb import (
    plot_time_v_wb_combined,
)
from pycaalp.gapp.file_formats import save_to_pkl, save_all_res_to_json


def import_config_dynamically(config_dir: str = "multiple_attributes/config.py"):
    """Import configuation file dynamically given the test's directory path.

    Args:
        config_dir : Configuaration file path.

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


def multiple_weight_runs(
    w_balanced: list,
    w_tech: list,
    w_hand: list,
    w_tol: list,
    w_mass: list,
    assem_fname: str,
    num_phases: int,
    res_fname: str,
    dfm_fname: str | None = None,
    rel_gap: float = 0.0,
    red_perc: int = 0,
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
    res = run_full_run_attr_change(
        line_bal_comb={"w_balanced": w_balanced},
        assemb_attr_comb={
            "w_tech": w_tech,
            "w_hand": w_hand,
            "w_tol": w_tol,
            "w_mass": w_mass,
        },
        test_assembly_fname=assem_fname,
        dfm_fname=dfm_fname,
        num_phases=num_phases,
        rel_gap=rel_gap,
        red_perc=red_perc,
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
    CONFIG_DIR = "multiple_attributes/config.py"
    args = parser.parse_args()
    if args.config:
        CONFIG_DIR = args.config
    config_data = import_config_dynamically(CONFIG_DIR).config_data
    RUN_MULTI_ATTR = True
    if args.only_plots:
        RUN_MULTI_ATTR = False

    # Set default values
    DFM_FNAME = None
    RELATIVE_GAP = 0.0
    REDUCTION_PERCENTAGE = 0

    # Upack config data
    for key, val in config_data.items():
        match key:
            case "assembly_fname":
                ASSEMBLY_FNAME = val
            case "dfm_fname":
                DFM_FNAME = val
            case "res_fname":
                RES_FNAME = val
            case "attribute_combination":
                W_TECH = val["w_tech"]
                W_HAND = val["w_hand"]
                W_TOL = val["w_tol"]
                W_MASS = val["w_mass"]
                W_BALANCED = val["w_balanced"]
            case "num_phases":
                NUM_PHASES = val
            case "relative_gap":
                RELATIVE_GAP = val
            case "reduction_percentage":
                REDUCTION_PERCENTAGE = val
            case "plots":
                PLOTS = val
            case _:
                raise AttributeError(f"{key} is not a valid config attribute")

    logger.info(f"Running {__file__}")
    logger.info(f"{ASSEMBLY_FNAME=}")
    if DFM_FNAME:
        logger.info(f"{DFM_FNAME=}")
    logger.info(f"{NUM_PHASES=}")
    logger.info(f"{RES_FNAME=}")
    # 1. Multiple weights tests
    if RUN_MULTI_ATTR:
        multiple_weight_runs(
            w_balanced=W_BALANCED,
            w_tech=W_TECH,
            w_hand=W_HAND,
            w_tol=W_TOL,
            w_mass=W_MASS,
            assem_fname=ASSEMBLY_FNAME,
            num_phases=NUM_PHASES,
            res_fname=RES_FNAME,
            dfm_fname=DFM_FNAME,
            rel_gap=RELATIVE_GAP,
            red_perc=REDUCTION_PERCENTAGE,
        )
    else:
        logger.info("Skipped multiple_weight_runs")

    # 2. Plots
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
            case "attribute_changes":
                logger.info("Running plot_attribute_changes_combined")
                plot_attribute_changes_combined(RES_FNAME, PLOT_DIR)
            case "attribute_development":
                logger.info("Running plot_attribute_development_combined")
                plot_attribute_development_combined(RES_FNAME, PLOT_DIR)
            case "time_per_phase":
                logger.info("Running plot_time_per_phase_combined")
                plot_time_per_phase_combined(RES_FNAME, PLOT_DIR)
            case "time_vs_wb":
                logger.info("Running plot_time_v_wb_combined")
                plot_time_v_wb_combined(RES_FNAME, PLOT_DIR)
