"""NOTE: use this when all the balancing weight is greater than zero (w_b>0)
This ensures that the time balancing part is run and there is at least one operation per phase.
"""

import os
import argparse
import matplotlib.pyplot as plt
from loguru import logger

from pycaalp.gapp.file_formats import load_pkl

MFCS_RGB = [(159, 186, 54), (153, 153, 153), (0, 101, 189), (0, 0, 0)]


def set_cols():
    for i, sett in enumerate(MFCS_RGB):
        temp_list = []
        for elem in sett:
            temp_list.append(elem / 255)
        MFCS_RGB[i] = (temp_list[0], temp_list[1], temp_list[2])


def plot_points_time_per_phase(attr_res: dict, attr: str, save_dir: str):

    _, ax = plt.subplots()

    indices = range(len(list(attr_res.keys())))
    bar_labels = [str(v[3]) for v in attr_res.keys()]
    bar_colors = []
    col_num = 0
    new_bar_labels = []
    visited_label = []
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    for i, l in enumerate(bar_labels):
        if i > 0 and bar_labels[i] != bar_labels[i - 1]:
            col_num += 1
        bar_colors.append(MFCS_RGB[-col_num - 1])

        if len(visited_label) > 0 and l in visited_label:
            new_bar_labels.append(f"_{l}")
        else:
            new_bar_labels.append(l)
            visited_label.append(l)

    for i, phase_times in enumerate(attr_res.values()):
        time_values = list(phase_times.values())
        plt.plot(
            [i] * len(time_values),
            time_values,
            "--.",
            color=bar_colors[i],
            linewidth=0.6,
            ms=12,
            mfc=bar_colors[i],
            label=new_bar_labels[i],
        )

    xtick_labels = [v[:3] for v in attr_res.keys()]
    plt.xticks(indices, xtick_labels, fontsize=5, rotation=10, fontweight="bold")

    plt.title(f"{attr} vs weights (tech,hand,tol)", fontweight="bold")
    plt.ylabel(f"{attr} weight changes")
    # plt.xlabel("technology, handling, tolerance weights")
    ax.legend(title="balancing weight", fontsize=8)

    plot_fname = os.path.join(save_dir, f"{attr}.svg")
    plt.savefig(plot_fname)
    plt.close()


def plot_time_per_phase(attr_name: str, res_pkl_fname: str, save_dir: str):

    multi_comb_res = load_pkl(res_pkl_fname)
    attr_vals = {}
    for weights, all_attr in multi_comb_res.items():
        attr_vals[weights] = all_attr[attr_name]

    plot_points_time_per_phase(attr=attr_name, attr_res=attr_vals, save_dir=save_dir)


def plot_time_per_phase_combined(res_fname, _plot_dir):
    set_cols()
    plot_time_per_phase("time_per_phase", res_fname, _plot_dir)
    plot_time_per_phase("absolute_time_per_phase", res_fname, _plot_dir)


if __name__ == "__main__":
    set_cols()
    RES_PKL_FNAME = "multi_attr_runs/results/res.pkl"
    PLOT_DIR = "multi_attr_runs/plots"

    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument(
        "--res-pkl-fname",
        dest="res_pkl_fname",
        type=str,
        required=False,
        help="Path to the pkl file result",
    )
    parser.add_argument(
        "--plot-dir",
        dest="plot_dir",
        type=str,
        required=False,
        help="Dir of the resulted plots",
    )
    args = parser.parse_args()
    if args.res_pkl_fname:
        RES_PKL_FNAME = args.res_pkl_fname

    if args.plot_dir:
        plot_dir = args.plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        PLOT_DIR = plot_dir

    logger.info(f"Running {__file__}")
    logger.info(f"{RES_PKL_FNAME=}")
    logger.info(f"{PLOT_DIR=}")

    plot_time_per_phase("time_per_phase", RES_PKL_FNAME, PLOT_DIR)
    plot_time_per_phase("absolute_time_per_phase", RES_PKL_FNAME, PLOT_DIR)
