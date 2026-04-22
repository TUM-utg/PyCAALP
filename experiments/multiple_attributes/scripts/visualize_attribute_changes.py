import os
import argparse
import numpy as np
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


def plot_bar_attr_change(attr_change_res: dict, attr: str, save_dir: str):
    _, ax = plt.subplots()
    indices = range(len(list(attr_change_res.keys())))

    bar_labels = [str(v[4]) for v in attr_change_res.keys()]
    bar_colors = []
    col_num = 0
    new_bar_labels = []
    visited_label = []
    for i, l in enumerate(bar_labels):
        if i > 0 and bar_labels[i] != bar_labels[i - 1]:
            col_num += 1
        bar_colors.append(MFCS_RGB[-col_num - 1])

        if len(visited_label) > 0 and l in visited_label:
            new_bar_labels.append(f"_{l}")
        else:
            new_bar_labels.append(l)
            visited_label.append(l)

    ax.bar(indices, attr_change_res.values(), label=new_bar_labels, color=bar_colors)

    plt.grid(True, axis="y", linewidth=0.3, color="gray", alpha=0.4)
    # Just use the first digit for x ticks clarity
    xtick_labels = [
        str((int(v[0]), int(v[1]), int(v[2]), int(v[3]))).replace(" ", "")
        for v in attr_change_res.keys()
    ]
    plt.xticks(indices, xtick_labels, fontsize=8, rotation=0)
    # ax.xaxis.set_label_position("top")
    plt.yticks(np.arange(0, max(attr_change_res.values()) + 1, step=1))
    print(f"{max(attr_change_res.values())=}")
    plt.title(
        f"{attr.capitalize()} switching sensitivity",
        fontweight="bold",
        fontname="Liberation Serif",
        fontsize=13,
    )
    plt.ylabel(
        f"Number of {attr} changes",
        fontname="Liberation Serif",
        fontsize=11,
    )
    plt.xlabel(
        r"User-defined engineering weights $(\mu_{tech}, \mu_{hand}, \mu_{tol}, \mu_{mass})$",
        fontname="Liberation Serif",
        fontsize=11,
    )
    ax.legend(title="λ", fontsize=7)

    plot_fname = os.path.join(save_dir, f"attr_change_{attr}.svg")
    plt.savefig(plot_fname)
    plt.close()


def count_attr_changes(val):
    num_changes = 0
    for i in range(1, len(val)):
        if val[i] != val[i - 1]:
            num_changes += 1
    return num_changes


def count_num_attr_changes(multi_combination_results: dict, attr: str) -> dict:
    attr_change_results = {}
    for coeffs, attr_vals_operations in multi_combination_results.items():
        attr_vals = list(attr_vals_operations[attr].values())
        attr_change_results[coeffs] = count_attr_changes(attr_vals)
    return attr_change_results


def plot_attr_changes(attr: str, res_pkl_fname: str, save_dir: str):
    multi_comb_res = load_pkl(res_pkl_fname)
    multi_comb_num_changes = count_num_attr_changes(multi_comb_res, attr)

    plot_bar_attr_change(
        attr_change_res=multi_comb_num_changes, attr=attr, save_dir=save_dir
    )


def plot_attribute_changes_combined(res_fname, _plot_dir):
    set_cols()
    plot_attr_changes("technology", res_fname, _plot_dir)
    # plot_attr_changes("handling", res_fname, _plot_dir)
    # plot_attr_changes("tolerance", res_fname, _plot_dir)


if __name__ == "__main__":
    set_cols()
    RES_PKL_FNAME = "multiple_attributes/results/res.pkl"
    PLOT_DIR = "multiple_attributes/plots"

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

    plot_attr_changes("technology", RES_PKL_FNAME, PLOT_DIR)
    # plot_attr_changes("handling", RES_PKL_FNAME, PLOT_DIR)
    # plot_attr_changes("tolerance", RES_PKL_FNAME, PLOT_DIR)
