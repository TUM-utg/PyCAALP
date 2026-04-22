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


def plot_line_attr_dev(attr_res: dict, attr: str, save_dir: str):

    plt.figure()
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    color_copy = MFCS_RGB.copy()
    for weight_set, val in attr_res.items():
        indices = range(len(list(val)) + 1)
        # Color: up to 4 colors i.e. limited to 4 plots
        weight_set_int = (
            int(weight_set[0]),
            int(weight_set[1]),
            int(weight_set[2]),
            int(weight_set[3]),
            weight_set[4],
        )
        attr_vals = list(val.values())
        accum_attr_vals = [sum(attr_vals[: i + 1]) for i in range(len(attr_vals))]
        accum_attr_vals.insert(0, 0.0)
        color = color_copy.pop()
        plt.plot(indices, accum_attr_vals, label=weight_set_int, color=color)
        plt.plot(indices, accum_attr_vals, ".", color=color)

    plt.title(
        f"Cumulative development of {attr.capitalize().split("_",)[1]} cost",
        fontweight="bold",
        fontname="Liberation Serif",
        fontsize=13,
    )
    plt.xlabel(
        "Assembly operation step",
        fontname="Liberation Serif",
        fontsize=11,
    )
    plt.ylabel(
        f"Cumulative {attr.split("_")[1]} cost",
        fontname="Liberation Serif",
        fontsize=11,
    )
    plt.xticks(indices)
    plt.legend(
        title=r"$(\mu_{tech}, \mu_{hand}, \mu_{tol}, \mu_{mass}, \lambda)$", fontsize=8
    )

    plot_fname = os.path.join(save_dir, f"attr_dev_{attr}.svg")
    plt.savefig(plot_fname)
    plt.close()


def plot_attr_development(attr: str, res_pkl_fname: str, save_dir: str):
    # Set the range of the data to be plotted
    iter_start = 4
    iter_end = 8
    multi_comb_res = load_pkl(res_pkl_fname)
    attr_vals = {}
    _iter = 0
    for weights, all_attr in multi_comb_res.items():
        if iter_start <= _iter <= iter_end:
            attr_vals[weights] = all_attr[attr]
        _iter += 1
    plot_line_attr_dev(attr_res=attr_vals, attr=attr, save_dir=save_dir)


def plot_attribute_development_combined(res_fname, _plot_dir):
    set_cols()
    plot_attr_development("absolute_handling", res_fname, _plot_dir)
    plot_attr_development("absolute_tolerance", res_fname, _plot_dir)
    plot_attr_development("absolute_mass", res_fname, _plot_dir)


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

    plot_attr_development("absolute_handling", RES_PKL_FNAME, PLOT_DIR)
    plot_attr_development("absolute_tolerance", RES_PKL_FNAME, PLOT_DIR)
