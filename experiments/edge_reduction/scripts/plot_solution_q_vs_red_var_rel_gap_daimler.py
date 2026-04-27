import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from pycaalp.gapp.file_formats import load_pkl

FORMAT = "svg"
MFCS_RGB = [(153, 153, 153), (0, 101, 189), (0, 0, 0), (159, 186, 54)]


def set_cols():
    for i, sett in enumerate(MFCS_RGB):
        temp_list = []
        for elem in sett:
            temp_list.append(elem / 255)
        MFCS_RGB[i] = (temp_list[0], temp_list[1], temp_list[2])
    return MFCS_RGB


def get_vals(res):
    val = {}
    for key, v in res.items():
        val[key] = v[2]
    return val


def get_std(res):
    val = {}
    for key, v in res.items():
        val[key] = v[5]
    return val


def plot_edge_reduction_quality(base_res_path, val, std):
    fig, ax1 = plt.subplots()

    markers = ["o", "s", "v", "^", "p"]
    cols = set_cols()
    col = cols[2]

    x_vals = np.array(list(val.keys()))
    y_vals = np.array(list(val.values()))
    std_vals = np.array(list(std.values()))

    # --- PLOT PRIMARY DATA (Left Axis) ---
    ax1.plot(
        x_vals,
        y_vals,
        "-",
        color=col,
        linewidth=1.0,
        marker=markers[0],
        ms=6,
        mfc=col,
        label="0%",
    )

    # Add the standard deviation band
    ax1.fill_between(
        x_vals,
        y_vals - std_vals,  # Lower bound
        y_vals + std_vals,  # Upper bound
        color=col,
        alpha=0.2,  # Transparency
        edgecolor="none",
    )

    plt.legend(title="MIP relative gap")

    # Axis Labels
    ax1.set_xlabel("Edge reduction [%]", fontname="Liberation Serif", fontsize=11)
    ax1.set_ylabel(
        "Maximum phase welding length [mm]", fontname="Liberation Serif", fontsize=11
    )

    plt.title(
        "Solution quality vs. Graph pruning",
        fontname="Liberation Serif",
        fontsize=13,
    )

    ax1.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    # Ensure variable FORMAT is defined, otherwise default to png
    fmt = FORMAT if "FORMAT" in globals() else "png"

    res_dir = os.path.dirname(base_res_path)
    plot_filename = os.path.join(res_dir, f"quality_vs_relative_gap.{fmt}")
    plt.savefig(plot_filename, format=fmt, dpi=1200)
    print(f"Saved quality vs relative gap plot in {plot_filename}")
    plt.close()


if __name__ == "__main__":
    # Add the correct files
    RES_PKL_FNAME_BASE = (
        "experiments/edge_reduction/sample_tests/daimler_n_p_5/daimler_np5.pkl"
    )
    res_base = load_pkl(RES_PKL_FNAME_BASE)

    val_base = get_vals(res_base)
    std_base = get_std(res_base)
    # time_base = get_time(res_base)
    print(val_base)
    print(std_base)

    plot_edge_reduction_quality(RES_PKL_FNAME_BASE, val_base, std_base)
