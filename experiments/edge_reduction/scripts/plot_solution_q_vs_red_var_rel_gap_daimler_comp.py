import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats

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


def get_val(res, ind):
    val = {}
    for key, v in res.items():
        val[key] = v[ind]
    return val


def ci95(values):
    n = len(values)
    se = stats.sem(values)  # standard error = std / sqrt(n)
    return se * stats.t.ppf(0.975, df=n - 1)  # t-critical for 95%, df=4


def plot_edge_reduction_quality(
    base_res_path, _val_base, _std_base, _val_rel_3, _std_rel_3
):
    fig, ax1 = plt.subplots()

    markers = ["o", "s", "v", "^", "p"]
    cols = set_cols()
    col = cols[1]
    col_rel = cols[2]

    x_vals = np.array(list(_val_base.keys()))
    y_vals = np.array(list(_val_base.values()))
    std_vals = np.array(list(_std_base.values()))

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

    y_vals_rel_3 = np.array(list(_val_rel_3.values()))
    std_vals_rel = np.array(list(_std_rel_3.values()))

    ax1.plot(
        x_vals,
        y_vals_rel_3,
        "-",
        color=col_rel,
        linewidth=1.0,
        marker=markers[1],
        ms=6,
        mfc=col_rel,
        label="10%",
    )

    # Add the standard deviation band
    ax1.fill_between(
        x_vals,
        y_vals_rel_3 - std_vals_rel,  # Lower bound
        y_vals_rel_3 + std_vals_rel,  # Upper bound
        color=col_rel,
        alpha=0.2,  # Transparency
        edgecolor="none",
    )
    plt.legend(title="MIP relative gap")

    # Axis Labels
    ax1.set_xlabel("Edge reduction [%]", fontname="Liberation Serif", fontsize=11)
    ax1.set_ylabel(
        "Maximum phase welding length [mm]", fontname="Liberation Serif", fontsize=11
    )

    ax2 = ax1.twinx()

    # Calculate limits for ax2 based on ax1
    baseline = min(y_vals)
    y1_min, y1_max = ax1.get_ylim()
    y2_min = ((y1_min - baseline) / baseline) * 100
    y2_max = ((y1_max - baseline) / baseline) * 100

    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel(
        "Deviation from Baseline [%]", fontname="Liberation Serif", fontsize=11
    )

    # Format ticks to show percentages
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax2.spines["right"].set_visible(True)
    plt.title(
        "Solution quality vs. Graph pruning",
        fontname="Liberation Serif",
        fontsize=13,
    )

    ax1.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    plt.tight_layout()
    # Ensure variable FORMAT is defined, otherwise default to png
    fmt = FORMAT if "FORMAT" in globals() else "png"

    res_dir = os.path.dirname(base_res_path)
    plot_filename = os.path.join(res_dir, f"quality_vs_relative_gap_comparison.{fmt}")
    plt.savefig(plot_filename, format=fmt, dpi=1200)
    print(f"Saved quality vs relative gap plot in {plot_filename}")
    plt.close()


if __name__ == "__main__":
    RES_PKL_FNAME_BASE = (
        "experiments/edge_reduction/sample_tests/daimler_n_p_5/daimler_np5.pkl"
    )
    RES_PKL_FNAME_REL_3 = (
        "experiments/edge_reduction/sample_tests/daimler_n_p_5_rel10/daimler_np5.pkl"
    )
    res_base = load_pkl(RES_PKL_FNAME_BASE)
    res_rel_3 = load_pkl(RES_PKL_FNAME_REL_3)

    val_base = get_val(res_base, 2)
    std_base = get_val(res_base, 5)
    ci95_base = {key: ci95(x) for key, x in get_val(res_base, 6).items()}

    val_rel_3 = get_val(res_rel_3, 2)
    std_rel_3 = get_val(res_rel_3, 5)
    ci95_rel_3 = {key: ci95(x) for key, x in get_val(res_rel_3, 6).items()}

    # print(get_val(res_base, 6))
    # print(get_val(res_rel_3, 6))

    plot_edge_reduction_quality(
        # RES_PKL_FNAME_BASE, val_base, std_base, val_rel_3, std_rel_3
        RES_PKL_FNAME_BASE,
        val_base,
        ci95_base,
        val_rel_3,
        ci95_rel_3,
    )
