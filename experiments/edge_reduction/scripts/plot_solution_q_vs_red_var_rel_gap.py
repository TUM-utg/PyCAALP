import os
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


def plot_edge_reduction_quality(base_res_path, val, val_rel_3):
    fig, ax1 = plt.subplots()

    markers = ["o", "s", "v", "^", "p"]
    cols = set_cols()
    col = cols[2]

    x_vals = list(val.keys())
    y_vals = list(val.values())

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

    x_vals_rel = list(val_rel_3.keys())
    y_vals_rel = list(val_rel_3.values())
    col = cols[1]
    ax1.plot(
        x_vals_rel,
        y_vals_rel,
        "-",
        color=col,
        linewidth=1.0,
        marker=markers[1],
        ms=6,
        mfc=col,
        label="3%",
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

    # Ensure variable FORMAT is defined, otherwise default to png
    fmt = FORMAT if "FORMAT" in globals() else "png"

    res_dir = os.path.dirname(base_res_path)
    plot_filename = os.path.join(res_dir, f"quality_vs_relative_gap.{fmt}")
    plt.savefig(plot_filename, format=fmt, dpi=1200)
    print(f"Saved quality vs relative gap plot in {plot_filename}")
    plt.close()


if __name__ == "__main__":
    # TODO: add the correct files
    # RES_PKL_FNAME_BASE = "edge_reduction/sample_tests/35up_n_p_3_new_dfm_rel_0_lambda_09/results/35up.pkl"
    # RES_PKL_FNAME_REL_3 = "edge_reduction/sample_tests/35up_n_p_3_new_dfm_rel_03_lambda_09/results/35up.pkl"
    res_base = load_pkl(RES_PKL_FNAME_BASE)
    res_rel_3 = load_pkl(RES_PKL_FNAME_REL_3)

    val_base = get_vals(res_base)
    print(val_base)

    val_rel_3 = get_vals(res_rel_3)
    print(val_rel_3)

    plot_edge_reduction_quality(RES_PKL_FNAME_BASE, val_base, val_rel_3)
