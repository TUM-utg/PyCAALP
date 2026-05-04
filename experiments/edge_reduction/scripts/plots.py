import os
import matplotlib.pyplot as plt
from matplotlib import ticker

from pycaalp.gapp.file_formats import load_pkl

MFCS_RGB = [(153, 153, 153), (0, 101, 189), (0, 0, 0), (159, 186, 54)]
FORMAT = "svg"


def set_cols():
    cols = []
    for _, sett in enumerate(MFCS_RGB):
        temp_list = []
        for elem in sett:
            temp_list.append(elem / 255)
        cols.append((temp_list[0], temp_list[1], temp_list[2]))
    return cols


def plot_edge_reduction_quality(
    val,
    res_dir,
    num_phases,
    plot_type,
    y_units,
    std=None,
    speedup_line=False,
    log=False,
    title="",
):
    markers = ["o", "s", "v", "^", "p"]

    # 1. Use subplots to allow for dual axes
    # fig, ax1 = plt.subplots(dpi=200, figsize=(8, 6))
    fig, ax1 = plt.subplots()

    # Use your color function if available, else default to black
    cols = set_cols()
    col = cols[2]

    x_vals = list(val.keys())
    y_vals = list(val.values())

    # --- PLOT PRIMARY DATA (Left Axis) ---
    ax1.plot(
        x_vals,
        y_vals,
        "--",
        color=col,
        linewidth=1.0,
        marker=markers[0],
        ms=6,
        mfc=col,
    )
    if std is not None:
        plt.fill_between(
            list(val.keys()),
            [y - s for y, s in zip(val.values(), std.values())],
            [y + s for y, s in zip(val.values(), std.values())],
            color=col,
            alpha=0.2,
            edgecolor="none",
        )

    # --- LOGIC FOR QUALITY / WELDING LENGTH PLOTS ---
    # Only applies special formatting if this is the "Quality" graph
    if "quality" in title.lower() or "length" in plot_type.lower():
        baseline = min(y_vals)

        # # A. Add the Green "Safe Zone" (0% to 60%)
        # if abs(baseline - y_vals[0]) < 1e-3:
        #     ax1.axvspan(0, 60, color="green", alpha=0.08, label="Stable Zone")
        #     ax1.text(
        #         30,
        #         min(y_vals) + 0.1,
        #         "No Quality Loss",
        #         color="green",
        #         fontsize=10,
        #         ha="center",
        #         va="bottom",
        #         fontweight="bold",
        #         alpha=0.6,
        #     )

        # B. Add Annotation for the jump at the end
        # if abs(baseline - y_vals[0]) > 1e-3:
        # baseline = 900  # Hardcoded the rel percentage 0 case
        # last_val = y_vals[-1]
        # pct_change = ((last_val - baseline) / baseline) * 100

        # Only annotate if there is a jump
        # if pct_change > 0 and abs(baseline - y_vals[0]) < 1e-3:
        #     ax1.annotate(
        #         f"+{pct_change:.2f}% (Negligible)",
        #         xy=(x_vals[-1], last_val),
        #         xytext=(
        #             x_vals[-1] - 15,
        #             last_val,
        #         ),  # Position text to the left of the dot
        #         arrowprops=dict(arrowstyle="->", color="black"),
        #         color="red",
        #         fontsize=10,
        #         ha="right",
        #     )

        # C. Add Secondary Axis (Right Side) for % Deviation
        ax2 = ax1.twinx()

        # Calculate limits fo ax2 based on ax1
        y1_min, y1_max = ax1.get_ylim()
        y2_min = ((y1_min - baseline) / baseline) * 100
        y2_max = ((y1_max - baseline) / baseline) * 100

        ax2.set_ylim(y2_min, y2_max)
        ax2.set_ylabel(
            "Deviation from Baseline [%]", fontname="Liberation Serif", fontsize=11
        )

        # Format ticks to show percentages
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax2.spines["right"].set_visible(True)

    # --- STANDARD PLOT ELEMENTS ---

    # Log scale if requested
    if log:
        ax1.set_yscale("log")

    # Speedup reference line
    if speedup_line:
        ax1.axhline(y=1, color="black", linestyle="--", linewidth=1)

    # Axis Labels
    ax1.set_xlabel("Edge reduction [%]", fontname="Liberation Serif", fontsize=11)
    ax1.set_ylabel(plot_type, fontname="Liberation Serif", fontsize=11)

    plt.title(
        title,
        fontname="Liberation Serif",
        fontsize=13,
    )

    ax1.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    # Saving
    title_plot_type_str = plot_type.replace(" ", "_")

    # Ensure variable FORMAT is defined, otherwise default to png
    fmt = FORMAT if "FORMAT" in globals() else "png"

    plot_filename = os.path.join(
        res_dir, f"{title_plot_type_str}_num_phases_{num_phases}.{fmt}"
    )
    plt.savefig(plot_filename, format=fmt, dpi=1200)
    print(f"Saved {plot_type} plot in {plot_filename}")
    plt.close()


def plot_edge_reduction(
    val,
    res_dir,
    num_phases,
    plot_type,
    y_units="",
    y_label="",
    speedup_line=False,
    log=False,
    title="",
    std=None,
    x_label="Edge reduction [%]",
):
    markers = ["o", "s", "v", "^", "p"]
    # mfcs = ["r", "g", "k", "c", "y"]
    plt.figure(dpi=200)
    cols = set_cols()
    col = cols[2]
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.plot(
        val.keys(),
        val.values(),
        "--",
        color=col,
        linewidth=1.0,
        marker=markers[0],
        ms=6,
        mfc=col,
        # label=sh_path_type,
    )

    if std is not None:
        plt.fill_between(
            list(val.keys()),
            [y - s for y, s in zip(val.values(), std.values())],
            [y + s for y, s in zip(val.values(), std.values())],
            color=col,
            alpha=0.2,
            edgecolor="none",
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # for x, y in zip(val.keys(), val.values()):
    #     plt.text(
    #         x,
    #         y * 1.05,
    #         f"{y:.1f}",  # format as needed
    #         fontsize=9,
    #         ha="right",  # horizontal alignment
    #         va="bottom",  # vertical alignment
    #         color=MFCS_RGB[0],
    #     )
    if log:
        plt.yscale("log")

    if speedup_line:
        plt.axhline(y=1, color="black", linestyle="--", linewidth=1)

    plt.title(
        title,
        fontname="Liberation Serif",
        fontsize=13,
        # fontweight="bold",
    )
    # plt.legend(fontsize=14)
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    title_plot_type_str = plot_type.replace(" ", "_")
    plot_filename = os.path.join(
        res_dir, f"{title_plot_type_str}_num_phases_{num_phases}.{FORMAT}"
    )
    plt.savefig(plot_filename, format=FORMAT, dpi=1200)
    print(f"Saved {plot_type} plot in {plot_filename}")
    plt.close()


def plot_objective_value(res_pkl_fname: str, res_dir: str, num_phases: int):
    set_cols()
    res = load_pkl(res_pkl_fname)
    val = {}
    std = {}
    for key, v in res.items():
        val[key] = v[1]
        std[key] = v[4]

    plot_edge_reduction(
        val=val,
        res_dir=res_dir,
        num_phases=num_phases,
        plot_type="Objective Value",
        y_units="-",
    )


def plot_total_time(res_pkl_fname: str, res_dir: str, num_phases: int):
    set_cols()
    res = load_pkl(res_pkl_fname)
    print(res)
    val = {}
    std = {}
    for key, v in res.items():
        val[key] = float(v[0] / 60)
        std[key] = float(v[3] / 60)

    plot_edge_reduction(
        val=val,
        res_dir=res_dir,
        num_phases=num_phases,
        plot_type="Total time",
        title="MIP solving time",
        std=std,
        y_label="Total time [min]",
    )


def plot_speedup_vs_qual_lost(res_pkl_fname, res_dir: str, num_phases: int):
    res = load_pkl(res_pkl_fname)
    if 0 not in res:
        raise ValueError("Speedup plot can't be generated: No data for reduction %=0")
    base_time = res[0][0]
    base_time_std = res[0][3]
    speedup_val = {}
    speedup_std = {}
    # Calculate speedups
    for key, v in res.items():
        if key == 0:
            continue
        t, t_std = v[0], v[3]
        t_max = t + t_std

        s = base_time / t
        speedup_val[key] = s

        s_std = base_time / t_max
        speedup_std[key] = abs(s_std - s)

    # Calculate solution degradation
    base_obj = res[0][1]
    obj_val = {}
    obj_std = {}
    for key, v in res.items():
        if key == 0:
            continue
        o, o_std = v[1], v[4]
        o_max = o + o_std

        d = (o - base_obj) / base_obj * 100
        obj_val[key] = d

        d_std = (o_max - base_obj) / base_obj * 100
        obj_std[key] = abs(d_std - d)

    print(obj_std)
    print(speedup_std)

    plt.figure(dpi=200)
    cols = set_cols()
    col = cols[2]
    x = []
    xerr = []
    y = []  # degradation
    yerr = []

    for key, v in speedup_val.items():
        x.append(v)
        xerr.append(speedup_std[key])
        y.append(obj_val[key])
        yerr.append(obj_std[key])
    plt.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=2, elinewidth=1.0, capthick=1
    )
    plt.plot(
        x,
        y,
        "-",
        color=col,
        linewidth=1.0,
        marker="o",
        ms=6,
        mfc=col,
    )
    plt.xscale("log")

    plt.ylabel(
        "Objective value deviation [%]", fontname="Liberation Serif", fontsize=11
    )
    plt.xlabel("Speedup [-]", fontname="Liberation Serif", fontsize=13)

    plt.title(
        "Computational Speedup vs. Solution Quality Degradation",
        fontname="Liberation Serif",
        fontsize=13,
        # fontweight="bold",
    )
    # plt.legend(fontsize=14)
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    title_plot_type_str = "speedup_vs_obj_degr"
    plot_filename = os.path.join(
        res_dir, f"{title_plot_type_str}_num_phases_{num_phases}.{FORMAT}"
    )
    plt.savefig(plot_filename, format=FORMAT, dpi=1200)
    print(f"Saved {title_plot_type_str} plot in {plot_filename}")
    plt.close()


def plot_speedup_vs_max_length_lost(res_pkl_fname, res_dir: str, num_phases: int):
    res = load_pkl(res_pkl_fname)
    if 0 not in res:
        raise ValueError("Speedup plot can't be generated: No data for reduction %=0")
    base_time = res[0][0]
    speedup_val = {}
    # Calculate speedups
    for key, v in res.items():
        if key == 0:
            continue
        speedup_val[key] = base_time / v[0]

    # Calculate solution degradation
    base_obj_val = res[0][2]
    obj_val = {}
    for key, v in res.items():
        if key == 0:
            continue
        obj_val[key] = (v[2] - base_obj_val) / base_obj_val * 100

    val = {}
    for key, speedup in speedup_val.items():
        val[speedup] = obj_val[key]

    plt.figure(dpi=200)
    cols = set_cols()
    col = cols[2]
    plt.plot(
        val.values(),
        val.keys(),
        "--",
        color=col,
        linewidth=1.0,
        marker="o",
        ms=6,
        mfc=col,
        # label=sh_path_type,
    )
    plt.yscale("log")
    # plt.xscale("log")

    plt.ylabel("Speedup [-]", fontname="Liberation Serif", fontsize=11)
    plt.xlabel(
        "Maximumum phase welding length deviation [%]",
        fontname="Liberation Serif",
        fontsize=11,
    )

    plt.title(
        "Computational Speedup vs. Solution Quality Degradation",
        fontname="Liberation Serif",
        fontsize=13,
        # fontweight="bold",
    )
    # plt.legend(fontsize=14)
    plt.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    title_plot_type_str = "speedup_vs_weld_len_degr"
    plot_filename = os.path.join(
        res_dir, f"{title_plot_type_str}_num_phases_{num_phases}.{FORMAT}"
    )
    plt.savefig(plot_filename, format=FORMAT, dpi=1200)
    print(f"Saved {title_plot_type_str} plot in {plot_filename}")
    plt.close()


def plot_speedup(res_pkl_fname, res_dir: str, num_phases: int):
    set_cols()
    res = load_pkl(res_pkl_fname)

    if 0 not in res:
        raise ValueError("Speedup plot can't be generated: No data for reduction %=0")
    base_time = res[0][0]
    val = {}
    for key, v in res.items():
        if key == 0:
            continue
        val[key] = base_time / v[0]

    plot_edge_reduction(
        val=val,
        res_dir=res_dir,
        num_phases=num_phases,
        plot_type="Computational speedup",
        title="Computational Speedup vs. Search Space Reduction",
        y_units="-",
        speedup_line=True,
        log="True",
    )


def plot_average_max_time(res_pkl_fname: str, res_dir: str, num_phases: int):
    set_cols()
    res = load_pkl(res_pkl_fname)
    val = {}
    std = {}
    for key, v in res.items():
        val[key] = v[2]
        std[key] = v[5]

    plot_edge_reduction_quality(
        val=val,
        res_dir=res_dir,
        num_phases=num_phases,
        std=std,
        title="Solution quality vs. Graph Pruning",
        plot_type="Max welding length [mm]",
        y_units="s",
    )
