"""Plot the k-sensitivity quality and timing curves from strategy_comparison.csv.

Standalone: reads only the CSV produced by test_strategy_comparison.py, so it can
be re-run on saved output without re-solving anything. Produces two separate
figures per (instance, num_phases, w_balanced) config found in the CSV:

    k_sensitivity_quality_<instance>_P<P>_w<wb>.svg   — objective gap to full MIP vs k
    k_sensitivity_timing_<instance>_P<P>_w<wb>.svg    — elapsed (build + solve) vs k

Run from the project root via:
    python -m experiments.strategy_comparison.plot_k_sensitivity
"""

import csv
import os

import matplotlib.pyplot as plt

CSV_FNAME = "experiments/strategy_comparison/assembly_1_np_3/strategy_comparison.csv"
FORMAT = "svg"

# TUM colour palette (grey, blue, black, green, orange, purple), as RGB 0-255.
MFCS_RGB = [
    (153, 153, 153),
    (0, 101, 189),
    (0, 0, 0),
    (159, 186, 54),
    (227, 114, 34),
    (101, 55, 142),
]

# Subgraph weight variants: (csv weight_attr, legend label, MFCS_RGB index, marker)
WEIGHT_STYLE = [
    ("edge_weight", "edge_w", 1, "o"),
    ("time_balanced_weight", "bal_w", 3, "s"),
    ("combined", "combined", 2, "v"),
    ("blended", "blended", 4, "D"),
    ("blended_union", "bl-union", 5, "X"),
]


def set_cols():
    """Return MFCS_RGB normalised from 0-255 to matplotlib 0-1 floats.

    Pure: it reads the 0-255 constant and returns a fresh list, so calling it
    more than once per process is safe (it does not re-divide a global).
    """
    return [tuple(elem / 255 for elem in sett) for sett in MFCS_RGB]


def read_rows(csv_file):
    """Read the results CSV into a list of dict rows."""
    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_config(rows):
    """Group rows by (instance, num_phases, w_balanced)."""
    configs = {}
    for row in rows:
        key = (row["instance"], row["num_phases"], row["w_balanced"])
        configs.setdefault(key, []).append(row)
    return configs


def _series(rows, weight_attr, field):
    """Sorted (k, field) series for one subgraph weight variant."""
    pts = []
    for row in rows:
        if (
            row["strategy"] == "subgraph_mip"
            and row["weight_attr"] == weight_attr
            and row["k"]
            and row[field]
        ):
            pts.append((float(row["k"]), float(row[field])))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _full_solve_s(rows):
    """Full-MIP solve time, used as the timing reference line."""
    for row in rows:
        if row["strategy"] == "full_mip" and row["solve_s"]:
            return float(row["solve_s"])
    return None


def _suffix(cfg):
    instance, num_phases, w_balanced = cfg
    return f"{instance}_P{num_phases}_w{w_balanced}"


def _title_tag(cfg):
    instance, num_phases, w_balanced = cfg
    return f"{instance}  P={num_phases}  λ={w_balanced}"


def plot_quality(rows, res_dir, cfg):
    """Objective gap to the full MIP (%) vs k, one line per weight variant."""
    cols = set_cols()
    fig, ax = plt.subplots()

    for weight_attr, label, ci, marker in WEIGHT_STYLE:
        x_vals, y_vals = _series(rows, weight_attr, "obj_vs_full_pct")
        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                "-",
                color=cols[ci],
                linewidth=1.0,
                marker=marker,
                ms=6,
                mfc=cols[ci],
                label=label,
            )

    ax.axhline(
        0.0, color=cols[0], linestyle="--", linewidth=1.0, label="full MIP (optimum)"
    )
    ax.set_xscale("log")
    ax.set_xlabel("k (shortest paths)", fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(
        "Objective gap to full MIP [%]", fontname="Liberation Serif", fontsize=11
    )
    ax.set_title(
        f"Solution quality vs. k  —  {_title_tag(cfg)}",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    plt.legend(title="Path weighting")

    fname = os.path.join(res_dir, f"k_sensitivity_quality_{_suffix(cfg)}.{FORMAT}")
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved quality plot in {fname}")
    plt.close(fig)


def plot_timing(rows, res_dir, cfg):
    """Elapsed time (build + solve) vs k, one line per weight variant."""
    cols = set_cols()
    fig, ax = plt.subplots()

    for weight_attr, label, ci, marker in WEIGHT_STYLE:
        x_vals, y_vals = _series(rows, weight_attr, "elapsed_s")
        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                "-",
                color=cols[ci],
                linewidth=1.0,
                marker=marker,
                ms=6,
                mfc=cols[ci],
                label=label,
            )

    full_t = _full_solve_s(rows)
    if full_t is not None:
        ax.axhline(
            full_t,
            color=cols[0],
            linestyle="--",
            linewidth=1.0,
            label="full MIP solve",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (shortest paths)", fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(
        "Elapsed: build + solve [s]", fontname="Liberation Serif", fontsize=11
    )
    ax.set_title(
        f"Runtime vs. k  —  {_title_tag(cfg)}",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    plt.legend(title="Path weighting")

    fname = os.path.join(res_dir, f"k_sensitivity_timing_{_suffix(cfg)}.{FORMAT}")
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved timing plot in {fname}")
    plt.close(fig)


if __name__ == "__main__":
    rows = read_rows(CSV_FNAME)
    res_dir = os.path.dirname(CSV_FNAME)
    for cfg, cfg_rows in group_by_config(rows).items():
        plot_quality(cfg_rows, res_dir, cfg)
        plot_timing(cfg_rows, res_dir, cfg)
