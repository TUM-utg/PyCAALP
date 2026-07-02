"""Overlay the edge-reduction curves of every λ in a sweep.

Standalone (reads only the combined CSV from ``combine_sweep.py``, never
re-solves). For each metric it draws one line per λ against reduction %, so the
quality/time trade-off of reduction can be read across λ at a glance — the
edge-reduction analogue of bl_union's ``plot_convergence``.

Run (from project root)::

    python -m experiments.edge_reduction.plot_lambda_sensitivity \
        experiments/edge_reduction/sweeps/assembly_1_np_3/edge_reduction_lambda_sweep.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MFCS_RGB = [(153, 153, 153), (0, 101, 189), (0, 0, 0), (159, 186, 54)]
MARKERS = ["o", "s", "v", "^", "p"]
FONT = "Liberation Serif"
FORMAT = "svg"

# (csv column, axis label, output basename). Total wall time is the headline
# timing metric (build incl. adaptive protection + solve); pure MIP solve time
# is kept as its own series so the protection overhead is visible.
METRICS = [
    ("mean_objective", "Objective", "objective"),
    ("mean_total_s", "Total wall time [s]", "total_time"),
    ("mean_solve_s", "MIP solve time [s]", "solve_time"),
    ("mean_max_phase_time", "Max phase time [mm]", "max_phase_time"),
]
STD_OF = {
    "mean_objective": "std_objective",
    "mean_total_s": "std_total_s",
    "mean_solve_s": "std_solve_s",
    "mean_max_phase_time": "std_max_phase_time",
}


def _cols():
    return [(r / 255, g / 255, b / 255) for r, g, b in MFCS_RGB]


def _read(csv_path: str):
    """Return {lam: {"x": [...], metric: [...], std: [...]}} sorted by red_perc."""
    per_lam = defaultdict(lambda: defaultdict(list))
    rows = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["w_balanced"]].append(row)
    for lam, lam_rows in rows.items():
        lam_rows.sort(key=lambda r: float(r["red_perc"]))
        per_lam[lam]["x"] = [float(r["red_perc"]) for r in lam_rows]
        for col, _, _ in METRICS:
            per_lam[lam][col] = [float(r[col]) for r in lam_rows]
            per_lam[lam][STD_OF[col]] = [float(r[STD_OF[col]]) for r in lam_rows]
    return per_lam


def plot_lambda_sensitivity(csv_path: str, out_dir: str | None = None) -> None:
    per_lam = _read(csv_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(out_dir, exist_ok=True)

    cols = _cols()
    lambdas = sorted(per_lam, key=float)

    for col, ylabel, basename in METRICS:
        fig, ax = plt.subplots()
        for i, lam in enumerate(lambdas):
            x = per_lam[lam]["x"]
            y = per_lam[lam][col]
            std = per_lam[lam][STD_OF[col]]
            color = cols[i % len(cols)]
            marker = MARKERS[i % len(MARKERS)]
            ax.plot(
                x, y, "--", color=color, linewidth=1.0,
                marker=marker, ms=5, mfc=color, label=f"λ={lam}",
            )
            ax.fill_between(
                x,
                [v - s for v, s in zip(y, std)],
                [v + s for v, s in zip(y, std)],
                color=color, alpha=0.12,
            )
        ax.set_xlabel("Edge reduction [%]", fontname=FONT, fontsize=11)
        ax.set_ylabel(ylabel, fontname=FONT, fontsize=11)
        ax.set_title(
            f"{ylabel} vs edge reduction", fontname=FONT, fontsize=13
        )
        ax.grid(linewidth=0.3, color="gray", alpha=0.4)
        ax.legend(prop={"family": FONT, "size": 9})
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(FONT)
        out_path = os.path.join(out_dir, f"lambda_sensitivity_{basename}.{FORMAT}")
        fig.savefig(out_path, format=FORMAT, dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="combined sweep CSV (from combine_sweep.py)")
    parser.add_argument("--out-dir", default=None, help="output dir for the SVGs")
    args = parser.parse_args()
    plot_lambda_sensitivity(args.csv, args.out_dir)
