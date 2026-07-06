"""Grid comparison: every output metric vs each of {k, %edges, elapsed time}.

Standalone; reads only a bl_union_convergence CSV. For each (instance, P) and
each subgraph method present it emits one small-multiples figure: rows are output
metrics (objective gap, makespan gap, imbalance, phase-time spread), columns are
the three x-axes the growth can be read against — k, subgraph size (% of full
graph), and the build+solve time to produce that subgraph. One line per λ.

Intended for the diverse-only, high-penalty run (``--methods diverse
--penalty <big>``), where aggressive re-routing sweeps the subgraph across the
whole graph and you want to see how *every* value moves with cost/coverage — but
it works for any run/method.

Run from the project root::

    python -m experiments.bl_union_convergence.plot_metrics_grid <csv>
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

from experiments.bl_union_convergence.plot_convergence import (
    MARKERS,
    _by_lambda,
    _lambda_colors,
    _methods,
    group_by_config,
    read_rows,
)

FORMAT = "svg"

# Output metrics (y) — "every output value" worth tracking as the subgraph grows.
METRICS = [
    ("obj_vs_full_pct", "Objective gap [%]"),
    ("vs_full_pct", "Makespan α gap [%]"),
    ("imbalance_pct", "Imbalance [%]"),
    ("phase_time_std", "Phase-time σ [s]"),
]

# x-axes to read the growth against (field, label, log-x?).
XAXES = [
    ("k", "k (paths)", True),
    ("subgraph_pct", "% of full-graph edges", True),
    ("elapsed_s", "build + solve [s]", True),
]


def _xy_sorted(k_rows, xfield, yfield):
    """(xs, ys) for one λ, sorted by x (so the line tracks the x-axis rather than
    k when x is %edges or time)."""
    pts = []
    for r in k_rows:
        if r[xfield] not in ("", None) and r[yfield] not in ("", None):
            pts.append((float(r[xfield]), float(r[yfield])))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_grid(cfg, cfg_rows, method, res_dir):
    by_lam = _by_lambda(cfg_rows, method=method)
    if not by_lam:
        return
    lambdas = sorted(by_lam)
    colors = _lambda_colors(lambdas)

    nrow, ncol = len(METRICS), len(XAXES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)

    for ri, (yfield, ylabel) in enumerate(METRICS):
        for ci, (xfield, xlabel, xlog) in enumerate(XAXES):
            ax = axes[ri][ci]
            for i, lam in enumerate(lambdas):
                xs, ys = _xy_sorted(by_lam[lam], xfield, yfield)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    ys,
                    "-",
                    color=colors[lam],
                    linewidth=1.0,
                    marker=MARKERS[i % len(MARKERS)],
                    ms=4,
                    mfc=colors[lam],
                    label=f"λ={lam:g}",
                )
            if xlog:
                ax.set_xscale("log")
            if ri == nrow - 1:
                ax.set_xlabel(xlabel, fontname="Liberation Serif", fontsize=10)
            if ci == 0:
                ax.set_ylabel(ylabel, fontname="Liberation Serif", fontsize=10)
            ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    instance, num_phases = cfg
    fig.suptitle(
        f"{method} — every metric vs k / %edges / time  ({instance}  P={num_phases})",
        fontname="Liberation Serif",
        fontsize=13,
    )
    # One shared legend (λ) to the right.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="λ", fontsize=8, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))

    fname = os.path.join(
        res_dir, f"metrics_grid_{method}_{instance}_P{num_phases}.{FORMAT}"
    )
    fig.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    csv_fname = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv_fname:
        sys.exit(
            "usage: python -m experiments.bl_union_convergence.plot_metrics_grid <csv>"
        )
    rows = read_rows(csv_fname)
    res_dir = os.path.dirname(csv_fname) or "."
    for cfg, cfg_rows in group_by_config(rows).items():
        for method in _methods(cfg_rows):
            plot_grid(cfg, cfg_rows, method, res_dir)
