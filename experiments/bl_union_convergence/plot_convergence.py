"""Plot bl-union objective convergence from bl_union_convergence.csv.

Standalone: reads only the CSV produced by bl_union_convergence.py, so it can be
re-run on saved output without re-solving anything. For each (instance, P) config
it produces two figures, one line per λ:

    convergence_obj_vs_pct_<instance>_P<P>.svg  — obj gap to full MIP vs % edges
    convergence_obj_vs_k_<instance>_P<P>.svg    — obj gap to full MIP vs k

The "% edges" figure is the graph-agnostic one: k-shortest paths overlap, so the
k that reaches convergence does not transfer across assemblies, but the subgraph
*edge fraction* at which the objective plateaus does. A star marks the
auto-stop point (objective plateau / edge saturation) on each λ line.

Run from the project root via:
    python -m experiments.bl_union_convergence.plot_convergence <csv>
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

DEFAULT_CSV = "experiments/bl_union_convergence/assembly_2_np_3/bl_union_convergence.csv"
FORMAT = "svg"
MARKERS = ["o", "s", "v", "^", "p", "D", "X", "<", ">", "*"]


def read_rows(csv_file):
    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_config(rows):
    """Group rows by (instance, num_phases)."""
    configs = {}
    for row in rows:
        configs.setdefault((row["instance"], row["num_phases"]), []).append(row)
    return configs


def _by_lambda(cfg_rows):
    """Sub-group a config's rows by λ, returning a dict {λ_float: [k-rows]} with
    only the swept k rows (the full_ref row, which has no k, is dropped)."""
    out = {}
    for row in cfg_rows:
        if not row["k"]:
            continue
        out.setdefault(float(row["w_balanced"]), []).append(row)
    for lam in out:
        out[lam].sort(key=lambda r: float(r["k"]))
    return out


def _xy(k_rows, xfield, yfield):
    xs, ys = [], []
    for r in k_rows:
        if r[xfield] and r[yfield] != "":
            xs.append(float(r[xfield]))
            ys.append(float(r[yfield]))
    return xs, ys


def _stop_point(k_rows, xfield, yfield):
    """(x, y) of the auto-stop row for this λ, or None."""
    for r in k_rows:
        if r["stop_reason"] and r["stop_reason"] != "full_ref" and r[xfield] and r[yfield] != "":
            return float(r[xfield]), float(r[yfield])
    return None


def _lambda_colors(lambdas):
    """Map each λ to a colour along viridis (λ is a continuous sweep dim, so a
    gradient reads better than the discrete TUM palette here)."""
    lo, hi = min(lambdas), max(lambdas)
    norm = mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
    return {lam: cm.viridis(norm(lam)) for lam in lambdas}


def _plot(cfg, by_lam, res_dir, xfield, xlabel, xlog, yfield, ylabel, fname_stem):
    fig, ax = plt.subplots()
    lambdas = sorted(by_lam)
    colors = _lambda_colors(lambdas)

    for i, lam in enumerate(lambdas):
        k_rows = by_lam[lam]
        xs, ys = _xy(k_rows, xfield, yfield)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            "-",
            color=colors[lam],
            linewidth=1.0,
            marker=MARKERS[i % len(MARKERS)],
            ms=5,
            mfc=colors[lam],
            label=f"λ={lam:g}",
        )
        sp = _stop_point(k_rows, xfield, yfield)
        if sp:
            ax.plot(sp[0], sp[1], marker="*", ms=13, color=colors[lam], mec="black", mew=0.6)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="full MIP")
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(ylabel, fontname="Liberation Serif", fontsize=11)
    instance, num_phases = cfg
    ax.set_title(
        f"bl-union convergence  —  {instance}  P={num_phases}",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(title="λ (★ = auto-stop)", fontsize=8, ncol=2)

    fname = os.path.join(res_dir, f"{fname_stem}_{instance}_P{num_phases}.{FORMAT}")
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    csv_fname = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = read_rows(csv_fname)
    res_dir = os.path.dirname(csv_fname) or "."
    for cfg, cfg_rows in group_by_config(rows).items():
        by_lam = _by_lambda(cfg_rows)
        if not by_lam:
            continue
        # Growth curves vs subgraph size (% of full-graph edges): objective gap
        # and makespan (alpha) gap. Plus the objective vs k for reference.
        _plot(
            cfg, by_lam, res_dir,
            xfield="subgraph_pct",
            xlabel="Subgraph size [% of full-graph edges]",
            xlog=True,
            yfield="obj_vs_full_pct",
            ylabel="Objective gap to full MIP [%]",
            fname_stem="convergence_obj_vs_pct",
        )
        _plot(
            cfg, by_lam, res_dir,
            xfield="subgraph_pct",
            xlabel="Subgraph size [% of full-graph edges]",
            xlog=True,
            yfield="vs_full_pct",
            ylabel="Makespan (α) gap to full MIP [%]",
            fname_stem="convergence_alpha_vs_pct",
        )
        _plot(
            cfg, by_lam, res_dir,
            xfield="k",
            xlabel="k (shortest paths)",
            xlog=True,
            yfield="obj_vs_full_pct",
            ylabel="Objective gap to full MIP [%]",
            fname_stem="convergence_obj_vs_k",
        )
