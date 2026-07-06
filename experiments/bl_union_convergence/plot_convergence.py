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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

DEFAULT_CSV = (
    "experiments/bl_union_convergence/assembly_2_np_3_methods/bl_union_convergence.csv"
)
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


def _methods(cfg_rows):
    """Subgraph methods present in the config, excluding the full_ref row.

    Falls back to a single 'bl_union' bucket for CSVs written before the method
    column existed (row.get returns None → treated as the sole method)."""
    ms = []
    for row in cfg_rows:
        if not row["k"]:
            continue
        m = row.get("method") or "bl_union"
        if m not in ms:
            ms.append(m)
    return ms


def _by_lambda(cfg_rows, method=None):
    """Sub-group a config's swept-k rows by λ (optionally filtered to one
    method). The full_ref row (no k) is dropped."""
    out = {}
    for row in cfg_rows:
        if not row["k"]:
            continue
        if method is not None and (row.get("method") or "bl_union") != method:
            continue
        out.setdefault(float(row["w_balanced"]), []).append(row)
    for lam in out:
        out[lam].sort(key=lambda r: float(r["k"]))
    return out


def _xy(k_rows, xfield, yfield):
    xs, ys = [], []
    for r in k_rows:
        # .get: tolerate CSVs written before a derived column (e.g.
        # alpha_vs_width_pct) existed, so old runs still plot the other curves.
        if r.get(xfield) and r.get(yfield, "") != "":
            xs.append(float(r[xfield]))
            ys.append(float(r[yfield]))
    return xs, ys


def _stop_point(k_rows, xfield, yfield):
    """(x, y) of the auto-stop row for this λ, or None."""
    for r in k_rows:
        if (
            r["stop_reason"]
            and r["stop_reason"] != "full_ref"
            and r.get(xfield)
            and r.get(yfield, "") != ""
        ):
            return float(r[xfield]), float(r[yfield])
    return None


# TUM colour palette (grey, blue, black, green, orange, purple), as RGB 0-255 —
# matches strategy_comparison's MFCS_RGB. λ is a continuous sweep, so we build a
# gradient from the TUM anchors rather than picking discrete swatches.
MFCS_RGB = [
    (153, 153, 153),
    (0, 101, 189),
    (0, 0, 0),
    (159, 186, 54),
    (227, 114, 34),
    (101, 55, 142),
]

# Gradient anchors along λ: grey → TUM blue → TUM green.
_TUM_GRADIENT_ANCHORS = [MFCS_RGB[0], MFCS_RGB[1], MFCS_RGB[3]]
_TUM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "tum", [tuple(c / 255 for c in rgb) for rgb in _TUM_GRADIENT_ANCHORS]
)


def _lambda_colors(lambdas):
    """Map each λ to a colour along a TUM-branded gradient (λ is a continuous
    sweep dim, so a gradient reads better than discrete swatches; the anchors are
    the TUM palette so it matches the other experiments)."""
    lo, hi = min(lambdas), max(lambdas)
    norm = mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
    return {lam: _TUM_CMAP(norm(lam)) for lam in lambdas}


def _plot(
    cfg, by_lam, res_dir, xfield, xlabel, xlog, yfield, ylabel, fname_stem, method,
    baseline_label="full MIP",
):
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
            ax.plot(
                sp[0], sp[1], marker="*", ms=13, color=colors[lam], mec="black", mew=0.6
            )

    ax.axhline(
        0.0, color="black", linestyle="--", linewidth=1.0, label=baseline_label
    )
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(ylabel, fontname="Liberation Serif", fontsize=11)
    instance, num_phases = cfg
    ax.set_title(
        f"{method.capitalize()} method convergence",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(title="λ (★ = auto-stop)", fontsize=8, ncol=2)

    fname = os.path.join(
        res_dir, f"{fname_stem}_{method}_{instance}_P{num_phases}.{FORMAT}"
    )
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved {fname}")
    plt.close(fig)


# Fixed colours/markers per method for the head-to-head figure (TUM palette).
METHOD_STYLE = {
    "bl_union": ((0 / 255, 101 / 255, 189 / 255), "o", "bl-union"),
    "diverse": ((159 / 255, 186 / 255, 54 / 255), "s", "diverse"),
}


def plot_best_gap_vs_lambda(cfg, cfg_rows, res_dir):
    """Head-to-head: best objective gap each method reaches vs λ.

    For every (method, λ) take the minimum obj gap over the k-sweep (the growth
    is monotone, so this is the converged value). This is the idea-#4 figure:
    it shows whether diverse enumeration closes the high-λ gap where bl-union
    plateaus."""
    fig, ax = plt.subplots()
    for method in _methods(cfg_rows):
        by_lam = _by_lambda(cfg_rows, method=method)
        xs, ys = [], []
        for lam in sorted(by_lam):
            gaps = [
                float(r["obj_vs_full_pct"])
                for r in by_lam[lam]
                if r["obj_vs_full_pct"] != ""
            ]
            if gaps:
                xs.append(lam)
                ys.append(min(gaps))
        if not xs:
            continue
        color, marker, label = METHOD_STYLE.get(method, ((0.4, 0.4, 0.4), "^", method))
        ax.plot(
            xs, ys, "-", color=color, linewidth=1.2, marker=marker, ms=6, label=label
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="full MIP")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="1% target")
    ax.set_xlabel("λ (w_balanced)", fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(
        "Best objective gap to full MIP [%]", fontname="Liberation Serif", fontsize=11
    )
    instance, num_phases = cfg
    ax.set_title(
        f"Method comparison  —  {instance}  P={num_phases}",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(fontsize=9)

    fname = os.path.join(
        res_dir, f"method_best_gap_vs_lambda_{instance}_P{num_phases}.{FORMAT}"
    )
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    csv_fname = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = read_rows(csv_fname)
    res_dir = os.path.dirname(csv_fname) or "."
    for cfg, cfg_rows in group_by_config(rows).items():
        methods = _methods(cfg_rows)
        if not methods:
            continue

        # Per-method growth curves: objective gap and makespan (α) gap vs %edges,
        # plus objective vs k for reference.
        for method in methods:
            by_lam = _by_lambda(cfg_rows, method=method)
            if not by_lam:
                continue
            _plot(
                cfg,
                by_lam,
                res_dir,
                xfield="subgraph_pct",
                xlabel="Subgraph size [% of full-graph edges]",
                xlog=True,
                yfield="obj_vs_full_pct",
                ylabel="Objective gap to full MIP [%]",
                fname_stem="convergence_obj_vs_pct",
                method=method,
            )
            _plot(
                cfg,
                by_lam,
                res_dir,
                xfield="subgraph_pct",
                xlabel="Subgraph size [% of full-graph edges]",
                xlog=True,
                yfield="vs_full_pct",
                ylabel="Makespan (α) gap to full MIP [%]",
                fname_stem="convergence_alpha_vs_pct",
                method=method,
            )
            # Makespan vs the perfect-balance floor (phase width = T_total/P):
            # 0 = ideal balance, the residual is the unavoidable misalignment.
            _plot(
                cfg,
                by_lam,
                res_dir,
                xfield="subgraph_pct",
                xlabel="Subgraph size [% of full-graph edges]",
                xlog=True,
                yfield="alpha_vs_width_pct",
                ylabel="Makespan (α) gap to phase width [%]",
                fname_stem="convergence_alpha_vs_phase_width",
                method=method,
                baseline_label="phase width (ideal)",
            )
            _plot(
                cfg,
                by_lam,
                res_dir,
                xfield="k",
                xlabel="k (shortest paths)",
                xlog=True,
                yfield="obj_vs_full_pct",
                ylabel="Objective gap to full MIP [%]",
                fname_stem="convergence_obj_vs_k",
                method=method,
            )

        # Head-to-head comparison (only meaningful with ≥2 methods, but harmless
        # with one).
        plot_best_gap_vs_lambda(cfg, cfg_rows, res_dir)
