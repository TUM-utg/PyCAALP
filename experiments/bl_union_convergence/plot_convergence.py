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


def _full_solve_s(k_rows):
    """This λ's full-MIP solve time (the timing reference), or None. Every k-row
    of the config carries it, so read it from the first one that has it."""
    for r in k_rows:
        if r.get("full_solve_s", "") != "":
            return float(r["full_solve_s"])
    return None


def plot_timing(cfg, by_lam, res_dir, method, xfield, xlabel, fname_stem):
    """Total subgraph time (build + solve) vs `xfield`, one line per λ, against
    each λ's full-MIP solve time as a dashed reference in the same colour. Shows
    how much cheaper the growing subgraph is than solving the full MIP outright."""
    fig, ax = plt.subplots()
    lambdas = sorted(by_lam)
    colors = _lambda_colors(lambdas)

    for i, lam in enumerate(lambdas):
        k_rows = by_lam[lam]
        xs, ys = _xy(k_rows, xfield, "elapsed_s")
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
        # This λ's full-MIP time: a horizontal dashed line in the same colour, so
        # each subgraph curve is read against its own reference.
        full_t = _full_solve_s(k_rows)
        if full_t is not None:
            ax.axhline(full_t, color=colors[lam], linestyle="--", linewidth=0.8, alpha=0.7)
        sp = _stop_point(k_rows, xfield, "elapsed_s")
        if sp:
            ax.plot(
                sp[0], sp[1], marker="*", ms=13, color=colors[lam], mec="black", mew=0.6
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(
        "Total time (build + solve) [s]", fontname="Liberation Serif", fontsize=11
    )
    instance, num_phases = cfg
    ax.set_title(
        f"{method.capitalize()} method timing",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(title="λ (★ = auto-stop, -- = full MIP)", fontsize=8, ncol=2)

    fname = os.path.join(
        res_dir, f"{fname_stem}_{method}_{instance}_P{num_phases}.{FORMAT}"
    )
    plt.savefig(fname, format=FORMAT, dpi=1200)
    print(f"Saved {fname}")
    plt.close(fig)


def plot_timing_columns(cfg, by_lam, res_dir, method, normalize=True):
    """One column per λ: each k's total subgraph time as a dot, the ★ marking the
    auto-stop k, against that λ's full-MIP solve time.

    `normalize=True` divides by the full-MIP time (so the ceiling is 1.0 for
    every λ — a fair cross-λ comparison despite the ~25× spread in absolute
    time). `normalize=False` keeps absolute seconds on a log axis, with each λ's
    full-MIP time drawn as its own coloured cap (the caps then vary in height)."""
    fig, ax = plt.subplots()
    lambdas = sorted(by_lam)
    colors = _lambda_colors(lambdas)

    for lam in lambdas:
        k_rows = by_lam[lam]
        full_t = _full_solve_s(k_rows)
        if not full_t:
            continue
        xs, ys = _xy(k_rows, "k", "elapsed_s")
        if not xs:
            continue
        vals = [y / full_t for y in ys] if normalize else ys
        if normalize:
            # Stem from 0 up to the tallest point (log-safe only in absolute mode).
            ax.vlines(lam, 0, max(vals), color=colors[lam], linewidth=1.0, alpha=0.6)
        else:
            # Stem from the fastest point up to this λ's full-MIP cap.
            ax.vlines(lam, min(vals), full_t, color=colors[lam], linewidth=1.0, alpha=0.6)
            # This λ's full-MIP time: a short horizontal cap in the same colour
            # (label the first one only, so the legend gets a single entry).
            ax.plot(
                lam, full_t, marker="_", ms=14, mew=1.6, color=colors[lam],
                label="full MIP" if lam == lambdas[0] else None,
            )
        ax.plot(
            [lam] * len(vals), vals, "o", color=colors[lam], ms=5, mec="none", alpha=0.9
        )
        sp = _stop_point(k_rows, "k", "elapsed_s")
        if sp:
            ax.plot(
                lam, sp[1] / full_t if normalize else sp[1], marker="*", ms=13,
                color=colors[lam], mec="black", mew=0.6,
            )

    if normalize:
        # y = 1.0 is the full-MIP cost for every λ (the shared reference ceiling).
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="full MIP")
        ylabel = "Total time / full-MIP time"
        legend_title = "★ = auto-stop k"
        fname_stem = "convergence_timing_columns"
    else:
        ax.set_yscale("log")
        ylabel = "Total time (build + solve) [s]"
        legend_title = "★ = auto-stop, — = full MIP"
        fname_stem = "convergence_timing_columns_abs"
    ax.set_xlabel("λ (w_balanced)", fontname="Liberation Serif", fontsize=11)
    ax.set_ylabel(ylabel, fontname="Liberation Serif", fontsize=11)
    instance, num_phases = cfg
    ax.set_title(
        f"{method.capitalize()} method timing vs full MIP",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(title=legend_title, fontsize=8)

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
            # Total time (build + solve) against each λ's full-MIP time, vs both
            # k and the graph-agnostic subgraph edge fraction.
            plot_timing(
                cfg, by_lam, res_dir, method,
                xfield="k",
                xlabel="k (shortest paths)",
                fname_stem="convergence_timing_vs_k",
            )
            plot_timing(
                cfg, by_lam, res_dir, method,
                xfield="subgraph_pct",
                xlabel="Subgraph size [% of full-graph edges]",
                fname_stem="convergence_timing_vs_pct",
            )
            # Per-λ columns of subgraph time vs that λ's full-MIP time, both
            # normalised (fair cross-λ) and absolute seconds (log axis).
            plot_timing_columns(cfg, by_lam, res_dir, method, normalize=True)
            plot_timing_columns(cfg, by_lam, res_dir, method, normalize=False)

        # Head-to-head comparison (only meaningful with ≥2 methods, but harmless
        # with one).
        plot_best_gap_vs_lambda(cfg, cfg_rows, res_dir)
