"""Plot k-sensitivity as small-multiples across the λ (w_balanced) sweep.

Standalone: reads only the combined CSV produced by concatenating the per-λ
runs of test_strategy_comparison.py (see run_sweep.sh), so it never re-solves.
For every (instance, num_phases) it emits two grid figures, one panel per λ:

    All x-axes are the subgraph size as % of full-graph edges (subgraph_pct),
    NOT the raw path count k — k is not comparable across assemblies (the same k
    is a different fraction of each digraph), whereas % of edges is.

    lambda_sweep_quality_<instance>_P<P>.svg
        Objective gap to the full MIP (%) vs subgraph size, one line per weight
        variant. Read the % of edges at which each curve flattens — the plateau —
        and how that onset shifts as λ changes.

    lambda_sweep_makespan_<instance>_P<P>.svg
        Max phase time (makespan, α) vs subgraph size, with the full-MIP α dashed
        per-panel reference. NOTE: at λ<1 the objective also rewards engineering
        weight fulfilment, so α alone is NOT what is minimised — a larger α at
        λ<1 can still be objective-optimal. Read this together with the quality
        figure, never on its own.

    lambda_k_heatmap_obj_<instance>_P<P>.svg
        λ × k heatmap of the objective gap to the full MIP (%), one panel per
        weight variant. Sequential scale (always ≥ 0); the plateau shows up as a
        band — scan each λ row for the first k column that has gone optimal.

    lambda_k_heatmap_makespan_<instance>_P<P>.svg
        Same layout for the max-phase-time (α) gap. Diverging scale centred on 0:
        at λ<1 the subgraph can dip below the full MIP's α (the full MIP minimises
        the objective, not α), so blue < full < red. Read with the objective one.

    lambda_min_k_<instance>_P<P>.svg
        Min subgraph size (% of edges) needed to get within a target gap, vs λ —
        one panel per target gap, one line per weight variant.

    k_vs_subgraph_<instance>_P<P>.svg
        Decoder: subgraph size (% of edges) vs paths enumerated (log x), one line
        per variant. combined is drawn at 2k (it unions up to k edge-weight + k
        balanced paths). Bridges the raw k a run uses to the % of graph covered.

Panels share x and y axes so the plateau/makespan can be compared across λ.

Run from the project root via:
    python -m experiments.strategy_comparison.plot_lambda_sweep [combined_csv]
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless / cluster safe

import matplotlib.pyplot as plt
import numpy as np

from experiments.strategy_comparison.plot_k_sensitivity import (
    WEIGHT_STYLE,
    read_rows,
    set_cols,
)

# Default to the combined CSV that run_sweep.sh concatenates the per-λ files into.
DEFAULT_CSV = "experiments/strategy_comparison/assembly_2_np_3/strategy_comparison.csv"
FORMAT = "svg"
NCOLS = 3  # panels per row in the small-multiples grid
TARGET_GAPS = (1.0, 5.0)  # %, the min-k curve thresholds
HEATMAP_ANNOTATE = False  # write the gap % into each heatmap cell
HEATMAP_VMAX: float | None = (
    5  # cap colour scale (%) so low-gap diffs show; None = auto
)
HEATMAP_CMAP = "jet"  # perceptually-uniform; dark = optimal → yellow = worst
HEATMAP_DIV_CMAP = "RdBu_r"  # diverging map for signed gaps (blue < full < red)
# The "% of full-graph edges" grid the heatmap columns are resampled onto is
# derived per-run from the measured subgraph_pct range (see _pct_grid), so it
# lands where the data is — assembly_1 reaches double-digit %, assembly_2 ~1%.


def group_by_instance(rows):
    """Group rows by (instance, num_phases); each group spans the whole λ sweep."""
    groups = {}
    for row in rows:
        groups.setdefault((row["instance"], row["num_phases"]), []).append(row)
    return groups


def _lambdas(rows):
    """Sorted unique λ (w_balanced) values present in the rows."""
    return sorted({float(r["w_balanced"]) for r in rows if r["w_balanced"] != ""})


def _rows_for_lambda(rows, lam):
    return [r for r in rows if r["w_balanced"] != "" and float(r["w_balanced"]) == lam]


def _full_alpha(rows):
    """Full-MIP makespan (α) for this λ subset — the reference line."""
    for row in rows:
        if row["strategy"] == "full_mip" and row["alpha_abs"]:
            return float(row["alpha_abs"])
    return None


def _xy(rows, weight_attr, xfield, yfield):
    """Sorted (xfield, yfield) subgraph_mip points for one weight variant.

    Generalises _series (which is hard-wired to x='k') so we can put the
    instance-comparable subgraph_pct on the x-axis instead of the raw k count.
    """
    pts = []
    for r in rows:
        if (
            r["strategy"] == "subgraph_mip"
            and r["weight_attr"] == weight_attr
            and r[xfield] not in ("", "None")
            and r[yfield] not in ("", "None")
        ):
            pts.append((float(r[xfield]), float(r[yfield])))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def _interp_to_grid(xs, ys, grid):
    """Interpolate ys(xs) onto grid; NaN outside the measured x-range (no
    extrapolation). xs must be sorted ascending."""
    out = np.full(len(grid), np.nan)
    if len(xs) >= 2:
        lo, hi = xs[0], xs[-1]
        for j, t in enumerate(grid):
            if lo <= t <= hi:
                out[j] = np.interp(t, xs, ys)
    return out


def _pct_range(rows):
    """Min/max subgraph_pct over all subgraph_mip rows (the measured x-range).

    The same k is a wildly different fraction of each assembly: on assembly_1 the
    k-paths reach double-digit % of edges, on assembly_2 they barely clear 1%.
    Deriving the axis/grid range from the data keeps both readable instead of
    squashing a 0.01–1.5% sweep against a hardcoded 0–100% axis.
    """
    vals = [
        float(r["subgraph_pct"])
        for r in rows
        if r["strategy"] == "subgraph_mip" and r["subgraph_pct"] not in ("", "None")
    ]
    return (min(vals), max(vals)) if vals else (0.01, 100.0)


def _pct_xlim(rows):
    """Padded (lo, hi) %-of-edges limits for a log axis spanning the measured
    subgraph_pct range."""
    lo, hi = _pct_range(rows)
    return (lo * 0.7, hi * 1.4)


def _pct_grid(rows, n=8):
    """Log-spaced %-of-edges grid spanning the measured subgraph_pct range, so the
    heatmap columns land where the data actually is. Replaces a fixed 5–80% grid
    that fell entirely outside assembly_2's ~0.01–1.5% coverage (all-NaN columns)."""
    lo, hi = _pct_range(rows)
    grid = []
    for g in np.logspace(np.log10(lo), np.log10(hi), n):
        val = float(f"{g:.2g}")  # 2 sig figs for readable tick labels
        if val not in grid:
            grid.append(val)
    return grid


def _k_vals(rows):
    """Sorted unique k over the subgraph_mip rows — the heatmap k columns / the
    x-axis grid for the by-k figures (every λ is run on the same k grid)."""
    return sorted(
        {
            float(r["k"])
            for r in rows
            if r["strategy"] == "subgraph_mip" and r["k"] not in ("", "None")
        }
    )


def _full_solve_s(rows):
    """Full-MIP solve time for this λ subset — the timing reference line."""
    for r in rows:
        if r["strategy"] == "full_mip" and r["solve_s"] not in ("", "None"):
            return float(r["solve_s"])
    return None


def _make_grid(n, sharey=True):
    """A (fig, flat-axes) grid sized for n panels.

    x is always shared (common k axis). y is shared for relative metrics that are
    comparable across λ (quality %); independent for absolute makespan so each
    panel's convergence to its own full-MIP α reference stays readable.
    """
    nrows = math.ceil(n / NCOLS)
    # constrained layout reserves room for titles, tick labels, sup-labels and
    # the outside legend, so per-panel titles never collide with the row above.
    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        sharex=True,
        sharey=sharey,
        figsize=(NCOLS * 3.6, nrows * 2.7),
        squeeze=False,
        layout="constrained",
    )
    flat = [ax for r in axes for ax in r]
    # Hide any unused panels in the last row.
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _panel(ax, rows, field, cols, xlim):
    """Draw one weight-variant line per series of (% of full-graph edges, field).

    x is subgraph_pct (instance-comparable), not raw k, on a log axis whose limits
    come from the data (xlim) — the coverage spans <0.01% to ~tens of %, so a
    linear 0–100 axis squashes everything against the left edge. Each variant sits
    at its own %edges, so e.g. edge_w covers far less of the graph than combined at
    the same k.
    """
    for weight_attr, label, ci, marker in WEIGHT_STYLE:
        x_vals, y_vals = _xy(rows, weight_attr, "subgraph_pct", field)
        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                "-",
                color=cols[ci],
                linewidth=1.0,
                marker=marker,
                ms=5,
                mfc=cols[ci],
                label=label,
            )
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.grid(True, which="both", linewidth=0.3, color="gray", alpha=0.4)


def _finish(fig, flat, n, suptitle, xlabel, ylabel, fname, sharey=True):
    """Single shared x/y labels, one legend, suptitle, save + close.

    Per-panel axis labels are dropped (they collided with the titles of the row
    below); one figure-level supxlabel/supylabel serves the whole grid. x tick
    labels are kept only on the bottom panel of each column; y tick labels only
    on the left column when y is shared.
    """
    for i, ax in enumerate(flat[:n]):
        col = i % NCOLS
        is_bottom = (i + NCOLS) >= n  # nothing below this panel
        if not is_bottom:
            ax.tick_params(labelbottom=False)
        if col != 0 and sharey:
            ax.tick_params(labelleft=False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.supxlabel(xlabel, fontname="Liberation Serif", fontsize=12)
    fig.supylabel(ylabel, fontname="Liberation Serif", fontsize=12)
    fig.suptitle(suptitle, fontname="Liberation Serif", fontsize=13)
    # Legend outside, centred on the right edge — never overlaps the panels.
    fig.legend(
        handles,
        labels,
        title="Path weighting",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_quality_grid(rows, res_dir, instance, num_phases, lambdas):
    """Objective gap to full MIP (%) vs k — one panel per λ."""
    cols = set_cols()
    xlim = _pct_xlim(rows)
    fig, flat = _make_grid(len(lambdas))
    for ax, lam in zip(flat, lambdas):
        sub = _rows_for_lambda(rows, lam)
        _panel(ax, sub, "obj_vs_full_pct", cols, xlim)
        ax.axhline(0.0, color=cols[0], linestyle="--", linewidth=1.0)
        ax.set_title(f"λ = {lam:g}", fontname="Liberation Serif", fontsize=11)
    fname = os.path.join(
        res_dir, f"lambda_sweep_quality_{instance}_P{num_phases}.{FORMAT}"
    )
    _finish(
        fig,
        flat,
        len(lambdas),
        f"Plateau of solution quality vs. subgraph size across λ  —  {instance}  P={num_phases}",
        "Subgraph size [% of full-graph edges]",
        "Objective gap to full MIP [%]",
        fname,
    )


def plot_makespan_grid(rows, res_dir, instance, num_phases, lambdas):
    """Makespan (α) vs k — one panel per λ, full-MIP α as dashed reference."""
    cols = set_cols()
    xlim = _pct_xlim(rows)
    fig, flat = _make_grid(len(lambdas), sharey=False)
    for ax, lam in zip(flat, lambdas):
        sub = _rows_for_lambda(rows, lam)
        _panel(ax, sub, "alpha_abs", cols, xlim)
        full_a = _full_alpha(sub)
        if full_a is not None:
            ax.axhline(full_a, color=cols[0], linestyle="--", linewidth=1.0)
        ax.set_title(f"λ = {lam:g}", fontname="Liberation Serif", fontsize=11)
    fname = os.path.join(
        res_dir, f"lambda_sweep_makespan_{instance}_P{num_phases}.{FORMAT}"
    )
    _finish(
        fig,
        flat,
        len(lambdas),
        f"Max phase time (α) vs. subgraph size across λ  —  {instance}  P={num_phases}"
        "   (α≠objective at λ<1; read with quality figure)",
        "Subgraph size [% of full-graph edges]",
        "Max phase time α [s]",
        fname,
        sharey=False,
    )


def _panel_k(ax, rows, field, cols, *, logy=False):
    """Like _panel but x is the raw k (log), not subgraph_pct. Used by the by-k
    line grids (quality / makespan / timing) where k is the natural sweep knob and
    the run is a single instance, so cross-assembly comparability is moot."""
    for weight_attr, label, ci, marker in WEIGHT_STYLE:
        x_vals, y_vals = _xy(rows, weight_attr, "k", field)
        if x_vals:
            ax.plot(
                x_vals,
                y_vals,
                "-",
                color=cols[ci],
                linewidth=1.0,
                marker=marker,
                ms=5,
                mfc=cols[ci],
                label=label,
            )
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.3, color="gray", alpha=0.4)


def plot_grid_by_k(
    rows,
    res_dir,
    instance,
    num_phases,
    lambdas,
    *,
    field,
    fname_key,
    suptitle,
    ylabel,
    sharey=True,
    logy=False,
    hline=None,
    full_ref=None,
):
    """Small-multiples line grid of `field` vs k, one panel per λ.

    hline draws a constant per-panel reference (e.g. 0 for the gap); full_ref(sub)
    draws a per-λ reference computed from that λ's full-MIP row (e.g. its α or its
    solve time). Both, either, or neither may be given.
    """
    cols = set_cols()
    fig, flat = _make_grid(len(lambdas), sharey=sharey)
    for ax, lam in zip(flat, lambdas):
        sub = _rows_for_lambda(rows, lam)
        _panel_k(ax, sub, field, cols, logy=logy)
        if hline is not None:
            ax.axhline(hline, color=cols[0], linestyle="--", linewidth=1.0)
        if full_ref is not None:
            ref = full_ref(sub)
            if ref is not None:
                ax.axhline(ref, color=cols[0], linestyle="--", linewidth=1.0)
        ax.set_title(f"λ = {lam:g}", fontname="Liberation Serif", fontsize=11)
    fname = os.path.join(res_dir, f"{fname_key}_{instance}_P{num_phases}.{FORMAT}")
    _finish(
        fig,
        flat,
        len(lambdas),
        f"{suptitle}  —  {instance}  P={num_phases}",
        "k (shortest paths)",
        ylabel,
        fname,
        sharey=sharey,
    )


def _heatmap(
    rows,
    res_dir,
    instance,
    num_phases,
    lambdas,
    *,
    field,
    clabel,
    suptitle,
    key,
    diverging,
    xmode="pct",
    vmax_cap=HEATMAP_VMAX,
):
    """λ × (subgraph size | k) heatmap of `field`, one panel per weight variant.

    xmode="pct": columns are % of full-graph edges, resampled onto a log grid so
    the same column is the same coverage on any assembly. xmode="k": columns are
    the raw k values actually run — exact, no interpolation (every λ shares the
    same k grid), readable within a single instance.

    Sequential (diverging=False): scale runs 0 → cap, for always-positive gaps
    like the objective gap. Diverging (diverging=True): scale is symmetric about
    0 so signed gaps (e.g. makespan, which can dip below the full MIP at λ<1)
    read as blue < full < red. vmax_cap caps the colour scale (None = auto-fit,
    needed when `field` is a time in seconds rather than a small %).
    Cell numbers are off by default (HEATMAP_ANNOTATE).
    """
    cmap = plt.get_cmap(HEATMAP_DIV_CMAP if diverging else HEATMAP_CMAP).copy()
    cmap.set_bad("0.85")  # absent (λ, k) cells in light grey

    variants = [(w, lbl) for (w, lbl, _ci, _m) in WEIGHT_STYLE]
    if xmode == "k":
        grid = _k_vals(rows)
        xlabel = "k (shortest paths)"

        def _fill(sub, w):
            d = dict(zip(*_xy(sub, w, "k", field)))
            return np.array([d.get(g, np.nan) for g in grid])

    else:
        grid = _pct_grid(rows)  # % of full-graph edges, comparable across runs
        xlabel = "Subgraph size [% of full-graph edges]"

        def _fill(sub, w):
            xs, ys = _xy(sub, w, "subgraph_pct", field)
            return _interp_to_grid(xs, ys, grid)

    mats = []
    for w, _ in variants:
        mat = np.full((len(lambdas), len(grid)), np.nan)
        for i, lam in enumerate(lambdas):
            mat[i] = _fill(_rows_for_lambda(rows, lam), w)
        mats.append(np.ma.masked_invalid(mat))
    finite = [m for m in mats if m.count()]
    data_max = max((m.max() for m in finite), default=1.0)
    data_min = min((m.min() for m in finite), default=0.0)

    if diverging:
        lim = vmax_cap if vmax_cap is not None else max(abs(data_min), data_max)
        vmin, vmax = -lim, lim
        extend = (
            "both"
            if data_min < vmin and data_max > vmax
            else "min" if data_min < vmin else "max" if data_max > vmax else "neither"
        )
        dark = lim  # text turns white near either saturated end
    else:
        vmin = 0.0
        vmax = vmax_cap if vmax_cap is not None else data_max
        extend = "max" if vmax < data_max else "neither"
        dark = vmax

    # Dedicated gridspec column for the colorbar so it can never overlap the
    # rightmost k cells (sharing/adjusting the panel axes is what caused that).
    nrows = len(variants)
    fig = plt.figure(figsize=(max(6.0, 0.7 * len(grid)), 2.0 * nrows))
    gs = fig.add_gridspec(nrows, 2, width_ratios=[1.0, 0.025], wspace=0.06, hspace=0.35)
    panel_axes = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]
    cax = fig.add_subplot(gs[:, 1])
    im = None
    last_ax = panel_axes[-1]
    for ax, (w, label), mat in zip(panel_axes, variants, mats):
        im = ax.imshow(
            mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
        )
        # Crisp white separators between cells (seaborn-style).
        ax.set_xticks(np.arange(-0.5, len(grid), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(lambdas), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([f"{p:g}" for p in grid], fontsize=8)
        if ax is not last_ax:
            ax.tick_params(labelbottom=False)  # %edges labels only on bottom panel
        ax.set_yticks(range(len(lambdas)))
        ax.set_yticklabels([f"{lam:g}" for lam in lambdas], fontsize=5)
        ax.set_ylabel("λ", fontname="Liberation Serif", fontsize=11)
        ax.set_title(label, fontname="Liberation Serif", fontsize=11, loc="left")

        if HEATMAP_ANNOTATE:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat.mask[i, j]:
                        continue
                    v = mat[i, j]
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.6 * dark else "black",
                    )
    last_ax.set_xlabel(
        xlabel,
        fontname="Liberation Serif",
        fontsize=11,
    )
    cbar = fig.colorbar(im, cax=cax, extend=extend)
    cbar.set_label(clabel, fontname="Liberation Serif", fontsize=11)
    cbar.outline.set_visible(False)
    fig.suptitle(
        f"{suptitle}  —  {instance}  P={num_phases}",
        fontname="Liberation Serif",
        fontsize=13,
    )
    fname = os.path.join(
        res_dir, f"lambda_k_heatmap_{key}_{instance}_P{num_phases}.{FORMAT}"
    )
    fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_heatmap(rows, res_dir, instance, num_phases, lambdas):
    """λ × k heatmap of the objective gap to the full MIP (always ≥ 0)."""
    _heatmap(
        rows,
        res_dir,
        instance,
        num_phases,
        lambdas,
        field="obj_vs_full_pct",
        clabel="Objective gap to full MIP [%]",
        suptitle="Objective deviation over λ × subgraph size",
        key="obj",
        diverging=False,
    )


def plot_makespan_heatmap(rows, res_dir, instance, num_phases, lambdas):
    """λ × k heatmap of the makespan (α) gap to the full MIP. Signed: at λ<1 the
    subgraph can beat the full MIP's α (negative) because the full MIP minimises
    the objective, not α — read alongside the objective heatmap."""
    _heatmap(
        rows,
        res_dir,
        instance,
        num_phases,
        lambdas,
        field="vs_full_pct",
        clabel="Max phase time gap to full MIP [%]",
        suptitle="Max phase time (α) deviation over λ × subgraph size",
        key="makespan",
        diverging=True,
    )


def _min_pct(sub, weight_attr, thr):
    """Smallest subgraph size (% of full-graph edges) at/after which the objective
    gap to the full MIP *stays* ≤ thr (%) — i.e. one step past the last subgraph
    that still violates the target.

    This is the conservative budget: the first dip under the threshold can be a
    lucky small-k subgraph that a slightly larger one falls back above (the gap
    curve is not guaranteed monotone), so reporting that first crossing
    understates what you actually need. The last crossing is what guarantees the
    target. None if the target is never reached within the tested range."""
    xs, ys = _xy(sub, weight_attr, "subgraph_pct", "obj_vs_full_pct")
    if not xs:
        return None
    last_above = None  # largest subgraph size that still exceeds the target
    for pct, v in zip(xs, ys):
        if v > thr:
            last_above = pct
    if last_above is None:
        return xs[0]  # already within target at the smallest subgraph
    for pct in xs:  # first size strictly past the last violation
        if pct > last_above:
            return pct
    return None  # largest subgraph still violates → target not robustly reached


def plot_min_k(rows, res_dir, instance, num_phases, lambdas):
    """Subgraph size (% of full-graph edges) needed to *stay* within a target
    objective gap, as a function of λ — one panel per target gap, one line per
    weight variant. The actionable distillation of the heatmap: 'what fraction of
    the graph does this weighting need at this λ'. A missing point means the
    weighting never robustly reached the target within the tested k range (see
    _min_pct)."""
    cols = set_cols()
    fig, axes = plt.subplots(
        1,
        len(TARGET_GAPS),
        sharey=True,
        figsize=(4.5 * len(TARGET_GAPS), 3.6),
        squeeze=False,
    )
    for ax, thr in zip(axes[0], TARGET_GAPS):
        for weight_attr, label, ci, marker in WEIGHT_STYLE:
            xs, ys = [], []
            for lam in lambdas:
                mp = _min_pct(_rows_for_lambda(rows, lam), weight_attr, thr)
                if mp is not None:
                    xs.append(lam)
                    ys.append(mp)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    "-",
                    color=cols[ci],
                    linewidth=1.0,
                    marker=marker,
                    ms=5,
                    mfc=cols[ci],
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_ylim(*_pct_xlim(rows))
        ax.set_xlabel("λ (w_balanced)", fontname="Liberation Serif", fontsize=11)
        ax.set_title(f"gap ≤ {thr:g}%", fontname="Liberation Serif", fontsize=11)
        ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    axes[0, 0].set_ylabel(
        "subgraph size to stay within target [% of edges]",
        fontname="Liberation Serif",
        fontsize=11,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"{instance.capitalize().replace("_"," ")} - Graph fraction needed to stay within "
        f"a target gap across λ (P={num_phases})",
        fontname="Liberation Serif",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(
        handles,
        labels,
        title="Path weighting",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    fname = os.path.join(res_dir, f"lambda_min_k_{instance}_P{num_phases}.{FORMAT}")
    fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_k_vs_subgraph(rows, res_dir, instance, num_phases):
    """Decoder figure linking enumeration effort to subgraph size.

    x = number of paths enumerated (log), y = subgraph size as % of full-graph
    edges. One line per weight variant; subgraph_pct is λ-invariant so each is a
    single curve (averaged over λ to be robust to any tiny variation). The
    combined variant unions up to k edge-weight + k balanced paths, so it is
    drawn at 2k — the x-axis is paths actually enumerated, making the cost of
    'combined' explicit. This is the bridge between the raw k a run uses and the
    % of the graph it covers (the same k is a different % on each assembly).
    """
    cols = set_cols()
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    for weight_attr, label, ci, marker in WEIGHT_STYLE:
        acc = {}
        for r in rows:
            if (
                r["strategy"] == "subgraph_mip"
                and r["weight_attr"] == weight_attr
                and r["k"]
                and r["subgraph_pct"] not in ("", "None")
            ):
                acc.setdefault(float(r["k"]), []).append(float(r["subgraph_pct"]))
        if not acc:
            continue
        ks = sorted(acc)
        ys = [sum(acc[k]) / len(acc[k]) for k in ks]
        mult = 2 if weight_attr == "combined" else 1
        xs = [k * mult for k in ks]
        num_ks_str = "(k)"
        if weight_attr == "combined":
            num_ks_str = "(2k)"
        elif weight_attr == "blended_union":
            num_ks_str = "(3k)"
        ax.plot(
            xs,
            ys,
            "-",
            color=cols[ci],
            linewidth=1.0,
            marker=marker,
            ms=5,
            mfc=cols[ci],
            label=f"{label} " + num_ks_str,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*_pct_xlim(rows))
    ax.set_xlabel(
        "Paths enumerated",
        fontname="Liberation Serif",
        fontsize=11,
    )
    ax.set_ylabel(
        "Subgraph size [% of full-graph edges]",
        fontname="Liberation Serif",
        fontsize=11,
    )
    ax.set_title(
        f"{instance.capitalize().replace("_"," ")} - Graph coverage vs. enumeration effort  (P={num_phases})",
        fontname="Liberation Serif",
        fontsize=13,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    ax.legend(title="Path weighting")
    fname = os.path.join(res_dir, f"k_vs_subgraph_{instance}_P{num_phases}.{FORMAT}")
    fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    csv_fname = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    rows = read_rows(csv_fname)
    res_dir = os.path.dirname(csv_fname)
    for (instance, num_phases), grp in group_by_instance(rows).items():
        lambdas = _lambdas(grp)
        if len(lambdas) < 2:
            print(
                f"Skipping {instance} P{num_phases}: only λ={lambdas} present "
                "(λ sweep plots need ≥2 λ values)."
            )
            continue
        # --- vs subgraph size (% of edges) -------------------------------
        plot_quality_grid(grp, res_dir, instance, num_phases, lambdas)
        plot_makespan_grid(grp, res_dir, instance, num_phases, lambdas)
        plot_heatmap(grp, res_dir, instance, num_phases, lambdas)
        plot_makespan_heatmap(grp, res_dir, instance, num_phases, lambdas)
        plot_min_k(grp, res_dir, instance, num_phases, lambdas)
        plot_k_vs_subgraph(grp, res_dir, instance, num_phases)

        # --- same families, but with raw k on the x-axis -----------------
        plot_grid_by_k(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="obj_vs_full_pct",
            fname_key="lambda_sweep_quality_byk",
            suptitle="Solution quality vs. k across λ",
            ylabel="Objective gap to full MIP [%]",
            hline=0.0,
        )
        plot_grid_by_k(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="alpha_abs",
            fname_key="lambda_sweep_makespan_byk",
            suptitle="Max phase time (α) vs. k across λ"
            "   (α≠objective at λ<1; read with quality figure)",
            ylabel="Max phase time α [s]",
            sharey=False,
            full_ref=_full_alpha,
        )
        _heatmap(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="obj_vs_full_pct",
            clabel="Objective gap to full MIP [%]",
            suptitle="Objective deviation over λ × k",
            key="obj_byk",
            diverging=False,
            xmode="k",
        )
        _heatmap(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="vs_full_pct",
            clabel="Max phase time gap to full MIP [%]",
            suptitle="Max phase time (α) deviation over λ × k",
            key="makespan_byk",
            diverging=True,
            xmode="k",
        )

        # --- timing: build + solve cost of the subgraph strategies -------
        plot_grid_by_k(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="elapsed_s",
            fname_key="lambda_sweep_timing_byk",
            suptitle="Runtime (build + solve) vs. k across λ"
            "   (dashed = full-MIP solve)",
            ylabel="Elapsed: build + solve [s]",
            logy=True,
            full_ref=_full_solve_s,
        )
        _heatmap(
            grp,
            res_dir,
            instance,
            num_phases,
            lambdas,
            field="elapsed_s",
            clabel="Elapsed: build + solve [s]",
            suptitle="Runtime over λ × k",
            key="time_byk",
            diverging=False,
            xmode="k",
            vmax_cap=None,  # seconds, not a %, so auto-fit the colour scale
        )
