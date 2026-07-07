"""Plot the two oracle-free convergence curves for adaptive_convergence.

Standalone (reads only the CSV, never re-solves). For each (instance, P) config
it emits, one line per λ vs subgraph size [% edges]:

  * convergence_obj_vs_ideal_*   — obj gap to the ideal (Σ shortest-path bound) [%]
  * convergence_alpha_vs_width_* — makespan (α) gap to phase width (T_total/P) [%]

Both references are analytical, so 0 is the best attainable and every curve sits
at or above it — no full MIP required.

Run (from the project root)::

    python -m experiments.adaptive_convergence.plot_convergence \
        experiments/adaptive_convergence/adaptive_convergence.csv
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FORMAT = "svg"
FONT = "Liberation Serif"
MARKERS = ["o", "s", "v", "^", "p", "D"]
MFCS_RGB = [(153, 153, 153), (0, 101, 189), (159, 186, 54)]
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "tum", [tuple(c / 255 for c in rgb) for rgb in MFCS_RGB]
)

# (csv y-field, axis label, output stem, dashed-baseline label)
PLOTS = [
    (
        "path_cost_vs_cmin_pct",
        "Path cost gap to shortest path [%]",
        "convergence_path_vs_cmin",
        "Sequence planning solution",
    ),
    (
        "alpha_vs_width_pct",
        "α gap to phase width [%]",
        "convergence_alpha_vs_width",
        "Line planning solution",
    ),
]


def _read(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _lambda_colors(lambdas):
    lo, hi = min(lambdas), max(lambdas)
    norm = mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)
    return {lam: _CMAP(norm(lam)) for lam in lambdas}


def _stop_xy(rows):
    for r in rows:
        if r.get("stop_reason") and r.get("subgraph_pct"):
            return r
    return None


def _draw_metric(ax, by_lam, lambdas, colors, yfield, ylabel, base_label, title=None):
    """Draw one metric's per-λ growth curves onto ``ax``. Returns the λ line
    handles (for a shared legend); does not add a legend itself."""
    handles = []
    for i, lam in enumerate(lambdas):
        lam_rows = sorted(by_lam[lam], key=lambda r: float(r["subgraph_pct"] or 0))
        xs, ys = [], []
        for r in lam_rows:
            if r.get("subgraph_pct") and r.get(yfield, "") != "":
                xs.append(float(r["subgraph_pct"]))
                ys.append(float(r[yfield]))
        if not xs:
            continue
        col = colors[float(lam)]
        (line,) = ax.plot(
            xs,
            ys,
            "-",
            color=col,
            linewidth=1.0,
            marker=MARKERS[i % len(MARKERS)],
            ms=5,
            mfc=col,
            label=f"λ={float(lam):g}",
        )
        handles.append(line)
        sp = _stop_xy(lam_rows)
        if sp and sp.get(yfield, "") != "":
            ax.plot(
                float(sp["subgraph_pct"]),
                float(sp[yfield]),
                marker="*",
                ms=13,
                color=col,
                mec="black",
                mew=0.6,
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label=base_label)
    ax.set_xscale("log")
    ax.set_xlabel("Subgraph size [% of full-graph edges]", fontname=FONT, fontsize=11)
    ax.set_ylabel(ylabel, fontname=FONT, fontsize=11)
    if title:
        ax.set_title(title, fontname=FONT, fontsize=12)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(FONT)
    return handles


def _configs(rows):
    """Group rows by (instance, P) then by λ."""
    configs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        configs[(r["instance"], r["num_phases"])][r["w_balanced"]].append(r)
    return configs


def _final_row(lam_rows):
    """The final-step solution for one λ: the stop row if present, else the row
    at the largest subgraph size (the last k reached)."""
    stops = [
        r
        for r in lam_rows
        if r.get("stop_reason") and r["stop_reason"] not in ("", "full_ref")
    ]
    if stops:
        return stops[0]
    valid = [r for r in lam_rows if r.get("subgraph_pct")]
    return max(valid, key=lambda r: float(r["subgraph_pct"])) if valid else None


def _stop_pct(by_lam):
    """The subgraph-% stop threshold for this config, for figure titles: the
    configured value from the CSV column if present, else the actual % at which
    a run auto-stopped. None if neither is available."""
    for lam_rows in by_lam.values():
        for r in lam_rows:
            v = r.get("stop_perc_graph", "")
            if v not in ("", None):
                return float(v)
    reached = [
        float(fr["subgraph_pct"])
        for fr in (_final_row(rows) for rows in by_lam.values())
        if fr and fr.get("stop_reason") == "stop_perc_graph" and fr.get("subgraph_pct")
    ]
    return max(reached) if reached else None


def plot(csv_path, out_dir=None):
    rows = _read(csv_path)
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    for (instance, num_phases), by_lam in _configs(rows).items():
        lambdas = sorted(by_lam, key=float)
        colors = _lambda_colors([float(x) for x in lambdas])
        for yfield, ylabel, stem, base_label in PLOTS:
            fig, ax = plt.subplots()
            _draw_metric(
                ax,
                by_lam,
                lambdas,
                colors,
                yfield,
                ylabel,
                base_label,
                title="Adaptive method convergence",
            )
            ax.legend(title="λ (★ = auto-stop)", fontsize=8, ncol=2)
            fname = os.path.join(out_dir, f"{stem}_{instance}_P{num_phases}.{FORMAT}")
            fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fname}")


def plot_fused(csv_path, out_dir=None):
    """One figure fusing every metric in PLOTS as side-by-side panels sharing a
    single λ legend on the right."""
    rows = _read(csv_path)
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    for (instance, num_phases), by_lam in _configs(rows).items():
        lambdas = sorted(by_lam, key=float)
        colors = _lambda_colors([float(x) for x in lambdas])
        n = len(PLOTS)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
        axes = axes if n > 1 else [axes]
        handles = []
        for ax, (yfield, ylabel, _stem, base_label) in zip(axes, PLOTS):
            # 0 = the analytical reference; ylabel already names it, so the panel
            # title just carries the baseline for a reader scanning the figure.
            h = _draw_metric(
                ax,
                by_lam,
                lambdas,
                colors,
                yfield,
                ylabel,
                base_label,
                title=f"{base_label}",
            )
            handles = h or handles  # keep the fullest set for the shared legend
        ref = Line2D([], [], color="black", linestyle="--", linewidth=1.0)
        fig.suptitle(
            "Adaptive method convergence",
            fontname=FONT,
            fontsize=13,
        )
        # Single shared legend on the right side of the whole figure.
        fig.legend(
            handles + [ref],
            [h.get_label() for h in handles] + ["reference (0)"],
            title="λ (★ = auto-stop)",
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fname = os.path.join(
            out_dir, f"convergence_fused_{instance}_P{num_phases}.{FORMAT}"
        )
        fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


# ASP / PLP bar colours (TUM blue / green).
ASP_COLOR = (0 / 255, 101 / 255, 189 / 255)
PLP_COLOR = (159 / 255, 186 / 255, 54 / 255)
PLP_COLOR = (153 / 255, 153 / 255, 153 / 255)


def plot_solutions(csv_path, out_dir=None):
    """Final-step solution comparison across λ (not convergence): a grouped bar
    chart, two bars per λ — ASP (assembly sequence planning: path-cost deviation
    from the cheapest sequence) and PLP (production line planning: makespan
    deviation from phase width). Reads left-to-right: as λ rises the ASP bar
    grows and the PLP bar shrinks — the trade-off the λ knob controls."""
    rows = _read(csv_path)
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    asp_f, plp_f = "path_cost_vs_cmin_pct", "alpha_vs_width_pct"
    for (instance, num_phases), by_lam in _configs(rows).items():
        stop_pct = _stop_pct(by_lam)
        stop_txt = f" (stop at {stop_pct:g}%)" if stop_pct is not None else ""
        lambdas = sorted(by_lam, key=float)
        labs, asp, plp = [], [], []
        for lam in lambdas:
            fr = _final_row(by_lam[lam])
            if fr is None or fr.get(asp_f, "") == "" or fr.get(plp_f, "") == "":
                continue
            labs.append(float(lam))
            asp.append(float(fr[asp_f]))
            plp.append(float(fr[plp_f]))
        if not labs:
            continue

        fig, ax = plt.subplots()
        x = list(range(len(labs)))
        w = 0.4
        b1 = ax.bar(
            [i - w / 2 for i in x],
            asp,
            w,
            color=ASP_COLOR,
            edgecolor="black",
            linewidth=0.4,
            label="ASP",
        )
        b2 = ax.bar(
            [i + w / 2 for i in x],
            plp,
            w,
            color=PLP_COLOR,
            edgecolor="black",
            linewidth=0.4,
            label="PLP",
        )
        ax.bar_label(b1, fmt="%.1f", fontsize=6, padding=1)
        ax.bar_label(b2, fmt="%.1f", fontsize=6, padding=1)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in labs])
        ax.set_xlabel("Time balancing weight (λ) ", fontname=FONT, fontsize=11)
        ax.set_ylabel("Deviation [%]", fontname=FONT, fontsize=11)
        ax.set_title(
            f"{instance.capitalize().replace("_"," ")} - ASP vs PLP deviation across λ - P={num_phases} {stop_txt}",
            fontname=FONT,
            fontsize=12,
        )
        ax.grid(True, axis="y", linewidth=0.3, color="gray", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(FONT)
        fname = os.path.join(
            out_dir, f"solutions_asp_plp_{instance}_P{num_phases}.{FORMAT}"
        )
        fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="adaptive_convergence CSV")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    plot(args.csv, args.out_dir)
    plot_fused(args.csv, args.out_dir)
    plot_solutions(args.csv, args.out_dir)
