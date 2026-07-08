"""Variable-λ plot merging adaptive_convergence results across P (num_phases).

Each ``<instance>_np_<P>/adaptive_convergence.csv`` is one phase count; the rows
self-identify via ``num_phases``, so this globs the per-P folders, concatenates
them (also writing ``<instance>_merged.csv`` as a convenience), and draws the
final-solution deviation vs λ with **one line per P**:

  * left panel  — ASP (assembly sequence planning): path-cost deviation [%]
  * right panel — PLP (production line planning): makespan deviation [%]

So you can read how the λ trade-off bends with the number of phases.

Run (from the project root)::

    python -m experiments.adaptive_convergence.plot_lambda_vs_phases \
        --dir experiments/adaptive_convergence --instance assembly_2
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.adaptive_convergence.plot_convergence import (  # noqa: E402
    _read,
    _final_row,
    FONT,
    FORMAT,
)

# (csv field, y-axis label, short panel title)
PANELS = [
    ("path_cost_vs_cmin_pct", "ASP deviation [%]", "ASP (0 = sum of shortest path)"),
    ("alpha_vs_width_pct", "PLP deviation [%]", "PLP (0 = phase width)"),
]

# Discrete colours for the P series (TUM blue, green, orange, purple, grey).
P_RGB = [
    (0, 101, 189),
    (159, 186, 54),
    (227, 114, 34),
    (101, 55, 142),
    (153, 153, 153),
]
MARKERS = ["o", "s", "v", "^", "p", "D"]


def _p_color(i):
    return tuple(c / 255 for c in P_RGB[i % len(P_RGB)])


def _write_merged(rows, out_csv):
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows -> {out_csv}")


def _stop_str(rows):
    """Display string for the subgraph-% stop threshold across the merged rows.
    Usually one value (e.g. '3%'); joins with '/' if the P runs differ."""
    vals = sorted(
        {
            float(r["stop_perc_graph"])
            for r in rows
            if r.get("stop_perc_graph", "") not in ("", None)
        }
    )
    return "/".join(f"{v:g}" for v in vals) + "%" if vals else ""


def plot_lambda_vs_phases(csv_paths, out_dir, instance_filter=None):
    rows = []
    for p in csv_paths:
        rows.extend(_read(p))
    if instance_filter:
        rows = [r for r in rows if r["instance"] == instance_filter]
    if not rows:
        raise SystemExit("no rows to plot (check --dir/--instance)")

    os.makedirs(out_dir, exist_ok=True)

    # instance -> P -> λ -> rows
    by_inst = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        by_inst[r["instance"]][int(r["num_phases"])][r["w_balanced"]].append(r)

    for instance, by_p in by_inst.items():
        inst_rows = [r for r in rows if r["instance"] == instance]
        _write_merged(inst_rows, os.path.join(out_dir, f"{instance}_merged.csv"))
        phases = sorted(by_p)
        fig, axes = plt.subplots(1, len(PANELS), figsize=(5.5 * len(PANELS), 4.5))
        axes = axes if len(PANELS) > 1 else [axes]
        handles = []
        for ax, (field, ylabel, short) in zip(axes, PANELS):
            for i, P in enumerate(phases):
                by_lam = by_p[P]
                lams = sorted(by_lam, key=float)
                xs, ys = [], []
                for lam in lams:
                    fr = _final_row(by_lam[lam])
                    if fr and fr.get(field, "") != "":
                        xs.append(float(lam))
                        ys.append(float(fr[field]))
                if not xs:
                    continue
                col = _p_color(i)
                (line,) = ax.plot(
                    xs,
                    ys,
                    "-",
                    color=col,
                    linewidth=1.2,
                    marker=MARKERS[i % len(MARKERS)],
                    ms=5,
                    mfc=col,
                    label=f"P={P}",
                )
                if ax is axes[0]:
                    handles.append(line)
            ax.set_xlabel("Time balancing weight (λ)", fontname=FONT, fontsize=11)
            ax.set_ylabel(ylabel, fontname=FONT, fontsize=11)
            ax.set_title(short, fontname=FONT, fontsize=12)
            ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontname(FONT)
        title = (
            f"{instance.capitalize().replace('_', ' ')} - "
            "ASP / PLP deviation vs λ across phases"
        )
        stop = _stop_str(inst_rows)
        if stop:
            title += f" (stop at {stop})"
        fig.suptitle(title, fontname=FONT, fontsize=13)
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            title="phases",
            fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fname = os.path.join(out_dir, f"lambda_vs_phases_{instance}.{FORMAT}")
        fig.savefig(fname, format=FORMAT, dpi=1200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="experiments/adaptive_convergence")
    parser.add_argument("--instance", default="assembly_2")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--csv",
        nargs="*",
        help="explicit CSV paths (overrides the --dir/--instance glob)",
    )
    args = parser.parse_args()
    if args.csv:
        csvs = args.csv
    else:
        csvs = sorted(
            glob.glob(
                os.path.join(
                    args.dir, f"{args.instance}_np_*", "adaptive_convergence.csv"
                )
            )
        )
    if not csvs:
        raise SystemExit(f"no CSVs found under {args.dir}/{args.instance}_np_*")
    plot_lambda_vs_phases(csvs, args.out_dir or args.dir, instance_filter=args.instance)
