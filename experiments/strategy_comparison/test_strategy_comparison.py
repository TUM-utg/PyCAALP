"""Experiment: compare MIP solving strategies on Assembly 1, with and without
time-balanced edge weights in the k-shortest-path enumeration.

Strategies
----------
1. Full MIP            — joint path-selection + phase-assignment on the complete digraph
2a. Path-Enum / edge_w  — k paths enumerated by engineering weight (edge_weight)
2b. Path-Enum / bal_w   — k paths enumerated by time_balanced_weight
3a. Subgraph / edge_w   — subgraph MIP on union of k paths (edge_weight)
3b. Subgraph / bal_w    — subgraph MIP on union of k paths (time_balanced_weight)

Results are written to strategy_comparison.csv.

Run from the project root via:
    python -m experiments.strategy_comparison.test_strategy_comparison
"""

import argparse
import csv
import json
import os
import time

from pycaalp.run import create_assembly_digraph, optimize
from pycaalp.time_balancing.subgraph_mip import (
    build_blended_union_subgraph,
    build_diverse_subgraph,
    build_kpath_subgraph,
    solve_by_subgraph_mip,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

FILE_NAME = "data/assembly_1/assembly_1_parts.json"
DFM_FILE_NAME = None

# Assembly 2
# FILE_NAME = "data/assembly_2/assembly_2_parts.json"
# DFM_FILE_NAME = "data/assembly_2/assembly_2_dfm.json"

# Instance identifier (e.g. "assembly_2") — recorded on every row so results
# from different assemblies/configs can be concatenated and told apart.
INSTANCE = os.path.basename(os.path.dirname(FILE_NAME))

NUM_PHASES = 3
W_BALANCED = 1.0

K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

# Blend values unioned by the blended_union strategy (frontier union). blend=0
# reproduces the edge_weight ranking, blend=1 is the continuous balance ranking,
# the middle catches compromise paths. λ-agnostic (the MIP still solves at λ).
BLEND_GRID = [0.0, 0.5, 1.0]

# Diverse strategy: penalised re-routing on the blended weight at the true λ
# (idea #4). PENALTY trades coverage for cost; ~0.5 already saturates to near
# edge-disjoint enumeration on these normalised weights.
PENALTY = 0.5

RESULTS_FILE = "experiments/strategy_comparison/strategy_comparison.csv"

CSV_FIELDS = [
    # Config identity (so rows from different runs can be concatenated)
    "instance",
    "num_phases",
    "w_balanced",
    "strategy",
    "weight_attr",
    "k",
    # Problem size
    "digraph_nodes",
    "digraph_edges",
    "subgraph_edges",
    "subgraph_pct",  # subgraph_edges / digraph_edges * 100
    # Timing breakdown (seconds)
    "digraph_build_s",  # one-off, shared by all strategies
    "build_s",  # k-path enumeration + subgraph build
    "solve_s",  # MIP solve
    "elapsed_s",  # build_s + solve_s
    "speedup_vs_full",  # full_mip solve_s / elapsed_s
    # Objective / balance quality
    "objective",
    "obj_vs_full_pct",
    "alpha_abs",  # max phase time (makespan)
    "vs_full_pct",  # alpha vs full
    "phase_time_min",
    "phase_time_max",
    "phase_time_std",
    "imbalance_pct",  # (max - min) / ideal * 100; P-agnostic
    "abs_time_per_phase",  # JSON list, any P
    "ops_per_phase",  # JSON list, any P
    "n_ops_total",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _record(
    ctx,
    strategy,
    weight_attr,
    k,
    build_s,
    solve_s,
    results,
    ops_list,
    subgraph_edges,
):
    """Build one CSV row.

    ctx carries the per-run constants (instance, num_phases, w_balanced, digraph
    size/build time, and the full-MIP reference alpha/obj/solve time).
    """
    num_phases = ctx["num_phases"]
    abs_times = results["absolute_time_per_phase"]
    # Phase times for ALL phases (0..P-1), so this works for any num_phases.
    phase_times = [abs_times.get(p, 0.0) for p in range(num_phases)]
    alpha_abs = max(phase_times) if phase_times else 0.0
    pt_min = min(phase_times) if phase_times else 0.0
    ideal = sum(phase_times) / num_phases if num_phases else 0.0
    std = (
        (sum((t - ideal) ** 2 for t in phase_times) / num_phases) ** 0.5
        if num_phases
        else 0.0
    )
    imbalance_pct = round((alpha_abs - pt_min) / ideal * 100, 2) if ideal else None

    full_alpha = ctx["full_alpha"]
    full_obj = ctx["full_obj"]
    vs_full = round((alpha_abs / full_alpha - 1) * 100, 2) if full_alpha else None
    objective = results.get("objective")
    obj_vs_full = (
        round((objective / full_obj - 1) * 100, 2)
        if objective is not None and full_obj
        else None
    )

    elapsed = build_s + solve_s
    speedup = round(ctx["full_solve_s"] / elapsed, 2) if elapsed else None
    digraph_edges = ctx["digraph_edges"]
    n_ops = [len(p) for p in ops_list]

    return {
        "instance": ctx["instance"],
        "num_phases": num_phases,
        "w_balanced": ctx["w_balanced"],
        "strategy": strategy,
        "weight_attr": weight_attr,
        "k": k if k is not None else "",
        "digraph_nodes": ctx["digraph_nodes"],
        "digraph_edges": digraph_edges,
        "subgraph_edges": subgraph_edges,
        "subgraph_pct": (
            round(subgraph_edges / digraph_edges * 100, 2) if digraph_edges else ""
        ),
        "digraph_build_s": round(ctx["digraph_build_s"], 3),
        "build_s": round(build_s, 3),
        "solve_s": round(solve_s, 3),
        "elapsed_s": round(elapsed, 3),
        "speedup_vs_full": speedup if speedup is not None else "",
        "objective": round(objective, 4) if objective is not None else "",
        "obj_vs_full_pct": obj_vs_full if obj_vs_full is not None else "",
        "alpha_abs": round(alpha_abs, 3),
        "vs_full_pct": vs_full if vs_full is not None else "",
        "phase_time_min": round(pt_min, 3),
        "phase_time_max": round(alpha_abs, 3),
        "phase_time_std": round(std, 3),
        "imbalance_pct": imbalance_pct if imbalance_pct is not None else "",
        "abs_time_per_phase": json.dumps([round(t, 3) for t in phase_times]),
        "ops_per_phase": json.dumps(n_ops),
        "n_ops_total": sum(n_ops),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # CLI overrides let one config (instance, P, λ) run per process so the sweep
    # can be fanned out one-per-core. Defaults reproduce the single-run behaviour.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w-balanced", type=float, default=W_BALANCED)
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    parser.add_argument(
        "--out",
        default=RESULTS_FILE,
        help="output CSV path (give each parallel task its own file)",
    )
    args = parser.parse_args()
    W_BALANCED = args.w_balanced
    NUM_PHASES = args.num_phases
    RESULTS_FILE = args.out

    print("=" * 72)
    print(
        f"Strategy Comparison Experiment — {INSTANCE}  P={NUM_PHASES}  λ={W_BALANCED}"
    )
    print("=" * 72)

    # Build digraph once (time_balanced_weight is set at construction time)
    print("\nBuilding assembly digraph …")
    ad, build_time = _timed(
        create_assembly_digraph,
        file_name=FILE_NAME,
        w_bal=W_BALANCED,
        num_phases=NUM_PHASES,
        dfm_file=DFM_FILE_NAME,
    )
    n_nodes = ad.assembly_digraph.number_of_nodes()
    n_edges = ad.assembly_digraph.number_of_edges()
    print(f"  {n_nodes} nodes, {n_edges} edges  ({build_time:.1f}s)")

    # Blended enumeration weight at the true λ: a single ranking that targets the
    # λ-specific compromise path (no union, no ratio to tune). Set once; reused
    # for every k below via weight_attr="blended_weight".
    ad.set_blended_weights(W_BALANCED)

    records = []

    # ------------------------------------------------------------------
    # Method 1: Full MIP (single run, no k)
    # ------------------------------------------------------------------
    print(f"\n[1] Full MIP  P={NUM_PHASES}  λ={W_BALANCED}")
    (results1, ops1), t1 = _timed(
        optimize,
        assembly_digraph=ad,
        num_phases=NUM_PHASES,
        w_balanced=W_BALANCED,
        hide_output=False,
        full_result_output=True,
    )
    full_alpha = max(results1["absolute_time_per_phase"].values())
    full_obj = results1["objective"]

    # Per-run constants shared by every recorded row.
    ctx = {
        "instance": INSTANCE,
        "num_phases": NUM_PHASES,
        "w_balanced": W_BALANCED,
        "digraph_nodes": n_nodes,
        "digraph_edges": n_edges,
        "digraph_build_s": build_time,
        "full_alpha": full_alpha,
        "full_obj": full_obj,
        "full_solve_s": t1,
    }

    r1 = _record(ctx, "full_mip", "—", None, 0.0, t1, results1, ops1, n_edges)
    records.append(r1)
    print(f"  obj={r1['objective']:.3f}  alpha={r1['alpha_abs']:.1f}s  time={t1:.2f}s")

    # ------------------------------------------------------------------
    # Methods 2 & 3: sweep over k, both weight variants
    # ------------------------------------------------------------------
    for k in K_VALUES:
        print(f"\nk = {k}")

        # Build each subgraph once (timed), then reuse it for solving below so
        # the enumeration/build cost is measured apart from the MIP solve.
        sg_ew, tb_ew = _timed(build_kpath_subgraph, ad, k, weight_attr="edge_weight")
        sg_bw, tb_bw = _timed(
            build_kpath_subgraph, ad, k, weight_attr="time_balanced_weight"
        )
        sg_cw, tb_cw = _timed(
            build_kpath_subgraph,
            ad,
            k,
            weight_attr=["edge_weight", "time_balanced_weight"],
        )
        sg_bl, tb_bl = _timed(build_kpath_subgraph, ad, k, weight_attr="blended_weight")
        sg_bu, tb_bu = _timed(build_blended_union_subgraph, ad, k, BLEND_GRID)
        sg_dv, tb_dv = _timed(build_diverse_subgraph, ad, k, W_BALANCED, PENALTY)
        sg_ew_edges = sg_ew.number_of_edges()
        sg_bw_edges = sg_bw.number_of_edges()
        sg_cw_edges = sg_cw.number_of_edges()
        sg_bl_edges = sg_bl.number_of_edges()
        sg_bu_edges = sg_bu.number_of_edges()
        sg_dv_edges = sg_dv.number_of_edges()
        print(
            f"  Subgraph edges — edge_w: {sg_ew_edges} ({sg_ew_edges/n_edges*100:.1f}%)  "
            f"bal_w: {sg_bw_edges} ({sg_bw_edges/n_edges*100:.1f}%)  "
            f"combined: {sg_cw_edges} ({sg_cw_edges/n_edges*100:.1f}%)  "
            f"blended: {sg_bl_edges} ({sg_bl_edges/n_edges*100:.1f}%)  "
            f"blended_union: {sg_bu_edges} ({sg_bu_edges/n_edges*100:.1f}%)  "
            f"diverse: {sg_dv_edges} ({sg_dv_edges/n_edges*100:.1f}%)"
        )

        # 3a: Subgraph MIP — edge_weight
        print(f"  [3a] Subgraph / edge_w   k={k} …")
        (results3a, ops3a), t3a = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_ew,
        )
        r3a = _record(
            ctx,
            "subgraph_mip",
            "edge_weight",
            k,
            tb_ew,
            t3a,
            results3a,
            ops3a,
            sg_ew_edges,
        )
        records.append(r3a)
        print(
            f"      obj={r3a['objective']:.3f} (vs_full={r3a['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3a['alpha_abs']:.1f}s  build={tb_ew:.3f}s solve={t3a:.3f}s  vs_full={r3a['vs_full_pct']:+.2f}%"
        )

        # 3b: Subgraph MIP — time_balanced_weight
        print(f"  [3b] Subgraph / bal_w    k={k} …")
        (results3b, ops3b), t3b = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_bw,
        )
        r3b = _record(
            ctx,
            "subgraph_mip",
            "time_balanced_weight",
            k,
            tb_bw,
            t3b,
            results3b,
            ops3b,
            sg_bw_edges,
        )
        records.append(r3b)
        print(
            f"      obj={r3b['objective']:.3f} (vs_full={r3b['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3b['alpha_abs']:.1f}s  build={tb_bw:.3f}s solve={t3b:.3f}s  vs_full={r3b['vs_full_pct']:+.2f}%"
        )

        # 3c: Subgraph MIP — combined (union of edge_w + bal_w, up to 2k paths)
        print(f"  [3c] Subgraph / combined k={k} …")
        (results3c, ops3c), t3c = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_cw,
        )
        r3c = _record(
            ctx,
            "subgraph_mip",
            "combined",
            k,
            tb_cw,
            t3c,
            results3c,
            ops3c,
            sg_cw_edges,
        )
        records.append(r3c)
        print(
            f"      obj={r3c['objective']:.3f} (vs_full={r3c['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3c['alpha_abs']:.1f}s  build={tb_cw:.3f}s solve={t3c:.3f}s  vs_full={r3c['vs_full_pct']:+.2f}%"
        )

        # 3d: Subgraph MIP — blended (single enumeration by w_blend at the true λ)
        print(f"  [3d] Subgraph / blended  k={k} …")
        (results3d, ops3d), t3d = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_bl,
        )
        r3d = _record(
            ctx,
            "subgraph_mip",
            "blended",
            k,
            tb_bl,
            t3d,
            results3d,
            ops3d,
            sg_bl_edges,
        )
        records.append(r3d)
        print(
            f"      obj={r3d['objective']:.3f} (vs_full={r3d['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3d['alpha_abs']:.1f}s  build={tb_bl:.3f}s solve={t3d:.3f}s  vs_full={r3d['vs_full_pct']:+.2f}%"
        )

        # 3e: Subgraph MIP — blended_union (frontier union of blended over BLEND_GRID)
        print(f"  [3e] Subgraph / bl-union k={k} …")
        (results3e, ops3e), t3e = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_bu,
        )
        r3e = _record(
            ctx,
            "subgraph_mip",
            "blended_union",
            k,
            tb_bu,
            t3e,
            results3e,
            ops3e,
            sg_bu_edges,
        )
        records.append(r3e)
        print(
            f"      obj={r3e['objective']:.3f} (vs_full={r3e['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3e['alpha_abs']:.1f}s  build={tb_bu:.3f}s solve={t3e:.3f}s  vs_full={r3e['vs_full_pct']:+.2f}%"
        )

        # 3f: Subgraph MIP — diverse (penalised re-routing on blended weight @ λ)
        print(f"  [3f] Subgraph / diverse  k={k} …")
        (results3f, ops3f), t3f = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg_dv,
        )
        r3f = _record(
            ctx,
            "subgraph_mip",
            "diverse",
            k,
            tb_dv,
            t3f,
            results3f,
            ops3f,
            sg_dv_edges,
        )
        records.append(r3f)
        print(
            f"      obj={r3f['objective']:.3f} (vs_full={r3f['obj_vs_full_pct']:+.2f}%)  "
            f"alpha={r3f['alpha_abs']:.1f}s  build={tb_dv:.3f}s solve={t3f:.3f}s  vs_full={r3f['vs_full_pct']:+.2f}%"
        )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("Plot with:  python -m experiments.strategy_comparison.plot_k_sensitivity")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 124)
    print(f"SUMMARY — {INSTANCE}  P={NUM_PHASES}  λ={W_BALANCED}")
    print("=" * 124)
    hdr = (
        f"{'Strategy':<14} {'Weight':<22} {'k':>6} "
        f"{'Build':>7} {'Solve':>8} {'Total':>8} {'Spdup':>6} "
        f"{'Obj':>9} {'ObjvsF':>8} {'Alpha':>9} {'AlvsF':>8} {'Edges':>7} {'%Grph':>6}"
    )
    print(hdr)
    print("-" * 124)
    for r in records:
        vs = f"{r['vs_full_pct']:+.2f}" if r["vs_full_pct"] != "" else "—"
        obj = f"{r['objective']:.3f}" if r["objective"] != "" else "—"
        obj_vs = f"{r['obj_vs_full_pct']:+.2f}" if r["obj_vs_full_pct"] != "" else "—"
        spd = f"{r['speedup_vs_full']:.1f}x" if r["speedup_vs_full"] != "" else "—"
        pct = f"{r['subgraph_pct']:.1f}" if r["subgraph_pct"] != "" else "—"
        print(
            f"{r['strategy']:<14} {r['weight_attr']:<22} {str(r['k']):>6} "
            f"{r['build_s']:>7.3f} {r['solve_s']:>8.3f} {r['elapsed_s']:>8.3f} {spd:>6} "
            f"{obj:>9} {obj_vs:>8} {r['alpha_abs']:>9.2f} {vs:>8} {r['subgraph_edges']:>7} {pct:>6}"
        )
    print("=" * 124)
