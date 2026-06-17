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

import csv
import time

from pycaalp.run import create_assembly_digraph, optimize
from pycaalp.time_balancing.path_mip import solve_by_path_mip
from pycaalp.time_balancing.subgraph_mip import (
    build_kpath_subgraph,
    solve_by_subgraph_mip,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

FILE_NAME = "data/assembly_1/assembly_1_2_tech_parts.json"

NUM_PHASES = 7
W_BALANCED = 0.9

K_VALUES = [10, 50, 200, 500]

RESULTS_FILE = "experiments/strategy_comparison/strategy_comparison.csv"

CSV_FIELDS = [
    "strategy",
    "weight_attr",
    "k",
    "elapsed_s",
    "digraph_edges",
    "subgraph_edges",
    "objective",
    "obj_vs_full_pct",
    "alpha_abs",
    "abs_time_phase_0",
    "abs_time_phase_1",
    "abs_time_phase_2",
    "ops_phase_0",
    "ops_phase_1",
    "ops_phase_2",
    "vs_full_pct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _record(
    strategy,
    weight_attr,
    k,
    elapsed,
    results,
    ops_list,
    full_alpha,
    full_obj,
    digraph_edges,
    subgraph_edges=None,
):
    abs_times = results["absolute_time_per_phase"]
    alpha_abs = max(abs_times.values())
    vs_full = round((alpha_abs / full_alpha - 1) * 100, 2) if full_alpha else None
    objective = results.get("objective")
    obj_vs_full = (
        round((objective / full_obj - 1) * 100, 2)
        if objective is not None and full_obj
        else None
    )

    return {
        "strategy": strategy,
        "weight_attr": weight_attr,
        "k": k if k is not None else "",
        "elapsed_s": round(elapsed, 3),
        "digraph_edges": digraph_edges,
        "subgraph_edges": (
            subgraph_edges if subgraph_edges is not None else digraph_edges
        ),
        "objective": round(objective, 3) if objective is not None else "",
        "obj_vs_full_pct": obj_vs_full if obj_vs_full is not None else "",
        "alpha_abs": round(alpha_abs, 3),
        "abs_time_phase_0": round(abs_times.get(0, 0), 3),
        "abs_time_phase_1": round(abs_times.get(1, 0), 3),
        "abs_time_phase_2": round(abs_times.get(2, 0), 3),
        "ops_phase_0": len(ops_list[0]) if len(ops_list) > 0 else 0,
        "ops_phase_1": len(ops_list[1]) if len(ops_list) > 1 else 0,
        "ops_phase_2": len(ops_list[2]) if len(ops_list) > 2 else 0,
        "vs_full_pct": vs_full if vs_full is not None else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Strategy Comparison Experiment — Assembly 1")
    print("=" * 72)

    # Build digraph once (time_balanced_weight is set at construction time)
    print("\nBuilding assembly digraph …")
    ad, build_time = _timed(
        create_assembly_digraph, file_name=FILE_NAME, w_bal=W_BALANCED
    )
    n_edges = ad.assembly_digraph.number_of_edges()
    print(
        f"  {ad.assembly_digraph.number_of_nodes()} nodes, {n_edges} edges  ({build_time:.1f}s)"
    )

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
        hide_output=True,
        full_result_output=True,
    )
    full_alpha = max(results1["absolute_time_per_phase"].values())
    full_obj = results1["objective"]
    r1 = _record(
        "full_mip", "—", None, t1, results1, ops1, full_alpha, full_obj, n_edges
    )
    records.append(r1)
    print(f"  obj={r1['objective']:.3f}  alpha={r1['alpha_abs']:.1f}s  time={t1:.2f}s")

    # ------------------------------------------------------------------
    # Methods 2 & 3: sweep over k, both weight variants
    # ------------------------------------------------------------------
    for k in K_VALUES:
        print(f"\nk = {k}")

        # Subgraph sizes for reporting
        sg_ew = build_kpath_subgraph(ad, k, weight_attr="edge_weight")
        sg_bw = build_kpath_subgraph(ad, k, weight_attr="time_balanced_weight")
        sg_cw = build_kpath_subgraph(
            ad, k, weight_attr=["edge_weight", "time_balanced_weight"]
        )
        sg_ew_edges = sg_ew.number_of_edges()
        sg_bw_edges = sg_bw.number_of_edges()
        sg_cw_edges = sg_cw.number_of_edges()
        print(
            f"  Subgraph edges — edge_w: {sg_ew_edges} ({sg_ew_edges/n_edges*100:.1f}%)  "
            f"bal_w: {sg_bw_edges} ({sg_bw_edges/n_edges*100:.1f}%)  "
            f"combined: {sg_cw_edges} ({sg_cw_edges/n_edges*100:.1f}%)"
        )

        # # 2a: Path-Enum MIP — edge_weight
        # print(f"  [2a] Path-Enum / edge_w  k={k} …")
        # (results2a, ops2a), t2a = _timed(
        #     solve_by_path_mip,
        #     assembly_digraph_obj=ad,
        #     k=k,
        #     num_phases=NUM_PHASES,
        #     w_balanced=W_BALANCED,
        #     hide_output=True,
        #     full_result_output=True,
        #     weight_attr="edge_weight",
        # )
        # r2a = _record(
        #     "path_enum_mip",
        #     "edge_weight",
        #     k,
        #     t2a,
        #     results2a,
        #     ops2a,
        #     full_alpha,
        #     full_obj,
        #     n_edges,
        # )
        # records.append(r2a)
        # print(
        #     f"      obj={r2a['objective']:.3f} (vs_full={r2a['obj_vs_full_pct']:+.2f}%)  alpha={r2a['alpha_abs']:.1f}s  time={t2a:.3f}s  vs_full={r2a['vs_full_pct']:+.2f}%"
        # )

        # 2b: Path-Enum MIP — time_balanced_weight
        # print(f"  [2b] Path-Enum / bal_w   k={k} …")
        # (results2b, ops2b), t2b = _timed(
        #     solve_by_path_mip,
        #     assembly_digraph_obj=ad,
        #     k=k,
        #     num_phases=NUM_PHASES,
        #     w_balanced=W_BALANCED,
        #     hide_output=True,
        #     full_result_output=True,
        #     weight_attr="time_balanced_weight",
        # )
        # r2b = _record(
        #     "path_enum_mip",
        #     "time_balanced_weight",
        #     k,
        #     t2b,
        #     results2b,
        #     ops2b,
        #     full_alpha,
        #     full_obj,
        #     n_edges,
        # )
        # records.append(r2b)
        # print(
        #     f"      obj={r2b['objective']:.3f} (vs_full={r2b['obj_vs_full_pct']:+.2f}%)  alpha={r2b['alpha_abs']:.1f}s  time={t2b:.3f}s  vs_full={r2b['vs_full_pct']:+.2f}%"
        # )

        # 3a: Subgraph MIP — edge_weight (disabled for now)
        # print(f"  [3a] Subgraph / edge_w   k={k} …")
        # (results3a, ops3a), t3a = _timed(
        #     solve_by_subgraph_mip,
        #     assembly_digraph_obj=ad,
        #     k=k,
        #     num_phases=NUM_PHASES,
        #     w_balanced=W_BALANCED,
        #     hide_output=True,
        #     full_result_output=True,
        #     weight_attr="edge_weight",
        # )
        # r3a = _record(
        #     "subgraph_mip",
        #     "edge_weight",
        #     k,
        #     t3a,
        #     results3a,
        #     ops3a,
        #     full_alpha,
        #     full_obj,
        #     n_edges,
        #     sg_ew_edges,
        # )
        # records.append(r3a)
        # print(
        #     f"      obj={r3a['objective']:.3f} (vs_full={r3a['obj_vs_full_pct']:+.2f}%)  alpha={r3a['alpha_abs']:.1f}s  time={t3a:.3f}s  vs_full={r3a['vs_full_pct']:+.2f}%"
        # )

        # 3b: Subgraph MIP — time_balanced_weight (disabled for now)
        # print(f"  [3b] Subgraph / bal_w    k={k} …")
        # (results3b, ops3b), t3b = _timed(
        #     solve_by_subgraph_mip,
        #     assembly_digraph_obj=ad,
        #     k=k,
        #     num_phases=NUM_PHASES,
        #     w_balanced=W_BALANCED,
        #     hide_output=True,
        #     full_result_output=True,
        #     weight_attr="time_balanced_weight",
        # )
        # r3b = _record(
        #     "subgraph_mip",
        #     "time_balanced_weight",
        #     k,
        #     t3b,
        #     results3b,
        #     ops3b,
        #     full_alpha,
        #     full_obj,
        #     n_edges,
        #     sg_bw_edges,
        # )
        # records.append(r3b)
        # print(
        #     f"      obj={r3b['objective']:.3f} (vs_full={r3b['obj_vs_full_pct']:+.2f}%)  alpha={r3b['alpha_abs']:.1f}s  time={t3b:.3f}s  vs_full={r3b['vs_full_pct']:+.2f}%"
        # )

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
            weight_attr=["edge_weight", "time_balanced_weight"],
        )
        r3c = _record(
            "subgraph_mip",
            "combined",
            k,
            t3c,
            results3c,
            ops3c,
            full_alpha,
            full_obj,
            n_edges,
            sg_cw_edges,
        )
        records.append(r3c)
        print(
            f"      obj={r3c['objective']:.3f} (vs_full={r3c['obj_vs_full_pct']:+.2f}%)  alpha={r3c['alpha_abs']:.1f}s  time={t3c:.3f}s  vs_full={r3c['vs_full_pct']:+.2f}%"
        )

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = (
        f"{'Strategy':<16} {'Weight':<22} {'k':>6} "
        f"{'Time(s)':>8} {'Obj':>10} {'Obj vsFull':>11} {'Alpha(s)':>9} {'vs Full':>8} {'Edges':>7}"
    )
    print(hdr)
    print("-" * 100)
    for r in records:
        vs = f"{r['vs_full_pct']:+.2f}%" if r["vs_full_pct"] != "" else "—"
        obj = f"{r['objective']:.3f}" if r["objective"] != "" else "—"
        obj_vs = f"{r['obj_vs_full_pct']:+.2f}%" if r["obj_vs_full_pct"] != "" else "—"
        print(
            f"{r['strategy']:<16} {r['weight_attr']:<22} {str(r['k']):>6} "
            f"{r['elapsed_s']:>8.3f} {obj:>10} {obj_vs:>11} {r['alpha_abs']:>9.2f} {vs:>8} {r['subgraph_edges']:>7}"
        )
    print("=" * 100)
