"""Experiment: objective convergence of the blended-union (bl-union) subgraph
strategy as k grows, on a single (instance, P, λ) config.

Why this experiment
-------------------
The strategy comparison picked ``bl-union`` (frontier union of the blended
enumeration over a small blend grid) as the selected method: its objective gap
to the full MIP is consistently <1%. This script isolates that one method and
sweeps k on a geometric (×2) grid to answer "how large must k be before the
objective converges, and at what subgraph size (% of full-graph edges) does
that happen".

The x-axis that matters is **% of full-graph edges**, not k: k-shortest paths
overlap heavily, so the k→edges map is graph-dependent and a fixed "good k" does
not transfer across assemblies. We therefore record ``subgraph_pct`` on every
row and stop automatically when the subgraph stops growing or the objective
plateaus — a graph-driven stop criterion rather than a magic k.

Automatic stop
--------------
The bl-union subgraph is monotone in k (a union), so the MIP objective over it is
non-increasing. We stop when either holds for ``PATIENCE`` consecutive steps:
  * objective plateau  — relative improvement < ``OBJ_EPS``;
  * edge saturation    — the subgraph gained no new edges.
``K_VALUES`` is only a safety cap; the run normally halts earlier.

Run (single config) from the project root::

    python -m experiments.bl_union_convergence.bl_union_convergence \
        --w-balanced 0.5 --num-phases 3 --out run/conv_0.5.csv

Fan a λ grid across cores with ``run_sweep.sh`` (see that script).

NOTE on assembly 2: the full MIP does not converge for λ > 0.85, so keep the λ
grid at or below 0.85 for that instance (the run scripts default to this).
"""

import argparse
import csv
import json
import os
import time

from pycaalp.run import create_assembly_digraph, optimize
from pycaalp.time_balancing.subgraph_mip import (
    build_blended_union_subgraph,
    solve_by_subgraph_mip,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

# Default target: assembly 2 (the reason this experiment exists). Override the
# instance by editing these two lines; everything else is config-driven.
FILE_NAME = "data/assembly_2/assembly_2_parts.json"
DFM_FILE_NAME = "data/assembly_2/assembly_2_dfm.json"

INSTANCE = os.path.basename(os.path.dirname(FILE_NAME))

NUM_PHASES = 3
W_BALANCED = 0.5

# Geometric (×2) k grid — clean log-x convergence. This is a SAFETY CAP: the
# automatic stop normally halts well before the last value.
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Blend values unioned by the bl-union strategy (frontier union). blend=0 is the
# pure edge_weight ranking, blend=1 the continuous balance ranking, the middle
# catches compromise paths. λ-agnostic (the MIP still solves at the true λ).
BLEND_GRID = [0.0, 0.5, 1.0]

# Automatic stop criterion.
OBJ_EPS = 0.5  # %: relative objective improvement below this counts as a plateau
PATIENCE = 2  # consecutive plateau / saturation steps required to stop

RESULTS_FILE = "experiments/bl_union_convergence/bl_union_convergence.csv"

CSV_FIELDS = [
    # Config identity (so rows from different runs can be concatenated)
    "instance",
    "num_phases",
    "w_balanced",
    "k",
    # Problem size
    "digraph_nodes",
    "digraph_edges",
    "subgraph_edges",
    "subgraph_pct",  # subgraph_edges / digraph_edges * 100  (the meaningful x-axis)
    "d_edges",  # new edges vs previous k (0 ⇒ saturated)
    # Timing breakdown (seconds)
    "digraph_build_s",  # one-off, shared by all k
    "build_s",  # bl-union enumeration + subgraph build
    "solve_s",  # MIP solve
    "elapsed_s",  # build_s + solve_s
    "full_solve_s",  # full-MIP reference solve time
    "speedup_vs_full",  # full_solve_s / elapsed_s
    # Objective / balance quality
    "objective",
    "obj_vs_full_pct",  # gap to full-MIP objective (the headline metric)
    "d_obj_pct",  # relative objective improvement vs previous k (≥0, monotone)
    "alpha_abs",  # max phase time (makespan)
    "vs_full_pct",  # alpha vs full
    "phase_time_min",
    "phase_time_max",
    "phase_time_std",
    "imbalance_pct",  # (max - min) / ideal * 100; P-agnostic
    "abs_time_per_phase",  # JSON list, any P
    "ops_per_phase",  # JSON list, any P
    "n_ops_total",
    # Convergence bookkeeping
    "stop_reason",  # "" until the row that triggers the stop
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
    k,
    build_s,
    solve_s,
    results,
    ops_list,
    subgraph_edges,
    d_edges,
    d_obj_pct,
    stop_reason,
):
    """Build one CSV row. ctx carries the per-run constants (instance, P, λ,
    digraph size/build time, full-MIP reference alpha/obj/solve time)."""
    num_phases = ctx["num_phases"]
    abs_times = results["absolute_time_per_phase"]
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
        "k": k if k is not None else "",
        "digraph_nodes": ctx["digraph_nodes"],
        "digraph_edges": digraph_edges,
        "subgraph_edges": subgraph_edges,
        "subgraph_pct": (
            round(subgraph_edges / digraph_edges * 100, 2) if digraph_edges else ""
        ),
        "d_edges": d_edges if d_edges is not None else "",
        "digraph_build_s": round(ctx["digraph_build_s"], 3),
        "build_s": round(build_s, 3),
        "solve_s": round(solve_s, 3),
        "elapsed_s": round(elapsed, 3),
        "full_solve_s": round(ctx["full_solve_s"], 3),
        "speedup_vs_full": speedup if speedup is not None else "",
        "objective": round(objective, 4) if objective is not None else "",
        "obj_vs_full_pct": obj_vs_full if obj_vs_full is not None else "",
        "d_obj_pct": round(d_obj_pct, 4) if d_obj_pct is not None else "",
        "alpha_abs": round(alpha_abs, 3),
        "vs_full_pct": vs_full if vs_full is not None else "",
        "phase_time_min": round(pt_min, 3),
        "phase_time_max": round(alpha_abs, 3),
        "phase_time_std": round(std, 3),
        "imbalance_pct": imbalance_pct if imbalance_pct is not None else "",
        "abs_time_per_phase": json.dumps([round(t, 3) for t in phase_times]),
        "ops_per_phase": json.dumps(n_ops),
        "n_ops_total": sum(n_ops),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
    print(f"bl-union convergence — {INSTANCE}  P={NUM_PHASES}  λ={W_BALANCED}")
    print("=" * 72)

    if INSTANCE == "assembly_2" and W_BALANCED > 0.85:
        # Full MIP does not converge here; the reference row would hang.
        print(
            f"WARNING: assembly_2 full MIP does not converge for λ={W_BALANCED} "
            "(> 0.85). This run may not terminate."
        )

    # Build digraph once. Pass num_phases so the blended weight's phase
    # boundaries match P (set_blended_weights uses ad.num_phases).
    print("\nBuilding assembly digraph …")
    ad, build_time = _timed(
        create_assembly_digraph,
        file_name=FILE_NAME,
        w_bal=W_BALANCED,
        dfm_file=DFM_FILE_NAME,
        num_phases=NUM_PHASES,
    )
    n_nodes = ad.assembly_digraph.number_of_nodes()
    n_edges = ad.assembly_digraph.number_of_edges()
    print(f"  {n_nodes} nodes, {n_edges} edges  ({build_time:.1f}s)")

    # Open the CSV up front and flush each row, so a killed cluster job keeps
    # whatever k it has already reached.
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    csv_file = open(RESULTS_FILE, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    def _emit(row):
        writer.writerow(row)
        csv_file.flush()

    # ------------------------------------------------------------------
    # Full-MIP reference (single solve, no k)
    # ------------------------------------------------------------------
    print(f"\n[ref] Full MIP  P={NUM_PHASES}  λ={W_BALANCED} …")
    (results_full, ops_full), t_full = _timed(
        optimize,
        assembly_digraph=ad,
        num_phases=NUM_PHASES,
        w_balanced=W_BALANCED,
        hide_output=True,
        full_result_output=True,
    )
    full_alpha = max(results_full["absolute_time_per_phase"].values())
    full_obj = results_full["objective"]

    ctx = {
        "instance": INSTANCE,
        "num_phases": NUM_PHASES,
        "w_balanced": W_BALANCED,
        "digraph_nodes": n_nodes,
        "digraph_edges": n_edges,
        "digraph_build_s": build_time,
        "full_alpha": full_alpha,
        "full_obj": full_obj,
        "full_solve_s": t_full,
    }

    r_full = _record(
        ctx, None, 0.0, t_full, results_full, ops_full, n_edges, None, None, "full_ref"
    )
    _emit(r_full)
    print(f"  obj={full_obj:.4f}  alpha={full_alpha:.1f}s  time={t_full:.2f}s")

    # ------------------------------------------------------------------
    # bl-union over the geometric k grid, with automatic early stop
    # ------------------------------------------------------------------
    prev_obj = None
    prev_edges = 0
    plateau_streak = 0
    sat_streak = 0

    print("\nbl-union k-sweep (auto-stop on plateau / edge saturation):")
    for k in K_VALUES:
        sg, build_s = _timed(build_blended_union_subgraph, ad, k, BLEND_GRID)
        sg_edges = sg.number_of_edges()
        d_edges = sg_edges - prev_edges

        (results_k, ops_k), solve_s = _timed(
            solve_by_subgraph_mip,
            assembly_digraph_obj=ad,
            k=k,
            num_phases=NUM_PHASES,
            w_balanced=W_BALANCED,
            hide_output=True,
            full_result_output=True,
            subgraph=sg,
        )
        obj_k = results_k["objective"]

        # Relative objective improvement vs previous k (monotone, ≥0).
        d_obj_pct = None
        if prev_obj is not None and prev_obj:
            d_obj_pct = (prev_obj - obj_k) / abs(prev_obj) * 100

        # Update streaks.
        if d_obj_pct is not None and d_obj_pct < OBJ_EPS:
            plateau_streak += 1
        else:
            plateau_streak = 0
        if d_edges == 0:
            sat_streak += 1
        else:
            sat_streak = 0

        stop_reason = ""
        if plateau_streak >= PATIENCE:
            stop_reason = "obj_plateau"
        elif sat_streak >= PATIENCE:
            stop_reason = "edge_saturation"

        row = _record(
            ctx,
            k,
            build_s,
            solve_s,
            results_k,
            ops_k,
            sg_edges,
            d_edges,
            d_obj_pct,
            stop_reason,
        )
        _emit(row)

        gap = row["obj_vs_full_pct"]
        pct = row["subgraph_pct"]
        dobj = f"{d_obj_pct:+.3f}%" if d_obj_pct is not None else "   —   "
        print(
            f"  k={k:>5}  edges={sg_edges:>4} ({pct:>5}%)  Δe={d_edges:>4}  "
            f"obj={obj_k:.4f}  gap={gap:+.2f}%  Δobj={dobj}  "
            f"build={build_s:.2f}s solve={solve_s:.2f}s"
        )

        # Carry state to the next k so the deltas / stop streaks are computed
        # against the previous step (not against the initial 0 / None).
        prev_obj = obj_k
        prev_edges = sg_edges

        if stop_reason:
            print(f"  → stop: {stop_reason} (after {PATIENCE} consecutive steps)")
            break

    csv_file.close()
    print(f"\nResults saved to {RESULTS_FILE}")
    print(
        "Plot with:  python -m experiments.bl_union_convergence.plot_convergence "
        f"{RESULTS_FILE}"
    )
