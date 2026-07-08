"""Oracle-free convergence of the adaptive subgraph — for problems where the
full MIP cannot be solved.

Why this experiment (vs bl_union_convergence)
---------------------------------------------
``bl_union_convergence`` measures the subgraph objective's gap to the **full
MIP** optimum — an oracle that costs hours on big assemblies and does not
converge at all for high λ on assembly_2. This experiment removes the oracle: it
references two quantities computable in closed form, with no solve, so it scales
to problems where the full MIP is out of reach.

  * **Ideal objective (Σ shortest-path bound).** A valid lower bound on the MIP
    objective:  ``ideal = (1-λ)·c_min + λ·eef·max(T_total/P, t_max)`` where
    ``c_min = sum_of_sh_path_weights`` (the edge_weight-shortest path) and
    ``eef`` is the model's equal-effect factor. Since ``eef·(T_total/P) = c_min``
    this reduces to ``c_min`` whenever ``t_max ≤ T_total/P`` (the usual case).
  * **Phase width (T_total/P).** The perfect-balance floor for the makespan α.

Because both are valid bounds, the two reported gaps are always ≥ 0 and 0 is the
best attainable:
  * ``obj_vs_ideal_pct``   = (obj/ideal − 1)·100   — how far obj sits above the
    analytical floor (its residual = method suboptimality + the intrinsic
    misalignment the floor cannot see; loose at high λ, tight at low λ).
  * ``alpha_vs_width_pct`` = (α/(T_total/P) − 1)·100 — how far the makespan sits
    above perfect balance (0 = ideal split).

The adaptive method is the default; the subgraph grows on a k grid and the run
stops at 10 % of the full-graph edges (or earlier if it saturates). A per-k CSV
with the full timing breakdown is written; plot it with ``plot_convergence.py``.

Run (single λ, from the project root)::

    python -m experiments.adaptive_convergence.adaptive_convergence \
        --w-balanced 0.9 --num-phases 3 \
        --out experiments/adaptive_convergence/adaptive_convergence.csv
"""

import argparse
import csv
import json
import os
import time

import networkx as nx

from pycaalp.run import create_assembly_digraph
from pycaalp.time_balancing.subgraph_mip import (
    build_adaptive_subgraph,
    solve_by_subgraph_mip,
)

# ---------------------------------------------------------------------------
# Config (single-λ baseline; override on the CLI)
# ---------------------------------------------------------------------------
FILE_NAME = "data/assembly_1/assembly_1_parts.json"
DFM_FILE_NAME = ""
INSTANCE = os.path.basename(os.path.dirname(FILE_NAME))

NUM_PHASES = 3
W_BALANCED = 0.9  # the regime where the full MIP is unaffordable — the point

# k grid (geometric, dense at the low end). A safety cap: the stop normally fires
# earlier at STOP_PERC_GRAPH % of the edges.
K_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]

BLEND_GRID = [0.0, 0.5, 1.0]  # adaptive Phase-1 bl-union blend grid
PENALTY = 0.5  # adaptive Phase-2 diverse re-routing penalty
METHOD = "adaptive"

STOP_PERC_GRAPH = 10.0  # stop once the subgraph reaches this % of the digraph
OBJ_EPS = 0.5  # %: relative obj improvement below this counts as a plateau
PATIENCE = 2  # consecutive saturation steps before the edge-saturation backstop

RESULTS_FILE = "experiments/adaptive_convergence/adaptive_convergence.csv"

CSV_FIELDS = [
    # Config identity (rows from different runs concatenate cleanly)
    "instance",
    "num_phases",
    "w_balanced",
    "method",
    "k",
    # Size
    "digraph_nodes",
    "digraph_edges",
    "subgraph_edges",
    "subgraph_pct",  # subgraph_edges / digraph_edges * 100  (the x-axis)
    "d_edges",  # new edges vs previous k (0 ⇒ saturated)
    # Timing breakdown (seconds) — no full-MIP; this is the whole cost
    "digraph_build_s",  # one-off, shared by all k
    "build_s",  # adaptive enumeration + subgraph build
    "solve_s",  # subgraph MIP solve
    "elapsed_s",  # build_s + solve_s
    # Objective / balance vs the analytical (oracle-free) references
    "objective",
    "ideal_obj",  # (1-λ)c_min + λ·eef·max(T/P, t_max)
    "obj_vs_ideal_pct",  # (obj/ideal - 1)*100   ≥ 0
    "d_obj_pct",  # relative obj improvement vs previous k (≥0, monotone)
    # Decomposition of the objective gap: obj-ideal = (1-λ)(path_cost-c_min)
    #   + λ·eef·(alpha-alpha_floor). Cost component (this pair) + makespan
    #   component (alpha_vs_width below). Both grow with λ BY DESIGN (the MIP
    #   trades engineering cost for balance) — they locate the solution in the
    #   cost/balance trade-off, they are NOT optimality gaps.
    "path_cost",  # Σ edge_weight of the SELECTED assembly path
    "c_min",  # Σ edge_weight of the shortest path (= sum_of_sh_path_weights)
    "path_cost_vs_cmin_pct",  # (path_cost/c_min - 1)*100   ≥ 0
    "alpha_abs",  # max phase time (makespan)
    "phase_width",  # T_total / P
    "t_max_abs",  # longest single operation time (absolute)
    "alpha_vs_width_pct",  # (alpha/max(phase_width, t_max) - 1)*100 ≥ 0; 0 = floor
    "imbalance_pct",  # (max - min) / ideal_width * 100; P-agnostic
    "abs_time_per_phase",  # JSON list, any P
    "ops_per_phase",  # JSON list, any P
    "n_ops_total",
    # Bookkeeping
    "stop_perc_graph",  # configured subgraph-% stop threshold (for plot titles)
    "stop_reason",
]


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def analytical_ideal(ad, num_phases, lam):
    """Return (ideal_obj, phase_width_abs, t_max_abs): the oracle-free objective
    lower bound (in the MIP's objective units), the mean phase load, and the
    longest single operation's time (both in absolute units, to match
    alpha_abs / absolute_time_per_phase).

    ``ideal_obj`` uses the model's *scaled* ``time`` (as the MIP objective and eef
    do). The achievable makespan floor is ``max(phase_width_abs, t_max_abs)``:
    when one operation is longer than the mean phase load, that op — not T/P — is
    the tightest floor, since its phase's time is ≥ t_max.
    """
    time_scaled = nx.get_edge_attributes(ad.graph, "time")  # model/objective units
    time_abs = nx.get_edge_attributes(ad.graph, "absolute_time")  # reporting units
    t_total_s = sum(time_scaled.values())
    t_max_s = max(time_scaled.values()) if time_scaled else 0.0
    c_min = ad.sum_of_sh_path_weights
    eef = c_min * num_phases / t_total_s if t_total_s else 0.0
    # α floor in the model's scaled units: max(mean load, largest single op).
    alpha_floor_s = max(t_total_s / num_phases if num_phases else 0.0, t_max_s)
    ideal_obj = (1 - lam) * c_min + lam * eef * alpha_floor_s
    phase_width_abs = sum(time_abs.values()) / num_phases if num_phases else 0.0
    t_max_abs = max(time_abs.values()) if time_abs else 0.0
    return ideal_obj, phase_width_abs, t_max_abs


def _row(
    ctx,
    k,
    build_s,
    solve_s,
    results,
    ops_list,
    sg_edges,
    d_edges,
    d_obj_pct,
    stop_reason,
):
    num_phases = ctx["num_phases"]
    abs_times = results["absolute_time_per_phase"]
    phase_times = [abs_times.get(p, 0.0) for p in range(num_phases)]
    alpha_abs = max(phase_times) if phase_times else 0.0
    pt_min = min(phase_times) if phase_times else 0.0
    phase_width = ctx["phase_width"]  # = T_total/P (absolute time units)
    t_max_abs = ctx["t_max_abs"]  # longest single operation (absolute)
    # Achievable makespan floor: the longest op when it exceeds the mean load.
    alpha_floor = max(phase_width, t_max_abs)
    ideal_obj = ctx["ideal_obj"]
    objective = results.get("objective")

    obj_vs_ideal = (
        round((objective / ideal_obj - 1) * 100, 2)
        if objective is not None and ideal_obj
        else None
    )
    alpha_vs_width = (
        round((alpha_abs / alpha_floor - 1) * 100, 2) if alpha_floor else None
    )
    # imbalance stays a spread metric relative to the mean phase load.
    imbalance = (
        round((alpha_abs - pt_min) / phase_width * 100, 2) if phase_width else None
    )
    # Cost component of the objective gap: Σ edge_weight over the selected edges
    # (results["operations"] is keyed by the (u, v) edge tuple) vs the shortest
    # path c_min. Both in edge_weight units, directly comparable.
    edge_w = ctx["edge_weight"]
    c_min = ctx["c_min"]
    path_cost = sum(edge_w.get(e, 0.0) for e in results.get("operations", {}))
    path_cost_vs_cmin = round((path_cost / c_min - 1) * 100, 2) if c_min else None
    elapsed = build_s + solve_s
    digraph_edges = ctx["digraph_edges"]
    n_ops = [len(p) for p in ops_list]

    return {
        "instance": ctx["instance"],
        "num_phases": num_phases,
        "w_balanced": ctx["w_balanced"],
        "method": ctx["method"],
        "k": k,
        "digraph_nodes": ctx["digraph_nodes"],
        "digraph_edges": digraph_edges,
        "subgraph_edges": sg_edges,
        "subgraph_pct": (
            round(sg_edges / digraph_edges * 100, 2) if digraph_edges else ""
        ),
        "d_edges": d_edges if d_edges is not None else "",
        "digraph_build_s": round(ctx["digraph_build_s"], 3),
        "build_s": round(build_s, 3),
        "solve_s": round(solve_s, 3),
        "elapsed_s": round(elapsed, 3),
        "objective": round(objective, 4) if objective is not None else "",
        "ideal_obj": round(ideal_obj, 4),
        "obj_vs_ideal_pct": obj_vs_ideal if obj_vs_ideal is not None else "",
        "d_obj_pct": round(d_obj_pct, 4) if d_obj_pct is not None else "",
        "path_cost": round(path_cost, 4),
        "c_min": round(c_min, 4),
        "path_cost_vs_cmin_pct": (
            path_cost_vs_cmin if path_cost_vs_cmin is not None else ""
        ),
        "alpha_abs": round(alpha_abs, 3),
        "phase_width": round(phase_width, 3),
        "t_max_abs": round(t_max_abs, 3),
        "alpha_vs_width_pct": alpha_vs_width if alpha_vs_width is not None else "",
        "imbalance_pct": imbalance if imbalance is not None else "",
        "abs_time_per_phase": json.dumps([round(t, 3) for t in phase_times]),
        "ops_per_phase": json.dumps(n_ops),
        "n_ops_total": sum(n_ops),
        "stop_perc_graph": ctx["stop_perc_graph"],
        "stop_reason": stop_reason,
    }


def main():
    global FILE_NAME, DFM_FILE_NAME, INSTANCE, NUM_PHASES, W_BALANCED
    global RESULTS_FILE, STOP_PERC_GRAPH, K_VALUES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w-balanced", type=float, default=W_BALANCED)
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    parser.add_argument("--file-name", default=FILE_NAME)
    parser.add_argument("--dfm-file", default=DFM_FILE_NAME)
    parser.add_argument("--out", default=RESULTS_FILE)
    parser.add_argument(
        "--stop-perc-graph",
        type=float,
        default=STOP_PERC_GRAPH,
        help="stop once the subgraph reaches this %% of edges (100 = disabled)",
    )
    parser.add_argument("--k-max", type=int, default=None)
    args = parser.parse_args()

    FILE_NAME = args.file_name
    DFM_FILE_NAME = args.dfm_file
    INSTANCE = os.path.basename(os.path.dirname(FILE_NAME))
    NUM_PHASES = args.num_phases
    W_BALANCED = args.w_balanced
    RESULTS_FILE = args.out
    STOP_PERC_GRAPH = args.stop_perc_graph
    if args.k_max is not None:
        K_VALUES = [k for k in K_VALUES if k <= args.k_max]

    print("=" * 72)
    print(
        f"adaptive convergence (oracle-free) — {INSTANCE}  "
        f"P={NUM_PHASES}  λ={W_BALANCED}"
    )
    print("=" * 72)

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

    ideal_obj, phase_width, t_max_abs = analytical_ideal(ad, NUM_PHASES, W_BALANCED)
    print(f"  ideal obj (Σ shortest-path bound) = {ideal_obj:.4f}")
    print(f"  phase width (T_total/P)           = {phase_width:.2f}")
    print(f"  longest op (t_max)                = {t_max_abs:.2f}"
          + ("  ← exceeds phase width, is the floor" if t_max_abs > phase_width else ""))

    ctx = {
        "instance": INSTANCE,
        "num_phases": NUM_PHASES,
        "w_balanced": W_BALANCED,
        "method": METHOD,
        "digraph_nodes": n_nodes,
        "digraph_edges": n_edges,
        "digraph_build_s": build_time,
        "ideal_obj": ideal_obj,
        "phase_width": phase_width,
        "t_max_abs": t_max_abs,
        "stop_perc_graph": STOP_PERC_GRAPH,
        "c_min": ad.sum_of_sh_path_weights,
        # edge_weight over the full digraph; selected edges are a subset.
        "edge_weight": nx.get_edge_attributes(ad.assembly_digraph, "edge_weight"),
    }

    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        print(f"\n[{METHOD}] k-sweep (stop at {STOP_PERC_GRAPH}% of edges):")
        prev_obj = None
        prev_edges = 0
        sat_streak = 0
        for k in K_VALUES:
            sg, build_s = _timed(
                build_adaptive_subgraph,
                ad,
                k,
                BLEND_GRID,
                W_BALANCED,
                penalty=PENALTY,
            )
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

            d_obj_pct = None
            if prev_obj is not None and prev_obj:
                d_obj_pct = (prev_obj - obj_k) / abs(prev_obj) * 100
            if d_edges == 0:
                sat_streak += 1
            else:
                sat_streak = 0

            sub_pct = sg_edges / n_edges * 100 if n_edges else 0.0
            stop_reason = ""
            if sat_streak >= PATIENCE:
                stop_reason = "edge_saturation"
            elif STOP_PERC_GRAPH and sub_pct > STOP_PERC_GRAPH:
                stop_reason = "stop_perc_graph"

            row = _row(
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
            writer.writerow(row)
            csv_file.flush()

            dobj = f"{d_obj_pct:+.3f}%" if d_obj_pct is not None else "   —   "
            print(
                f"  k={k:>5}  {sub_pct:5.1f}% edges  obj={obj_k:.4f}  "
                f"obj_vs_ideal={row['obj_vs_ideal_pct']}%  "
                f"α_vs_width={row['alpha_vs_width_pct']}%  Δobj={dobj}  "
                f"build={build_s:.2f}s solve={solve_s:.2f}s"
            )

            prev_obj = obj_k
            prev_edges = sg_edges
            if stop_reason:
                print(f"  → stop: {stop_reason}")
                break

    print(f"\nDone. CSV → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
