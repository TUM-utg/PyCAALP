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
import pickle
import time

from pycaalp.run import create_assembly_digraph, optimize
from pycaalp.time_balancing.subgraph_mip import (
    build_adaptive_subgraph,
    build_blended_union_subgraph,
    build_diverse_subgraph,
    solve_by_subgraph_mip,
)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

# Default target: assembly 2 (the reason this experiment exists). Override the
# instance by editing these two lines; everything else is config-driven.
# FILE_NAME = "data/assembly_2/assembly_2_parts.json"
# DFM_FILE_NAME = "data/assembly_2/assembly_2_dfm.json"

FILE_NAME = "data/assembly_1/assembly_1_parts.json"
DFM_FILE_NAME = ""

INSTANCE = os.path.basename(os.path.dirname(FILE_NAME))

NUM_PHASES = 3
W_BALANCED = 0.5

# k grid — denser at the low end so the subgraph-growth curve (obj/alpha vs %
# edges) is well sampled where the action is, then geometric. A SAFETY CAP: the
# stop criterion normally halts before the last value.
K_VALUES = [
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
]

# Blend values unioned by the bl-union strategy (frontier union). blend=0 is the
# pure edge_weight ranking, blend=1 the continuous balance ranking, the middle
# catches compromise paths. λ-agnostic (the MIP still solves at the true λ).
BLEND_GRID = [0.0, 0.5, 1.0]

METHODS_DEFAULT = "bl_union,diverse, adaptive"
PENALTY = 0.5  # diverse re-routing penalty (see diverse_shortest_paths)

# Stop criterion. Two useful ones (see --stop):
#   edge_saturation — grow until the subgraph genuinely stops expanding. Oracle-
#       free, so it is the honest "we cannot cover more of the graph" stop. This
#       is the DEFAULT for tracing the growth curve, because obj_plateau is too
#       eager: at high λ bl-union's objective flatlines while still >1% from the
#       optimum, so an objective plateau cannot tell "converged to the optimum"
#       from "stuck below it".
#   obj_plateau — relative objective improvement < OBJ_EPS. Cheap; fine at low λ
#       where bl-union really does hit the optimum, misleading at high λ.
#   none — run the whole k grid (full trajectory).
# Independently, --gap-target X stops as soon as the gap to the (cached) full-MIP
# objective is ≤ X% — the characterization stop: "how little graph for <X%".
STOP_DEFAULT = "edge_saturation"
OBJ_EPS = 0.5  # %: relative objective improvement below this counts as a plateau
PATIENCE = 2  # consecutive plateau / saturation steps required to stop

# Where per-(instance, P, λ) full-MIP references are cached (they cost 2–5 h each
# on assembly_2). Keyed so any run/experiment folder reuses the same solve.
CACHE_DIR = "experiments/bl_union_convergence/full_mip_cache"

RESULTS_FILE = "experiments/bl_union_convergence/bl_union_convergence.csv"

CSV_FIELDS = [
    # Config identity (so rows from different runs can be concatenated)
    "instance",
    "num_phases",
    "w_balanced",
    "method",  # subgraph strategy: bl_union | diverse | full_ref
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


def _cache_path(cache_dir, instance, num_phases, w_balanced):
    return os.path.join(
        cache_dir, f"full_ref_{instance}_P{num_phases}_lam{w_balanced}.pkl"
    )


def _solve_full(ad, num_phases, w_balanced):
    (results, ops), solve_s = _timed(
        optimize,
        assembly_digraph=ad,
        num_phases=num_phases,
        w_balanced=w_balanced,
        hide_output=True,
        full_result_output=True,
    )
    status = results.get("scip_status", "unknown")
    if status != "optimal":
        # Not proven optimal → this objective is only an incumbent and can be
        # BEATEN by an exactly-solved subgraph (negative "gaps"). Do not treat it
        # as the optimum.
        print(
            f"  !! FULL MIP NOT OPTIMAL (status={status}, gap={results.get('scip_gap')}) "
            f"— reference obj {results['objective']:.4f} is an incumbent, not the optimum"
        )
    return results, ops, solve_s, status


def solve_full_ref_cached(
    ad, instance, num_phases, w_balanced, cache_dir, refresh, no_cache=False
):
    """Return (results, ops, solve_s, cached) for the full MIP.

    Robustness (this reference must never be silently wrong):
    * ``no_cache`` — always solve fresh, no read/write.
    * cache entries are validated on digraph size (nodes/edges);
    * a cached entry that was NOT proven optimal is distrusted and re-solved;
    * **monotone-safe write:** the cache keeps the *lowest-objective* full solve
      ever seen (the true optimum is the min over all valid full solves), so a
      worse/early-stopped solve can never overwrite a better one. This is what
      prevents the poisoning that made a subgraph appear to beat the full MIP.
    """
    n_nodes = ad.assembly_digraph.number_of_nodes()
    n_edges = ad.assembly_digraph.number_of_edges()
    path = _cache_path(cache_dir, instance, num_phases, w_balanced)

    def _load_valid():
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            c = pickle.load(fh)
        if c.get("n_nodes") != n_nodes or c.get("n_edges") != n_edges:
            print(f"  [cache stale] {path} — digraph size changed")
            return None
        return c

    if no_cache:
        results, ops, solve_s, _ = _solve_full(ad, num_phases, w_balanced)
        return results, ops, solve_s, False

    if not refresh:
        c = _load_valid()
        if c is not None:
            status = c["results"].get("scip_status", "unknown")
            if status == "optimal" or status == "unknown":  # unknown = pre-status cache
                print(
                    f"  [cache hit] {path}  (solve was {c['solve_s']:.1f}s, status={status})"
                )
                return c["results"], c["ops"], c["solve_s"], True
            print(
                f"  [cache distrust] {path} status={status} (not optimal) — re-solving"
            )

    results, ops, solve_s, status = _solve_full(ad, num_phases, w_balanced)

    # Monotone-safe write: keep whichever full solve has the lower objective.
    existing = _load_valid()
    if (
        existing is not None
        and existing["results"]["objective"] <= results["objective"]
    ):
        print(
            f"  [cache keep] existing obj {existing['results']['objective']:.4f} "
            f"≤ new {results['objective']:.4f} — not overwriting"
        )
        return existing["results"], existing["ops"], existing["solve_s"], True

    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "instance": instance,
                "num_phases": num_phases,
                "w_balanced": w_balanced,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "results": results,
                "ops": ops,
                "solve_s": solve_s,
            },
            fh,
        )
    print(f"  [cache save] {path}  (status={status})")
    return results, ops, solve_s, False


def _record(
    ctx,
    method,
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
        "method": method,
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
    parser.add_argument(
        "--stop",
        choices=["edge_saturation", "obj_plateau", "none"],
        default=STOP_DEFAULT,
        help="k-sweep stop criterion (default: %(default)s)",
    )
    parser.add_argument(
        "--gap-target",
        type=float,
        default=None,
        help="also stop once the gap to the full-MIP objective is ≤ this %% "
        "(characterisation stop; uses the cached full-MIP reference)",
    )
    parser.add_argument(
        "--methods",
        default=METHODS_DEFAULT,
        help="comma list of subgraph strategies to sweep: bl_union, diverse "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=PENALTY,
        help="diverse re-routing penalty (default: %(default)s)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="cap the k grid (bounds Yen enumeration cost; the growth curve's "
        "interesting range on assembly_2 is small k)",
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore any cached full-MIP reference and re-solve it (still writes "
        "monotone-safe: keeps the better of old/new)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="never read or write the full-MIP cache — always solve fresh",
    )
    args = parser.parse_args()
    W_BALANCED = args.w_balanced
    NUM_PHASES = args.num_phases
    RESULTS_FILE = args.out
    STOP = args.stop
    GAP_TARGET = args.gap_target
    PENALTY = args.penalty
    METHODS = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.k_max is not None:
        K_VALUES = [k for k in K_VALUES if k <= args.k_max]

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
    results_full, ops_full, t_full, cached = solve_full_ref_cached(
        ad,
        INSTANCE,
        NUM_PHASES,
        W_BALANCED,
        args.cache_dir,
        args.refresh_cache,
        no_cache=args.no_cache,
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
        ctx,
        "full_ref",
        None,
        0.0,
        t_full,
        results_full,
        ops_full,
        n_edges,
        None,
        None,
        "full_ref",
    )
    _emit(r_full)
    print(f"  obj={full_obj:.4f}  alpha={full_alpha:.1f}s  time={t_full:.2f}s")

    # ------------------------------------------------------------------
    # Subgraph-growth k-sweep, once per method, with automatic stop
    # ------------------------------------------------------------------
    def _build(method, k):
        """Build the k-subgraph for one method (timed by the caller)."""
        if method == "bl_union":
            return build_blended_union_subgraph(ad, k, BLEND_GRID)
        if method == "diverse":
            return build_diverse_subgraph(ad, k, W_BALANCED, penalty=PENALTY)
        if method == "adaptive":
            return build_adaptive_subgraph(
                ad, k, BLEND_GRID, W_BALANCED, penalty=PENALTY
            )
        raise ValueError(f"unknown method: {method}")

    def run_method(method):
        prev_obj = None
        prev_edges = 0
        plateau_streak = 0
        sat_streak = 0

        print(
            f"\n[{method}] k-sweep (stop: {STOP}"
            + (f", gap_target={GAP_TARGET}%" if GAP_TARGET is not None else "")
            + "):"
        )
        for k in K_VALUES:
            sg, build_s = _timed(_build, method, k)
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

            if d_obj_pct is not None and d_obj_pct < OBJ_EPS:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if d_edges == 0:
                sat_streak += 1
            else:
                sat_streak = 0

            gap_pct = (obj_k / full_obj - 1) * 100 if full_obj else None

            # A subgraph is a subset of the full digraph, so its optimum can
            # NEVER beat the full-MIP optimum. gap_pct < 0 (beyond numerical
            # noise) therefore proves the cached full-MIP reference is stale /
            # corrupt — loudly flag it so the negative "gaps" are not mistaken
            # for a real result. Re-run with --refresh-cache to fix.
            if gap_pct is not None and gap_pct < -1e-3:
                print(
                    f"  !! STALE REFERENCE: subgraph obj {obj_k:.4f} beats cached "
                    f"full ref {full_obj:.4f} (gap {gap_pct:+.2f}%) at k={k}. "
                    f"The full_ref for λ={W_BALANCED} is corrupt — re-run with "
                    f"--refresh-cache."
                )

            # gap_target (characterisation) takes priority; else oracle-free stop.
            stop_reason = ""
            if GAP_TARGET is not None and gap_pct is not None and gap_pct <= GAP_TARGET:
                stop_reason = "gap_target"
            elif STOP == "edge_saturation" and sat_streak >= PATIENCE:
                stop_reason = "edge_saturation"
            elif STOP == "obj_plateau" and plateau_streak >= PATIENCE:
                stop_reason = "obj_plateau"

            row = _record(
                ctx,
                method,
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

            prev_obj = obj_k
            prev_edges = sg_edges

            if stop_reason:
                detail = (
                    f"gap {gap_pct:+.2f}% ≤ {GAP_TARGET}%"
                    if stop_reason == "gap_target"
                    else f"after {PATIENCE} consecutive steps"
                )
                print(f"  → stop: {stop_reason} ({detail})")
                break

    for method in METHODS:
        run_method(method)

    csv_file.close()
    print(f"\nResults saved to {RESULTS_FILE}")
    print(
        "Plot with:  python -m experiments.bl_union_convergence.plot_convergence "
        f"{RESULTS_FILE}"
    )
