"""Seed the full-MIP reference cache from an existing bl_union_convergence CSV.

The full-MIP reference (obj + per-phase times) for each (instance, P, λ) is
already recorded in the ``full_ref`` rows of a previous run's CSV. Those solves
cost hours on assembly_2, so rather than re-solve them the first time the cached
harness runs, replay them into the pickle cache that
``bl_union_convergence.solve_full_ref_cached`` reads.

Only what the harness needs is reconstructed: ``objective`` and
``absolute_time_per_phase`` (for the gap/alpha references) plus per-phase op
counts (for the reference row's display). The actual operation identities are
not in the CSV and are not needed downstream.

Run from the project root::

    python -m experiments.bl_union_convergence.seed_cache_from_csv \
        experiments/bl_union_convergence/assembly_2_np_3/bl_union_convergence.csv
"""

import csv
import json
import os
import pickle
import sys

from experiments.bl_union_convergence.bl_union_convergence import CACHE_DIR, _cache_path


def seed(csv_path, cache_dir=CACHE_DIR, overwrite=False):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    os.makedirs(cache_dir, exist_ok=True)
    n_written = 0
    for r in rows:
        if r.get("stop_reason") != "full_ref":
            continue
        instance = r["instance"]
        num_phases = int(r["num_phases"])
        w_balanced = r["w_balanced"]  # keep the string form for the filename
        path = _cache_path(cache_dir, instance, num_phases, w_balanced)
        if os.path.exists(path) and not overwrite:
            print(f"  skip (exists): {path}")
            continue

        phase_times = json.loads(r["abs_time_per_phase"])
        ops_counts = json.loads(r["ops_per_phase"])
        # Rebuild the minimal structures the harness consumes.
        results = {
            "objective": float(r["objective"]),
            "absolute_time_per_phase": {i: t for i, t in enumerate(phase_times)},
        }
        # ops list: identities are lost, but only len() is used downstream.
        ops = [["_"] * c for c in ops_counts]

        with open(path, "wb") as out:
            pickle.dump(
                {
                    "instance": instance,
                    "num_phases": num_phases,
                    "w_balanced": float(w_balanced),
                    "n_nodes": int(r["digraph_nodes"]),
                    "n_edges": int(r["digraph_edges"]),
                    "results": results,
                    "ops": ops,
                    "solve_s": float(r["full_solve_s"]),
                },
                out,
            )
        n_written += 1
        print(f"  seeded: {path}  (obj={results['objective']:.4f})")

    print(f"\nSeeded {n_written} full-MIP references into {cache_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m experiments.bl_union_convergence.seed_cache_from_csv <csv> [--overwrite]")
    seed(sys.argv[1], overwrite="--overwrite" in sys.argv[2:])
