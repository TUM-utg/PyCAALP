"""Combine a λ-sweep of edge-reduction runs into one tidy CSV.

Standalone and re-runnable (reads only saved output, writes no solve). Given a
sweep folder produced by ``run_sweep.sh`` — ``<instance>_np_<P>/`` holding
``lambda_<lam>/res.pkl`` subfolders — it flattens every per-λ result dict

    {red_perc: (mean_solve, mean_obj, mean_maxphase,
                std_solve,  std_obj,  std_maxphase,
                mean_total, std_total, mean_build, std_build)}

into one row per (λ, reduction%) with config-identity columns, so the sweep is
one concatenable CSV (matching the strategy_comparison / bl_union convention).

``mean_total_s`` is the honest wall-clock cost (digraph build incl. adaptive
protection + reduction, plus MIP construction + solve); ``mean_solve_s`` is the
pure MIP solve (``getSolvingTime``); ``mean_build_s`` is the build alone. Older
result pkls with only the first 6 fields leave the new columns blank.

Run (from project root)::

    python -m experiments.edge_reduction.combine_sweep \
        experiments/edge_reduction/sweeps/assembly_1_np_3
"""

import argparse
import csv
import glob
import os
import re

from pycaalp.gapp.file_formats import load_pkl

FIELDS = [
    "instance",
    "num_phases",
    "w_balanced",
    "red_perc",
    "mean_total_s",
    "mean_solve_s",
    "mean_build_s",
    "mean_objective",
    "mean_max_phase_time",
    "std_total_s",
    "std_solve_s",
    "std_build_s",
    "std_objective",
    "std_max_phase_time",
]


def _parse_rundir(rundir: str) -> tuple[str, int | str]:
    """Recover (instance, num_phases) from a ``<instance>_np_<P>[_tag]`` folder."""
    base = os.path.basename(os.path.normpath(rundir))
    match = re.search(r"_np_(\d+)", base)
    num_phases = int(match.group(1)) if match else ""
    instance = base.split("_np_")[0] if "_np_" in base else base
    return instance, num_phases


def combine_sweep(rundir: str, out_csv: str | None = None) -> str:
    """Write one CSV row per (λ, reduction%) for every lambda_*/res.pkl in rundir."""
    instance, num_phases = _parse_rundir(rundir)
    pkls = sorted(glob.glob(os.path.join(rundir, "lambda_*", "*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No lambda_*/res.pkl found under {rundir}")

    if out_csv is None:
        out_csv = os.path.join(rundir, "edge_reduction_lambda_sweep.csv")

    rows = []
    for pkl in pkls:
        lam_dir = os.path.basename(os.path.dirname(pkl))
        lam = lam_dir[len("lambda_"):] if lam_dir.startswith("lambda_") else lam_dir
        results = load_pkl(pkl)
        for red_perc, metrics in sorted(results.items()):
            # Total/build timing (indices 6-9) is absent in pre-timing pkls.
            has_total = len(metrics) >= 10
            rows.append(
                {
                    "instance": instance,
                    "num_phases": num_phases,
                    "w_balanced": lam,
                    "red_perc": red_perc,
                    "mean_total_s": metrics[6] if has_total else "",
                    "mean_solve_s": metrics[0],
                    "mean_build_s": metrics[8] if has_total else "",
                    "mean_objective": metrics[1],
                    "mean_max_phase_time": metrics[2],
                    "std_total_s": metrics[7] if has_total else "",
                    "std_solve_s": metrics[3],
                    "std_build_s": metrics[9] if has_total else "",
                    "std_objective": metrics[4],
                    "std_max_phase_time": metrics[5],
                }
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Combined {len(pkls)} λ runs ({len(rows)} rows) -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rundir", help="sweep folder (<instance>_np_<P>)")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args()
    combine_sweep(args.rundir, args.out)
