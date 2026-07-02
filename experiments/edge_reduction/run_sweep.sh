#!/usr/bin/env bash
# Fan the edge-reduction experiment across cores: one process per w_balanced (λ),
# each running the full reduction-% sweep (0..70%) on its own digraph.
#
# Everything lands in a single self-contained folder
#   experiments/edge_reduction/sweeps/<instance>_np_<P>/
# holding: one lambda_<λ>/ subfolder per λ (its res.pkl/json + per-λ plots), a
# combined CSV across all λ, cross-λ overlay plots, and a run.log.
#
# Usage (from project root):
#   ./experiments/edge_reduction/run_sweep.sh <base-config.py>                 # default λ grid
#   ./experiments/edge_reduction/run_sweep.sh <base-config.py> 0.0 0.5 0.9      # custom λ grid
#
# The base config supplies everything except λ (assembly file, reduction grid,
# reps, gap, plots); λ is overridden per task. Env knobs: JOBS, NUM_PHASES,
# RUN_TAG.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <base-config.py> [lambda ...]" >&2
    exit 1
fi
CONFIG="$1"; shift
if [ ! -f "$CONFIG" ]; then echo "config not found: $CONFIG" >&2; exit 1; fi

if [ "$#" -gt 0 ]; then
    LAMBDAS=("$@")
else
    LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

# Pull instance + default P from the base config so the folder name and log are
# a single source of truth.
read -r INSTANCE DEFP FILE_NAME < <(python -c "
import importlib.util, os
s = importlib.util.spec_from_file_location('c', '$CONFIG')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
d = m.config_data
print(os.path.basename(os.path.dirname(d['assembly_fname'])),
      d.get('num_phases', 3), d['assembly_fname'])
")
P="${NUM_PHASES:-$DEFP}"

RUNDIR="experiments/edge_reduction/sweeps/${INSTANCE}_np_${P}${RUN_TAG:+_${RUN_TAG}}"
# Start clean; keep rm -rf pinned to the sweeps tree.
case "$RUNDIR" in experiments/edge_reduction/sweeps/*) rm -rf "$RUNDIR" ;; esac
mkdir -p "$RUNDIR"
LOG="${RUNDIR}/run.log"

JOBS="${JOBS:-$(nproc)}"
if [ "$JOBS" -gt "${#LAMBDAS[@]}" ]; then JOBS="${#LAMBDAS[@]}"; fi

{
    echo "================ edge-reduction λ sweep ================"
    echo "date         : $(date -Is)"
    echo "host         : $(hostname)"
    echo "git commit   : $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "python       : $(python --version 2>&1)"
    echo "base config  : ${CONFIG}"
    echo "instance     : ${INSTANCE}"
    echo "parts file   : ${FILE_NAME}"
    echo "num_phases   : ${P}"
    echo "lambda grid  : ${LAMBDAS[*]}"
    echo "jobs (par.)  : ${JOBS}"
    echo "rundir       : ${RUNDIR}"
    echo "======================================================="
} | tee "$LOG"

echo "Running ${#LAMBDAS[@]} λ configs, up to ${JOBS} at a time -> ${RUNDIR}/"

# -P runs JOBS at once; each task is fully independent (own digraph + own solves).
# Each task's stdout goes to its own log so parallel output is not interleaved.
printf '%s\n' "${LAMBDAS[@]}" | xargs -P "$JOBS" -I {} bash -c '
    lam="$1"; rundir="$2"; config="$3"; p="$4"
    python -m experiments.edge_reduction.run \
        --config-file "$config" \
        --w-balanced "$lam" \
        --num-phases "$p" \
        --res-fname "${rundir}/lambda_${lam}/res.pkl" \
        > "${rundir}/task_lambda_${lam}.log" 2>&1
' _ {} "$RUNDIR" "$CONFIG" "$P"

# Fold each task log into the run log, then drop the per-task logs.
for lam in "${LAMBDAS[@]}"; do
    tl="${RUNDIR}/task_lambda_${lam}.log"
    [ -f "$tl" ] || continue
    {
        echo
        echo "----------------- λ = ${lam} -----------------"
        cat "$tl"
    } >> "$LOG"
    rm -f "$tl"
done

# Combine per-λ results into one CSV, then overlay the λ curves.
COMBINED="${RUNDIR}/edge_reduction_lambda_sweep.csv"
python -m experiments.edge_reduction.combine_sweep "$RUNDIR" --out "$COMBINED" | tee -a "$LOG"
python -m experiments.edge_reduction.plot_lambda_sensitivity "$COMBINED" | tee -a "$LOG"

echo "Done. Results + plots + log in ${RUNDIR}/"
