#!/usr/bin/env bash
# Fan the strategy-comparison sweep across cores: one process per w_balanced (λ),
# each building its own digraph and solving its own MIPs, writing a per-task CSV.
#
# Everything for one run lands in a single self-contained folder
#   experiments/strategy_comparison/<instance>_np_<P>/
# holding: the per-λ CSVs, the combined CSV, a run.log with the full experiment
# details (+ each task's stdout), and the plots — so results and figures live
# together and a folder can be archived or moved as a unit.
#
# Usage (from project root):
#   ./experiments/strategy_comparison/run_sweep.sh                  # default λ grid
#   ./experiments/strategy_comparison/run_sweep.sh 0.0 0.25 0.5 1.0 # custom λ grid
#
# Env overrides:
#   JOBS       max concurrent processes (default: nproc, capped at #λ)
#   NUM_PHASES phases passed to every task (default: script default, P)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODULE="experiments.strategy_comparison.test_strategy_comparison"

# λ grid: CLI args, or a default spread across [0, 1].
if [ "$#" -gt 0 ]; then
    LAMBDAS=("$@")
else
    LAMBDAS=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
fi

# Pull config from the experiment module so the folder name and log stay a
# single source of truth (instance, default P, k grid, data files).
read -r INSTANCE DEFP < <(python -c \
    "import ${MODULE} as t; print(t.INSTANCE, t.NUM_PHASES)")
P="${NUM_PHASES:-$DEFP}"
KVALUES=$(python -c "import ${MODULE} as t; print(t.K_VALUES)")
FILE_NAME=$(python -c "import ${MODULE} as t; print(t.FILE_NAME)")
DFM_FILE=$(python -c "import ${MODULE} as t; print(t.DFM_FILE_NAME)")

RUNDIR="experiments/strategy_comparison/${INSTANCE}_np_${P}"
mkdir -p "$RUNDIR"
LOG="${RUNDIR}/run.log"

JOBS="${JOBS:-$(nproc)}"
if [ "$JOBS" -gt "${#LAMBDAS[@]}" ]; then JOBS="${#LAMBDAS[@]}"; fi

# ---------------------------------------------------------------------------
# Log header: enough to reproduce the run.
# ---------------------------------------------------------------------------
{
    echo "================ strategy-comparison sweep ================"
    echo "date         : $(date -Is)"
    echo "host         : $(hostname)"
    echo "git commit   : $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "python       : $(python --version 2>&1)"
    echo "instance     : ${INSTANCE}"
    echo "parts file   : ${FILE_NAME}"
    echo "dfm file     : ${DFM_FILE}"
    echo "num_phases   : ${P}"
    echo "lambda grid  : ${LAMBDAS[*]}"
    echo "k values     : ${KVALUES}"
    echo "jobs (par.)  : ${JOBS}"
    echo "rundir       : ${RUNDIR}"
    echo "=========================================================="
} | tee "$LOG"

PHASE_ARG="--num-phases ${P}"

echo "Running ${#LAMBDAS[@]} λ configs, up to ${JOBS} at a time → ${RUNDIR}/"

# -P runs JOBS at once; each task is fully independent (own digraph + own solves).
# Each task's stdout is captured to its own log so parallel output is not interleaved.
printf '%s\n' "${LAMBDAS[@]}" | xargs -P "$JOBS" -I {} bash -c '
    lam="$1"; rundir="$2"; phase_arg="$3"; module="$4"
    python -m "$module" --w-balanced "$lam" $phase_arg \
        --out "${rundir}/strategy_comparison_lambda_${lam}.csv" \
        > "${rundir}/task_lambda_${lam}.log" 2>&1
' _ {} "$RUNDIR" "$PHASE_ARG" "$MODULE"

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

# Concatenate per-λ CSVs: header from the first file, data rows from all.
COMBINED="${RUNDIR}/strategy_comparison.csv"
first=1
for f in "${RUNDIR}"/strategy_comparison_lambda_*.csv; do
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$COMBINED"
        first=0
    else
        tail -n +2 "$f" >> "$COMBINED"
    fi
done
echo "Combined CSV → ${COMBINED}" | tee -a "$LOG"

# Plot into the same folder (plotter writes next to the CSV it is given).
python -m experiments.strategy_comparison.plot_lambda_sweep "$COMBINED" | tee -a "$LOG"

echo "Done. Results + plots + log in ${RUNDIR}/"
