#!/usr/bin/env bash
# Fan the bl-union convergence experiment across cores: one process per
# w_balanced (λ), each building its own digraph, solving its own full-MIP
#
# Everything for one run lands in a single self-contained folder
#   experiments/bl_union_convergence/<instance>_np_<P>/
# holding: the per-λ CSVs, the combined CSV, a run.log with the full experiment
# details (+ each task's stdout), and the plots.
#
# Usage (from project root):
#   ./experiments/bl_union_convergence/run_sweep.sh                  # default λ grid
#   ./experiments/bl_union_convergence/run_sweep.sh 0.0 0.25 0.5 0.85 # custom λ grid
#
# NOTE: the default λ grid stops at 0.85 because the assembly_2 full MIP does
# not converge for λ > 0.85 (the reference solve would hang). Override the grid
# explicitly if you target an instance where higher λ converges.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODULE="experiments.bl_union_convergence.bl_union_convergence"

if [ "$#" -gt 0 ]; then
    LAMBDAS=("$@")
else
    LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

# Pull config from the experiment module so the folder name and log stay a
# single source of truth (instance, default P, k grid, data files).
read -r INSTANCE DEFP < <(python -c \
    "import ${MODULE} as t; print(t.INSTANCE, t.NUM_PHASES)")
P="${NUM_PHASES:-$DEFP}"
KVALUES=$(python -c "import ${MODULE} as t; print(t.K_VALUES)")
BLENDS=$(python -c "import ${MODULE} as t; print(t.BLEND_GRID)")
FILE_NAME=$(python -c "import ${MODULE} as t; print(t.FILE_NAME)")
DFM_FILE=$(python -c "import ${MODULE} as t; print(t.DFM_FILE_NAME)")

# RUN_TAG (env) appends a suffix so a variant run (e.g. diverse-only, high
# penalty) lands in its own folder instead of overwriting the main comparison.
RUNDIR="experiments/bl_union_convergence/${INSTANCE}_np_${P}${RUN_TAG:+_${RUN_TAG}}"
# Start clean: a reused folder can mix stale per-λ CSVs / task logs from a
# previous (possibly killed) run into the combined CSV and run.log. The guard
# keeps rm -rf pinned to the experiment tree. (The full-MIP cache lives in a
# separate CACHE_DIR and is untouched.)
case "$RUNDIR" in experiments/bl_union_convergence/*) rm -rf "$RUNDIR" ;; esac
mkdir -p "$RUNDIR"
LOG="${RUNDIR}/run.log"

JOBS="${JOBS:-$(nproc)}"
if [ "$JOBS" -gt "${#LAMBDAS[@]}" ]; then JOBS="${#LAMBDAS[@]}"; fi

# ---------------------------------------------------------------------------
# Log header: enough to reproduce the run.
# ---------------------------------------------------------------------------
{
    echo "================ bl-union convergence sweep ================"
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
    echo "blend grid   : ${BLENDS}"
    echo "jobs (par.)  : ${JOBS}"
    echo "rundir       : ${RUNDIR}"
    echo "==========================================================="
} | tee "$LOG"

# Optional pass-through knobs (env): STOP, K_MAX, GAP_TARGET, METHODS, PENALTY,
# REFRESH_CACHE.
EXTRA_ARGS="--num-phases ${P}"
[ -n "${STOP:-}" ]        && EXTRA_ARGS="${EXTRA_ARGS} --stop ${STOP}"
[ -n "${K_MAX:-}" ]       && EXTRA_ARGS="${EXTRA_ARGS} --k-max ${K_MAX}"
[ -n "${GAP_TARGET:-}" ]  && EXTRA_ARGS="${EXTRA_ARGS} --gap-target ${GAP_TARGET}"
[ -n "${METHODS:-}" ]     && EXTRA_ARGS="${EXTRA_ARGS} --methods ${METHODS}"
[ -n "${PENALTY:-}" ]     && EXTRA_ARGS="${EXTRA_ARGS} --penalty ${PENALTY}"
[ -n "${REFRESH_CACHE:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --refresh-cache"
[ -n "${NO_CACHE:-}" ]      && EXTRA_ARGS="${EXTRA_ARGS} --no-cache"
echo "extra args   : ${EXTRA_ARGS}" | tee -a "$LOG"

echo "Running ${#LAMBDAS[@]} λ configs, up to ${JOBS} at a time → ${RUNDIR}/"

# -P runs JOBS at once; each task is fully independent (own digraph + own solves).
# Each task's stdout is captured to its own log so parallel output is not interleaved.
printf '%s\n' "${LAMBDAS[@]}" | xargs -P "$JOBS" -I {} bash -c '
    lam="$1"; rundir="$2"; extra="$3"; module="$4"
    python -m "$module" --w-balanced "$lam" $extra \
        --out "${rundir}/bl_union_convergence_lambda_${lam}.csv" \
        > "${rundir}/task_lambda_${lam}.log" 2>&1
' _ {} "$RUNDIR" "$EXTRA_ARGS" "$MODULE"

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
COMBINED="${RUNDIR}/bl_union_convergence.csv"
first=1
for f in "${RUNDIR}"/bl_union_convergence_lambda_*.csv; do
    [ -f "$f" ] || continue
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$COMBINED"
        first=0
    else
        tail -n +2 "$f" >> "$COMBINED"
    fi
done
echo "Combined CSV → ${COMBINED}" | tee -a "$LOG"

# Plot into the same folder (plotters write next to the CSV they are given).
python -m experiments.bl_union_convergence.plot_convergence "$COMBINED" | tee -a "$LOG"
python -m experiments.bl_union_convergence.plot_metrics_grid "$COMBINED" | tee -a "$LOG"

echo "Done. Results + plots + log in ${RUNDIR}/"
