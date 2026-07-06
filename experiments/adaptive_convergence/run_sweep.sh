#!/usr/bin/env bash
# Fan the oracle-free adaptive-convergence experiment across cores: one process
# per w_balanced (λ), each building its own digraph and running the adaptive
# k-sweep. No full MIP anywhere — so unlike bl_union_convergence the λ grid can
# span 0..1 even on assembly_2 (that instance's full MIP does not converge > 0.85,
# which is exactly why this oracle-free experiment exists).
#
# Everything lands in one self-contained folder
#   experiments/adaptive_convergence/<instance>_np_<P>/
# holding: per-λ CSVs, a combined CSV, the convergence plots (obj-vs-ideal,
# path-vs-c_min, α-vs-phase-width, overlaid across λ), and a run.log.
#
# Usage (from project root):
#   ./experiments/adaptive_convergence/run_sweep.sh                 # default λ grid
#   ./experiments/adaptive_convergence/run_sweep.sh 0.0 0.5 0.9 1.0  # custom λ grid
#
# Env knobs: JOBS, NUM_PHASES, STOP_PERC_GRAPH, K_MAX, RUN_TAG.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODULE="experiments.adaptive_convergence.adaptive_convergence"

if [ "$#" -gt 0 ]; then
    LAMBDAS=("$@")
else
    LAMBDAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
fi

# Pull config from the module so the folder name / log stay a single source of
# truth (instance, default P, data files).
read -r INSTANCE DEFP FILE_NAME DFM_FILE < <(python -c \
    "import ${MODULE} as t; print(t.INSTANCE, t.NUM_PHASES, t.FILE_NAME, t.DFM_FILE_NAME or '(none)')")
P="${NUM_PHASES:-$DEFP}"

RUNDIR="experiments/adaptive_convergence/${INSTANCE}_np_${P}${RUN_TAG:+_${RUN_TAG}}"
# Start clean; keep rm -rf pinned to this experiment tree.
case "$RUNDIR" in experiments/adaptive_convergence/*) rm -rf "$RUNDIR" ;; esac
mkdir -p "$RUNDIR"
LOG="${RUNDIR}/run.log"

JOBS="${JOBS:-$(nproc)}"
if [ "$JOBS" -gt "${#LAMBDAS[@]}" ]; then JOBS="${#LAMBDAS[@]}"; fi

{
    echo "============ adaptive convergence (oracle-free) sweep ============"
    echo "date         : $(date -Is)"
    echo "host         : $(hostname)"
    echo "git commit   : $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "python       : $(python --version 2>&1)"
    echo "instance     : ${INSTANCE}"
    echo "parts file   : ${FILE_NAME}"
    echo "dfm file     : ${DFM_FILE}"
    echo "num_phases   : ${P}"
    echo "lambda grid  : ${LAMBDAS[*]}"
    echo "jobs (par.)  : ${JOBS}"
    echo "rundir       : ${RUNDIR}"
    echo "================================================================="
} | tee "$LOG"

# Optional pass-through knobs (env).
EXTRA_ARGS="--num-phases ${P}"
[ -n "${STOP_PERC_GRAPH:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --stop-perc-graph ${STOP_PERC_GRAPH}"
[ -n "${K_MAX:-}" ]           && EXTRA_ARGS="${EXTRA_ARGS} --k-max ${K_MAX}"
echo "extra args   : ${EXTRA_ARGS}" | tee -a "$LOG"

echo "Running ${#LAMBDAS[@]} λ configs, up to ${JOBS} at a time -> ${RUNDIR}/"

# -P runs JOBS at once; each task is fully independent (own digraph + own solves)
# and, being oracle-free, fully deterministic. Per-task stdout goes to its own log.
printf '%s\n' "${LAMBDAS[@]}" | xargs -P "$JOBS" -I {} bash -c '
    lam="$1"; rundir="$2"; extra="$3"; module="$4"
    python -m "$module" --w-balanced "$lam" $extra \
        --out "${rundir}/adaptive_convergence_lambda_${lam}.csv" \
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

# Concatenate per-λ CSVs: header from the first, data rows from all.
COMBINED="${RUNDIR}/adaptive_convergence.csv"
first=1
for f in "${RUNDIR}"/adaptive_convergence_lambda_*.csv; do
    [ -f "$f" ] || continue
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$COMBINED"; first=0
    else
        tail -n +2 "$f" >> "$COMBINED"
    fi
done
echo "Combined CSV → ${COMBINED}" | tee -a "$LOG"

# Plot into the same folder (overlays all λ on each of the three figures).
python -m experiments.adaptive_convergence.plot_convergence "$COMBINED" | tee -a "$LOG"

echo "Done. Results + plots + log in ${RUNDIR}/"
