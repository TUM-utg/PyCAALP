#!/usr/bin/env bash
# Fan the strategy-comparison sweep across cores: one process per w_balanced (λ),
# each building its own digraph and solving its own MIPs, writing a per-task CSV.
# Concatenate the per-task CSVs into one combined file at the end.
#
# Usage (from project root):
#   ./experiments/strategy_comparison/run_sweep.sh                  # default λ grid
#   ./experiments/strategy_comparison/run_sweep.sh 0.0 0.25 0.5 1.0 # custom λ grid
#
# Env overrides:
#   JOBS       max concurrent processes (default: nproc, capped at #λ)
#   NUM_PHASES phases passed to every task (default: script default)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUTDIR="experiments/strategy_comparison/sweep"
mkdir -p "$OUTDIR"

# λ grid: CLI args, or a default spread across [0, 1].
if [ "$#" -gt 0 ]; then
    LAMBDAS=("$@")
else
    LAMBDAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
fi

JOBS="${JOBS:-$(nproc)}"
if [ "$JOBS" -gt "${#LAMBDAS[@]}" ]; then JOBS="${#LAMBDAS[@]}"; fi
PHASE_ARG=""
if [ -n "${NUM_PHASES:-}" ]; then PHASE_ARG="--num-phases ${NUM_PHASES}"; fi

echo "Running ${#LAMBDAS[@]} λ configs, up to ${JOBS} at a time → ${OUTDIR}/"

# -P runs JOBS at once; each task is fully independent (own digraph + own solves).
printf '%s\n' "${LAMBDAS[@]}" | xargs -P "$JOBS" -I {} \
    python -m experiments.strategy_comparison.test_strategy_comparison \
        --w-balanced {} ${PHASE_ARG} \
        --out "${OUTDIR}/strategy_comparison_lambda_{}.csv"

# Concatenate: header from the first file, data rows from all.
COMBINED="experiments/strategy_comparison/strategy_comparison.csv"
first=1
for f in "${OUTDIR}"/strategy_comparison_lambda_*.csv; do
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$COMBINED"
        first=0
    else
        tail -n +2 "$f" >> "$COMBINED"
    fi
done
echo "Combined CSV → ${COMBINED}"
