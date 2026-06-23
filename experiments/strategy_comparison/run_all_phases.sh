#!/usr/bin/env bash
# Run the full strategy-comparison λ sweep for a range of phase counts P, by
# calling run_sweep.sh once per P with NUM_PHASES=P. Each P writes its own
# self-contained <instance>_np_<P>/ folder (per-λ CSVs, combined CSV, run.log,
# plots).
#
# The P sweeps run SEQUENTIALLY because each run_sweep.sh already fans its λ grid
# across up to JOBS cores — running several P at once would oversubscribe. One
# failed P is reported and skipped; the rest still run, and the script exits
# non-zero if any failed.
#
# Usage (from project root):
#   ./experiments/strategy_comparison/run_all_phases.sh                 # P=2..6, default λ grid
#   ./experiments/strategy_comparison/run_all_phases.sh 0.0 0.5 1.0     # P=2..6, custom λ grid
#
# Env overrides:
#   P_MIN, P_MAX   phase-count range (default 2 .. 6)
#   JOBS           max concurrent λ processes per P (passed through to run_sweep.sh)
set -uo pipefail   # NOT -e: a failed P must not abort the remaining phase counts

cd "$(git rev-parse --show-toplevel)"
HERE="experiments/strategy_comparison"

P_MIN="${P_MIN:-2}"
P_MAX="${P_MAX:-6}"

start_all=$(date +%s)
fail=0
declare -a failed_phases=()

for P in $(seq "$P_MIN" "$P_MAX"); do
    echo
    echo "########################################################################"
    echo "### NUM_PHASES = ${P}   ($(date -Is))"
    echo "########################################################################"
    t0=$(date +%s)
    if NUM_PHASES="$P" bash "${HERE}/run_sweep.sh" "$@"; then
        echo "### P=${P} done in $(( $(date +%s) - t0 ))s"
    else
        rc=$?
        echo "### P=${P} FAILED (exit ${rc})" >&2
        failed_phases+=("$P")
        fail=1
    fi
done

echo
echo "========================================================================"
echo "All phase sweeps P=${P_MIN}..${P_MAX} finished in $(( $(date +%s) - start_all ))s"
if [ "$fail" -ne 0 ]; then
    echo "FAILED phase counts: ${failed_phases[*]}" >&2
else
    echo "All succeeded."
fi
echo "========================================================================"
exit "$fail"
