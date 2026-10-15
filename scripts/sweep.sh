#!/usr/bin/env bash
#
# Data-percentage sweep: run the search (and optionally the test) once per
# training budget, matching the paper's experiments.
#
#   ./scripts/sweep.sh schisto                    # all six percentages
#   ./scripts/sweep.sh larvae 025 100             # just these two
#   ./scripts/sweep.sh schisto --gpu 1            # pick a GPU
#   ./scripts/sweep.sh schisto --test             # also test after each search
#   ./scripts/sweep.sh schisto --prefix myrun     # change the exp-name prefix
#
# Each percentage becomes its own experiment, named <prefix>_<pct>pct, so runs
# never overwrite each other. The config's exp_name supplies the default prefix.
#
# Each percentage runs as a separate `autoflim search` invocation so the OS
# reclaims GPU and host memory between budgets; a single long-lived process
# accumulates allocator fragmentation across a sweep this long.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DATASET="${1:-}"
if [ -z "$DATASET" ] || [ "$DATASET" = "-h" ] || [ "$DATASET" = "--help" ]; then
    # Print the comment block at the top of this file as the usage text.
    sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi
shift

CONFIG="configs/${DATASET}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: no config at $CONFIG" >&2
    exit 1
fi

PERCENTAGES=()
GPU=""
PREFIX=""
RUN_TEST=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)    GPU="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --test)   RUN_TEST=1; shift ;;
        --)       shift; EXTRA+=("$@"); break ;;
        -*)       EXTRA+=("$1"); shift ;;
        *)        PERCENTAGES+=("$1"); shift ;;
    esac
done

# Default to the paper's six budgets.
if [ ${#PERCENTAGES[@]} -eq 0 ]; then
    PERCENTAGES=(001 005 025 050 075 100)
fi

# Default the experiment prefix to the config's own exp_name.
if [ -z "$PREFIX" ]; then
    PREFIX="$(python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$CONFIG')) or {}
print((cfg.get('output') or {}).get('exp_name', 'exp'))
")"
fi

GPU_ARG=()
[ -n "$GPU" ] && GPU_ARG=(--gpu-ids "$GPU")

echo "=========================================================="
echo "Sweep      : $DATASET"
echo "Config     : $CONFIG"
echo "Percentages: ${PERCENTAGES[*]}"
echo "Exp prefix : $PREFIX"
[ -n "$GPU" ] && echo "GPU        : $GPU"
[ $RUN_TEST -eq 1 ] && echo "Testing    : after each search"
echo "=========================================================="
echo

FAILED=()

for PCT in "${PERCENTAGES[@]}"; do
    EXP="${PREFIX}_${PCT}pct"

    echo "=========================================================="
    echo "$DATASET | pct=$PCT | exp=$EXP"
    echo "=========================================================="

    if autoflim search -c "$CONFIG" \
            --train-percentage "$PCT" \
            --exp-name "$EXP" \
            "${GPU_ARG[@]}" "${EXTRA[@]}"; then
        echo "search pct=$PCT done."
    else
        echo "ERROR: search failed for pct=$PCT — continuing with the rest." >&2
        FAILED+=("search:$PCT")
        continue
    fi

    if [ $RUN_TEST -eq 1 ]; then
        if autoflim test -c "$CONFIG" \
                --train-percentage "$PCT" \
                --exp-name "$EXP" \
                "${GPU_ARG[@]}"; then
            echo "test pct=$PCT done."
        else
            echo "ERROR: test failed for pct=$PCT." >&2
            FAILED+=("test:$PCT")
        fi
    fi
    echo
done

echo "=========================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "Sweep complete: ${#PERCENTAGES[@]} percentage(s), no failures."
else
    echo "Sweep finished with ${#FAILED[@]} failure(s): ${FAILED[*]}"
    echo "Re-run a single budget with:"
    echo "  autoflim search -c $CONFIG --train-percentage <pct> --exp-name ${PREFIX}_<pct>pct --resume"
fi
echo "=========================================================="

[ ${#FAILED[@]} -eq 0 ] || exit 1
