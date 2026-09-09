#!/usr/bin/env bash
# Two-process run of the base models, then the blend.
#
#   proc 1: lgbm + xgb   - share one process (one OpenMP runtime),
#                          trained sequentially, all cores each
#   proc 2: nn           - its own process: no GBDT OpenMP, so torch
#                          keeps every core (see models/nn.py)
# (CatBoost dropped from the default set - see models/__init__.py)
#
# Running more than these two in parallel just oversubscribes the 8 cores and
# gets slower. Output is line-prefixed so the interleaved logs stay readable.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

"$PY" folds.py

# process substitution so $! is the python PID (real exit code), not sed's
"$PY" run.py nn > >(sed 's/^/[nn]    /') 2>&1 &
nn_pid=$!
"$PY" run.py lgbm xgb 2>&1 | sed 's/^/[trees] /'
wait "$nn_pid"

"$PY" blend.py
