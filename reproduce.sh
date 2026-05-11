#!/usr/bin/env bash
# One-click reproduction. Runs in <1 minute on the frozen data/processed/ files.
# To re-pull from FEMA OpenFEMA, set REFETCH=1 (10–30 min, snapshot-dependent).
set -euo pipefail

cd "$(dirname "$0")"

echo "[reproduce] Python: $(python3 --version)"

if [ "${REFETCH:-0}" = "1" ]; then
    echo "[reproduce] REFETCH=1 — re-pulling raw claims from OpenFEMA…"
    rm -f data/raw/fema_sandy_claims.csv data/raw/fema_katrina_claims.csv data/raw/fema_harvey_claims.csv
    rm -f data/processed/*.csv
    PYTHONPATH=code python3 code/fetch_fema.py
fi

echo "[reproduce] Running unified joint calibration…"
PYTHONPATH=code python3 code/notebooks/unified_calibration.py

echo
echo "[reproduce] Done."
echo "  Parameters → params/unified_params.csv"
echo "  Figure     → figures/10_unified_calibration.png"
echo
echo "Compare against:"
echo "  params/unified_params_canonical.csv  (this archive's frozen canonical fit)"
echo "  params/PROVENANCE.md                 (full audit trail)"
