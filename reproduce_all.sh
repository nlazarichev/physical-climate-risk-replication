#!/usr/bin/env bash
# Reproduce all paper Tables and Figures end-to-end.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/9]  Joint calibration (Tables 3+4 + Fig 6)"
PYTHONPATH=code python3 code/notebooks/unified_calibration.py | tail -10
echo "[2/9]  Duration curves (Fig 4)"
PYTHONPATH=code python3 code/notebooks/generate_duration_curves.py
echo "[3/9]  Validation tests 1+2+3 (Table 5 + Fig 11)"
PYTHONPATH=code python3 code/notebooks/backtest_validation.py | tail -8
echo "[4/9]  JRC benchmark (Table 6 + Fig 13)"
PYTHONPATH=code python3 code/notebooks/climada_comparison.py | tail -5
echo "[5/9]  Spearman vs IHP (Table 7 + Fig 14)"
PYTHONPATH=code python3 code/notebooks/defaults_backtest.py | tail -5
echo "[6/9]  Feature enrichment (Table 8 + Fig 12)"
PYTHONPATH=code python3 code/notebooks/feature_enrichment.py | tail -10
echo "[7/9]  Sector adjustment kappa (Sec 3.3)"
PYTHONPATH=code python3 code/notebooks/kappa_table.py | tail -5
echo "[8/9]  ADM pipeline (Tables 9+10) + tau-sens (Table 11) + insurance-sens (Table 12)"
PYTHONPATH=code python3 code/notebooks/adm_pipeline.py | tail -8
PYTHONPATH=code python3 code/notebooks/tau_sensitivity.py | tail -5
PYTHONPATH=code python3 code/notebooks/insurance_sensitivity.py | tail -5
echo "[9/9]  24-company OOS (Table 13) + geographic transferability (Sec 6.4)"
PYTHONPATH=code python3 code/notebooks/oos_24_companies.py | tail -10
PYTHONPATH=code python3 code/notebooks/geographic_transferability.py | tail -5

echo
echo "All paper tables and figures regenerated."
echo "Summary in REPLICATION_RESULTS.md; provenance in params/PROVENANCE.md."
