"""
Section 5.7 — Insurance coverage sensitivity.

Reproduces paper Table 12: ι ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
→ CapEx_net, OpEx, ECL, ICR_C ranges for ADM 12 facilities.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from damage_functions.duration_richards import DurationRichards, DurationRichardsParams

params = pd.read_csv(os.path.join(ROOT, "params", "unified_params.csv"))['value'].values
m = DurationRichards(DurationRichardsParams.from_vector(params))

KAPPA_TOT = 0.21
EBIT_M = 3539
IE_M = 643
ANNUAL_PROB_FACTOR = 0.18

clusters = [
    ('Illinois River',  4, 1200, 1.5,  72),
    ('Mississippi River', 3, 800, 2.0, 120),
    ('Ohio River',      2,  600, 1.0,  48),
    ('Missouri River',  3,  700, 1.8,  96),
]

# Compute total CapEx_gross under the canonical τ
total_capex_gross = sum(float(m(h, tau)) * KAPPA_TOT * ppe for _, _, ppe, h, tau in clusters)

# OpEx ratio at sector-level DF (weighted average of cluster DRs by CapEx contribution)
# Paper §5.3: opex_ratio = 0.15 + 0.8 × DF_sector^1.5 with DF_sector = CapEx_total / Asset_total = 0.111
DF_sector = total_capex_gross / sum(c[2] for c in clusters)
opex_ratio = 0.15 + 0.8 * (DF_sector ** 1.5)

print(f"DF_sector = total CapEx_gross / total PP&E = "
      f"{total_capex_gross:.1f} / {sum(c[2] for c in clusters)} = {DF_sector:.4f}")
print(f"OpEx ratio = 0.15 + 0.8 × {DF_sector:.4f}^1.5 = {opex_ratio:.4f}")

print(f"\n{'ι':>4}  {'CapEx_net($M)':>14} {'r_OpEx':>7} {'OpEx($M)':>9} "
      f"{'E[CL]_event':>12} {'E[CL]_annual':>13} {'ICR_C':>6}")
print('-' * 75)

results = []
for iota in [0.4, 0.5, 0.6, 0.7, 0.8]:
    capex_net = total_capex_gross * (1 - iota)
    opex = capex_net * opex_ratio
    ecl_event = capex_net + opex
    ecl_annual = ANNUAL_PROB_FACTOR * ecl_event
    icr_c = (EBIT_M - ecl_annual) / IE_M
    bold = " *" if iota == 0.6 else "  "
    print(f"{iota:>4.1f}{bold}{capex_net:>12.1f}   {opex_ratio:>7.4f} {opex:>9.1f} "
          f"{ecl_event:>12.1f} {ecl_annual:>13.1f} {icr_c:>6.2f}")
    results.append({'iota': iota, 'capex_net_M': capex_net, 'opex_M': opex,
                   'ecl_event_M': ecl_event, 'ecl_annual_M': ecl_annual, 'icr_c': icr_c})

print('-' * 75)
print("* = baseline (paper Table 12 highlighted row)")
print()
print(f"Paper Table 12 (baseline ι=0.6): CapEx_net=133.8M, OpEx=24.1M, E[CL]=157.9M, ICR_C=5.46")
print(f"Ours (baseline ι=0.6):           "
      f"CapEx_net={results[2]['capex_net_M']:.1f}M, OpEx={results[2]['opex_M']:.1f}M, "
      f"E[CL]={results[2]['ecl_event_M']:.1f}M, ICR_C={results[2]['icr_c']:.2f}")

out = os.path.join(ROOT, "params", "insurance_sensitivity.csv")
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nSaved -> {out}")
