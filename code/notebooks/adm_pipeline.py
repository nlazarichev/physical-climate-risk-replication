"""
ADM full credit-risk pipeline (paper §5).

Reproduces:
  Table: 12 ADM facility clusters → DR(h, τ) under S2 fit + JRC comparison
         + sector adjustment κ + insurance offset (ι=0.6) + OpEx
  Total CapEx loss, OpEx, ECL, ICR_C, ΔPD

Inputs:
  facility (h, τ, value) data from adm.tex (12 facilities in 4 clusters)
  ADM 10-K headline financials (EBIT, IE, total assets) — paper §5 narrative
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import sys
sys.path.insert(0, os.path.join(ROOT, "code"))
from damage_functions.duration_richards import DurationRichards, DurationRichardsParams

params = pd.read_csv(os.path.join(ROOT, "params", "unified_params.csv"))['value'].values
m = DurationRichards(DurationRichardsParams.from_vector(params))

# 12 facilities in 4 clusters (from adm.tex Table)
clusters = [
    ('Illinois River corridor',  4, 1200, 1.5,  72),
    ('Mississippi River (Iowa)', 3,  800, 2.0, 120),
    ('Ohio River',               2,  600, 1.0,  48),
    ('Missouri River',           3,  700, 1.8,  96),
]

# JRC simple residential curve (from paper, for comparison)
def jrc_curve(h):
    # JRC residential NA depth-damage (depth in m → DR)
    h_pts = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    dr_pts = np.array([0.00, 0.25, 0.40, 0.50, 0.57, 0.67, 0.74, 0.79, 0.82])
    return float(np.interp(h, h_pts, dr_pts))

# Sector adjustment (from §3.3 — corporate-scale extrapolation, ρ_gradient=0.25)
KAPPA_TOT = 0.21  # 1-story large industrial (paper §3.3 Table)
INSURANCE = 0.6   # paper §5 representative for Fortune 500 agri-industrial

# OpEx ratio function from paper: r_OpEx(DF) = 0.15 + 0.8 · DF^1.5
def opex_ratio(df):
    return 0.15 + 0.8 * (df ** 1.5)

print(f"\n{'Cluster':<28} {'n':>2} {'PP&E$M':>7} {'h':>4} {'tau':>4}  "
      f"{'DR_S2':>6} {'DR_JRC':>7} {'κ':>5} {'CapEx_M':>8} {'CapEx_net':>10} {'OpEx_M':>7} {'CL_M':>6}")
print('-'*120)

total_capex_gross = 0
total_capex_net = 0
total_opex = 0
total_value = 0
n_total = 0

for name, n, ppe, h, tau in clusters:
    dr_s2 = float(m(h, tau))
    dr_jrc = jrc_curve(h)
    capex_gross = dr_s2 * KAPPA_TOT * ppe
    capex_net = capex_gross * (1 - INSURANCE)
    opex = opex_ratio(dr_s2) * capex_net  # OpEx scaled by net CapEx
    cl = capex_net + opex
    print(f'{name:<28} {n:>2} {ppe:>7} {h:>4.1f} {tau:>4} '
          f'{dr_s2:>6.3f} {dr_jrc:>7.3f} {KAPPA_TOT:>5.2f} '
          f'{capex_gross:>8.1f} {capex_net:>10.1f} {opex:>7.1f} {cl:>6.1f}')
    total_capex_gross += capex_gross
    total_capex_net += capex_net
    total_opex += opex
    total_value += ppe
    n_total += n

print('-'*120)
print(f'TOTAL: {n_total} facilities, ${total_value:,.0f}M PP&E in flood zone')
print(f'  Total CapEx (gross): ${total_capex_gross:.1f}M  (paper: $334.5M)')
print(f'  Net of insurance:    ${total_capex_net:.1f}M  (paper: $133.8M)')
print(f'  Total OpEx:          ${total_opex:.1f}M  (paper: $24.1M)')
print(f'  Total ECL:           ${total_capex_net + total_opex:.1f}M  (paper: $157.9M)')

# ADM 10-K actual financials (FY2023, from paper §5.4)
EBIT_M = 3539   # ADM Operating profit FY2023
IE_M = 643      # Interest expense (Total debt $9.6B × 6.7% avg interest rate)
TOTAL_DEBT_B = 9.6
TOTAL_ASSETS_B = 53.3

# Annual ECL = sum_m p_m * E[CL]_event with seasonal flood probability profile
# Paper §5.4: peak ~8% in March-June; total annualisation factor ≈ 0.18
ANNUAL_PROB_FACTOR = 0.18
ECL_event_M = total_capex_net + total_opex
ECL_annual_M = ANNUAL_PROB_FACTOR * ECL_event_M

ICR_baseline = EBIT_M / IE_M
ICR_climate = (EBIT_M - ECL_annual_M) / IE_M

print(f'\nADM credit-risk translation (FY2023 from 10-K):')
print(f'  EBIT:                       ${EBIT_M:,}M')
print(f'  Interest expense:            ${IE_M:,}M  (= ${TOTAL_DEBT_B}B × 6.7%)')
print(f'  ECL per event:               ${ECL_event_M:.1f}M  (paper: $157.9M)')
print(f'  Annual ECL (factor {ANNUAL_PROB_FACTOR:.2f}):     ${ECL_annual_M:.1f}M  (paper: $28.4M)')
print(f'  ICR baseline:                {ICR_baseline:.2f}  (paper: 5.50)')
print(f'  ICR climate-stressed:        {ICR_climate:.2f}  (paper: 5.46)')

# Damodaran-style spread mapping (simplified from §3.4)
def icr_to_default_spread_pct(icr):
    # Approximate Damodaran ICR-to-spread (large firms)
    pts = np.array([-100, 0.2, 0.5, 0.8, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.25, 5.5, 6.5, 8.5, 100])
    spread = np.array([14.32, 11.54, 9.78, 7.78, 5.94, 4.86, 4.12, 3.61, 3.08, 2.36, 2.00, 1.62, 1.42, 1.21, 0.85, 0.59])
    return float(np.interp(icr, pts, spread))

RR = 0.4
spread_baseline = icr_to_default_spread_pct(ICR_baseline) / 100.0
spread_climate = icr_to_default_spread_pct(ICR_climate) / 100.0
PD_baseline = spread_baseline / (1 - RR)
PD_climate = spread_climate / (1 - RR)
delta_PD = (PD_climate - PD_baseline) * 1e4  # in basis points

print(f'  PD baseline:            {PD_baseline*100:.3f}%')
print(f'  PD climate-stressed:    {PD_climate*100:.3f}%')
print(f'  ΔPD:                    {delta_PD:.1f} bps  (paper: ~80 bps reported)')

# save
out = os.path.join(ROOT, "params", "adm_results.csv")
result = pd.DataFrame([
    {'cluster': c[0], 'n_facilities': c[1], 'ppe_M': c[2], 'depth_m': c[3], 'duration_h': c[4],
     'DR_S2': float(m(c[3], c[4])), 'DR_JRC': jrc_curve(c[3]),
     'kappa_tot': KAPPA_TOT, 'capex_gross_M': float(m(c[3], c[4])) * KAPPA_TOT * c[2]}
    for c in clusters
])
result.to_csv(out, index=False)
print(f'\nSaved -> {out}')
