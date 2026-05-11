"""
Section 5.5 — Sensitivity to τ assignment.

Reproduces paper Table 11: ±50% τ → DR shifts per ADM facility cluster,
total CapEx $280M (0.5τ) → $338M (τ) → $378M (1.5τ).
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
clusters = [
    ('Illinois River',  4, 1200, 1.5,  72),
    ('Mississippi River', 3, 800, 2.0, 120),
    ('Ohio River',      2,  600, 1.0,  48),
    ('Missouri River',  3,  700, 1.8,  96),
]

print(f"{'Cluster':<22} {'h':>4} {'τ':>4}  "
      f"{'DR(0.5τ)':>9} {'DR(τ)':>7} {'DR(1.5τ)':>9}  "
      f"{'CapEx(0.5τ)':>11} {'CapEx(τ)':>9} {'CapEx(1.5τ)':>12}")
print('-' * 105)

totals = [0, 0, 0]
for name, n, ppe, h, tau in clusters:
    row = [name, h, tau]
    drs, capxs = [], []
    for f in [0.5, 1.0, 1.5]:
        dr = float(m(h, tau * f))
        cap = dr * KAPPA_TOT * ppe
        drs.append(dr)
        capxs.append(cap)
    print(f"{name:<22} {h:>4.1f} {tau:>4} "
          f" {drs[0]:>9.3f} {drs[1]:>7.3f} {drs[2]:>9.3f}  "
          f"{capxs[0]:>10.0f}M {capxs[1]:>8.0f}M {capxs[2]:>11.0f}M")
    for i, c in enumerate(capxs):
        totals[i] += c

print('-' * 105)
print(f"{'TOTAL gross CapEx':<22} {'':>4} {'':>4}                            "
      f"{totals[0]:>10.0f}M {totals[1]:>8.0f}M {totals[2]:>11.0f}M")
print(f"\nPaper Table 11: total spans 280M (0.5τ) → 338M (τ) → 378M (1.5τ); -17% to +12% around baseline")
print(f"Ours:           total spans {totals[0]:.0f}M → {totals[1]:.0f}M → {totals[2]:.0f}M; "
      f"{100*(totals[0]/totals[1]-1):+.0f}% to {100*(totals[2]/totals[1]-1):+.0f}%")

out = os.path.join(ROOT, "params", "tau_sensitivity.csv")
pd.DataFrame({
    'cluster': [c[0] for c in clusters] + ['TOTAL'],
    'h_m': [c[3] for c in clusters] + [None],
    'tau_h': [c[4] for c in clusters] + [None],
    'capex_0p5tau_M': [float(m(c[3], c[4]*0.5)) * KAPPA_TOT * c[2] for c in clusters] + [totals[0]],
    'capex_1p0tau_M': [float(m(c[3], c[4]*1.0)) * KAPPA_TOT * c[2] for c in clusters] + [totals[1]],
    'capex_1p5tau_M': [float(m(c[3], c[4]*1.5)) * KAPPA_TOT * c[2] for c in clusters] + [totals[2]],
}).to_csv(out, index=False)
print(f"\nSaved -> {out}")
