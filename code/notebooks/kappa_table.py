"""
Calibrate κ (sector adjustment) from 86,900 NFIP non-residential claims.

For each (storeys × coverage band) cell and each depth bin:
    κ_cell = median(DR_commercial) / DR_residential(depth_midpoint, τ=0)
where DR_residential is from the canonical S2 Duration-Richards fit.

Then κ_cell = N-weighted average across depth bins.

Output: params/kappa_table.csv  (Table in §3.3 of paper)
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import sys
sys.path.insert(0, os.path.join(ROOT, "code"))
from damage_functions.duration_richards import DurationRichards, DurationRichardsParams

# canonical S2 fit
params = pd.read_csv(os.path.join(ROOT, "params", "unified_params.csv"))['value'].values
m = DurationRichards(DurationRichardsParams.from_vector(params))

# load 86,900 commercial claims
df = pd.read_parquet(os.path.join(ROOT, "data", "raw", "fema_nfip_commercial_claims.parquet"))

df['payout'] = df['amountPaidOnBuildingClaim'].fillna(0) + df['amountPaidOnContentsClaim'].fillna(0)
df['coverage'] = df['totalBuildingInsuranceCoverage'].fillna(0) + df['totalContentsInsuranceCoverage'].fillna(0)
df['DR'] = np.clip(df['payout'] / df['coverage'].replace(0, np.nan), 0, 1.5)
df['depth_m'] = pd.to_numeric(df['waterDepth'], errors='coerce') * 0.0254

clean = df[
    (df['coverage'] > 1000) &
    (df['DR'] > 0) & (df['DR'] <= 1.5) &
    (df['depth_m'] > 0) & (df['depth_m'] <= 6)
].copy()
print(f'Cleaned non-residential claims: {len(clean):,}')

floors = pd.to_numeric(clean['numberOfFloorsInTheInsuredBuilding'], errors='coerce').fillna(1).clip(1, 4)
clean['floors_group'] = np.where(floors <= 1, '1-story',
                          np.where(floors <= 2, '2-story', '3+'))
clean['coverage_band'] = pd.cut(clean['coverage'], bins=[0, 50000, 200000, 500000, np.inf],
                                labels=['<50K', '50-200K', '200-500K', '>500K'])

depth_bins = [0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 6.0]
clean['depth_bin'] = pd.cut(clean['depth_m'], bins=depth_bins)

results = []
for floors_g in ['1-story', '2-story', '3+']:
    for cov_b in ['<50K', '50-200K', '200-500K', '>500K']:
        sub = clean[(clean['floors_group'] == floors_g) & (clean['coverage_band'] == cov_b)]
        if len(sub) < 20:
            continue
        kappas, weights = [], []
        for ival, grp in sub.groupby('depth_bin', observed=True):
            if len(grp) < 20:
                continue
            depth_mid = (ival.left + ival.right) / 2
            dr_resid = float(m(depth_mid, 0))
            if dr_resid < 0.05:
                continue
            dr_com = grp['DR'].median()
            kappas.append(dr_com / dr_resid)
            weights.append(len(grp))
        if not kappas:
            continue
        ka = np.array(kappas); wa = np.array(weights, dtype=float)
        k_avg = np.average(ka, weights=wa)
        k_std = np.sqrt(np.average((ka - k_avg)**2, weights=wa))
        results.append({'storeys': floors_g, 'coverage': cov_b,
                       'kappa_claims': float(np.clip(k_avg, 0.05, 3.0)),
                       'kappa_std': float(k_std), 'n_claims': int(len(sub))})

t = pd.DataFrame(results)
out = os.path.join(ROOT, 'params', 'kappa_table.csv')
t.to_csv(out, index=False)
print(f'\nκ_claims table (Section 3.3):\n')
print(t.to_string(index=False))
print(f'\nSaved -> {out}')
print(f'N total claims used: {clean.shape[0]:,}')
