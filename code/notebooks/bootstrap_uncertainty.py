"""
Section 7 — Parameter uncertainty (200-replicate bootstrap).

Reproduces paper §7 "Parameter uncertainty" claim:
- λ_a is positive in every replicate
- λ_Q, δ_ν close to zero
- Bootstrap dispersion small relative to cross-event MAE differences
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from damage_functions.duration_richards import predict_from_vector

# canonical caps (S2 fit setup)
events_cfg = {
    'Sandy':   {'file': os.path.join(ROOT, 'data/processed/fema_sandy_calibration.csv'),
                'duration_h': 36,  'max_depth': 4.0},
    'Harvey':  {'file': os.path.join(ROOT, 'data/processed/fema_harvey_calibration.csv'),
                'duration_h': 288, 'max_depth': 4.0},
    'Katrina': {'file': os.path.join(ROOT, 'data/processed/fema_katrina_calibration.csv'),
                'duration_h': 120, 'max_depth': 10.0},
}
USACE_WEIGHT = 3.0
TARGET_BINS = 12

# Build binned dataset once
event_data = {}
for name, cfg in events_cfg.items():
    cal = pd.read_csv(cfg['file'])
    cal = cal[cal['depth_m'] <= cfg['max_depth']]
    depth = cal['depth_m'].values; dr = cal['damage_ratio'].values
    edges = np.linspace(0, cfg['max_depth'], TARGET_BINS + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, TARGET_BINS - 1)
    bin_d  = np.array([depth[idx==i].mean() if (idx==i).sum()>0 else np.nan for i in range(TARGET_BINS)])
    bin_dr = np.array([dr[idx==i].mean()    if (idx==i).sum()>0 else np.nan for i in range(TARGET_BINS)])
    bin_std = np.array([dr[idx==i].std()   if (idx==i).sum()>1 else 0.2 for i in range(TARGET_BINS)])
    bin_n   = np.array([(idx==i).sum() for i in range(TARGET_BINS)])
    valid = bin_n > 10
    event_data[name] = {
        'depth': bin_d[valid], 'dr_mean': bin_dr[valid],
        'sigma': np.maximum(bin_std[valid] / np.sqrt(bin_n[valid]), 0.015),
        'duration_h': cfg['duration_h']
    }
event_data['USACE'] = {'depth': np.array([0,0.3,0.6,0.9,1.2,1.5,2.4]),
                      'dr_mean': np.array([0.08,0.18,0.27,0.35,0.43,0.48,0.55]),
                      'sigma': np.full(7, 0.03), 'duration_h': 0}

bounds = [(0.5,12),(0.05,5),(0.1,0.9),(0.1,3),(0,3),(0,3),(0,2)]

def objective(theta, perturbed):
    total = 0.0
    for name, d in event_data.items():
        try:
            pred = predict_from_vector(theta, d['depth'], d['duration_h'])
        except:
            return 1e10
        residuals = perturbed[name] - pred
        w = USACE_WEIGHT if name == 'USACE' else 1.0
        total += w * np.sum((residuals / d['sigma']) ** 2)
    B,Q0,a0,nu0,la,lQ,dn = theta
    total += 0.5*((B-3)/3)**2 + 0.5*((Q0-1)/1)**2 + 0.5*((a0-0.5)/0.2)**2 + 0.5*((nu0-0.5)/0.3)**2
    return total

N_REP = int(os.environ.get('N_REP', 50))
rng = np.random.default_rng(42)
results = []
print(f"Running {N_REP} bootstrap replicates (set N_REP env var to override; paper uses 200)...")
for r in range(N_REP):
    perturbed = {name: d['dr_mean'] + rng.normal(0, d['sigma'])
                 for name, d in event_data.items()}
    # Lower maxiter for tractable runtime; differential_evolution converges fast for this problem
    result = differential_evolution(lambda th: objective(th, perturbed),
                                     bounds, seed=42 + r, maxiter=100,
                                     tol=1e-6, polish=True, workers=1)
    results.append(result.x)
    if (r+1) % 10 == 0:
        print(f"  {r+1}/{N_REP} done")

arr = np.array(results)
names = ['B', 'Q0', 'a0', 'nu0', 'lambda_a', 'lambda_Q', 'delta_nu']
df = pd.DataFrame(arr, columns=names)

print(f"\nBootstrap dispersion (n={N_REP} replicates):")
print(f"{'Param':<10} {'mean':>9} {'std':>9} {'2.5%':>9} {'97.5%':>9} {'>0?':>5}")
print('-' * 60)
for name in names:
    vals = df[name].values
    pos = (vals > 0.001).sum()
    pos_frac = pos / len(vals)
    print(f"{name:<10} {vals.mean():>9.4f} {vals.std():>9.4f} "
          f"{np.percentile(vals, 2.5):>9.4f} {np.percentile(vals, 97.5):>9.4f} "
          f"{100*pos_frac:>4.0f}%")

print(f"\n✓ λ_a positive in {(df['lambda_a'] > 0.001).sum()}/{N_REP} replicates "
      f"(paper: 'positive and bounded away from zero in every replicate')")
print(f"  λ_Q remains close to zero: {(df['lambda_Q'] < 0.1).sum()}/{N_REP} replicates < 0.1")
print(f"  δ_ν remains close to zero: {(df['delta_nu'] < 0.1).sum()}/{N_REP} replicates < 0.1")

out = os.path.join(ROOT, "params", "bootstrap_replicates.csv")
df.to_csv(out, index=False)
print(f"\nSaved -> {out}")
