"""
24-company OOS validation — paper §6 / Table-OOS.

NOTE: Direct end-to-end reproduction of the predicted ECL from facility-level
inputs (depth h, value V, insurance ι) is NOT possible from this archive alone:
those inputs come from 10-K filings + per-company flood-zone analysis that the
paper does not redistribute.

What we DO verify:
  1. The 24-company validation table extracted from harvey_event_study.tex
     is internally consistent — paper's predicted ECL ratio statistics match
     paper's claimed summary (median 0.90×, 23/24 within 0.4-2.5×).
  2. The κ_tot lookup values used in the table are produced by our
     calibrated κ table (params/kappa_table.csv) × ρ_gradient = 0.25.

Reproducing the predicted ECL itself requires the per-company facility input
table (depth_m, value_M, insurance) which is referenced but not bundled in
the upstream repo. That table is the natural Phase 2 deliverable for full
end-to-end OOS validation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

df = pd.read_csv(os.path.join(ROOT, "data", "processed", "oos_24_companies.csv"))
df['ratio'] = df['ECL_paper_M'] / df['actual_loss_M']
df['log_ratio'] = np.log(df['ratio'])

print(f"\n{'Company':<25} {'Event':<10} {'St':>2} {'kappa':>5} {'ECL$M':>7} {'Act$M':>6} {'Ratio':>6}")
print('-'*72)
for _, r in df.iterrows():
    print(f"{r['company']:<25} {r['event']:<10} {int(r['storeys']):>2} {r['kappa_tot']:>5.2f} "
          f"{r['ECL_paper_M']:>7.0f} {r['actual_loss_M']:>6.0f} {r['ratio']:>5.2f}x")
print('-'*72)

n = len(df)
median_r = df['ratio'].median()
mean_abs_log = df['log_ratio'].abs().mean()
within_loose = ((df['ratio'] >= 0.4) & (df['ratio'] <= 2.5)).sum()
within_tight = ((df['ratio'] >= 0.5) & (df['ratio'] <= 2.0)).sum()
print(f'\nN = {n}')
print(f'Median predicted/actual: {median_r:.2f}x  (paper claims 0.90x)')
print(f'Mean |log(ratio)|:       {mean_abs_log:.3f}  (paper claims 0.37)')
print(f'Within 0.4-2.5x:         {within_loose}/{n} ({100*within_loose/n:.0f}%)  (paper: 23/24=96%)')
print(f'Within 0.5-2.0x:         {within_tight}/{n} ({100*within_tight/n:.0f}%)  (paper: 21/24=88%)')

# κ check vs our table
print(f'\nκ_tot lookup match check (κ_total = κ_claims × ρ_gradient with ρ_gradient=0.25):')
kt_path = os.path.join(ROOT, 'params', 'kappa_table.csv')
if os.path.exists(kt_path):
    kt = pd.read_csv(kt_path)
    rho = 0.25
    # paper's kappa_tot per storeys (>$500K coverage band, the corporate-scale extrapolation)
    paper_st = {'1-story': 0.21, '2-story': 0.16, '3+': 0.14}
    print(f'\n{"Storeys":<10} {"κ_claims (ours)":>16} {"× ρ_grad (0.25)":>16} {"= κ_tot":>10} {"paper κ_tot":>12}')
    for sg, paper_kt in paper_st.items():
        match = kt[(kt['storeys'] == sg) & (kt['coverage'] == '>500K')]
        if not match.empty:
            kc = match.iloc[0]['kappa_claims']
            our_kt = kc * rho
            print(f'{sg:<10} {kc:>16.3f} {kc*rho:>16.3f} {our_kt:>10.3f} {paper_kt:>12.2f}')

out = os.path.join(ROOT, 'params', 'oos_results.csv')
df.to_csv(out, index=False)
print(f'\nSaved -> {out}')

# plot
fig, ax = plt.subplots(figsize=(9, 6))
colors = {'Harvey': '#d62728', 'Thailand': '#1f77b4', 'KZN': '#2ca02c',
          'Helene': '#9467bd', 'Michigan': '#8c564b', 'Sandy': '#e377c2',
          'Florence': '#bcbd22', 'Ida': '#17becf', 'Desmond': '#7f7f7f'}
for ev, sub in df.groupby('event'):
    ax.scatter(sub['actual_loss_M'], sub['ECL_paper_M'], s=80, alpha=0.7,
               c=colors.get(ev, 'gray'), label=f'{ev} (n={len(sub)})')
mn, mx = 10, 600
ax.plot([mn, mx], [mn, mx], 'k-', alpha=0.4, lw=1, label='1:1 (perfect)')
ax.plot([mn, mx], [0.5*mn, 0.5*mx], 'k--', alpha=0.3, lw=0.7)
ax.plot([mn, mx], [2*mn, 2*mx], 'k--', alpha=0.3, lw=0.7, label='0.5x / 2x bands')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Actual disclosed loss ($M)'); ax.set_ylabel('Paper-predicted ECL ($M)')
ax.set_title(f'24-Company OOS — paper-reported ECL vs actual\n'
             f'median ratio {median_r:.2f}x, {within_loose}/{n} within 0.4-2.5x')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig_out = os.path.join(ROOT, 'figures', 'oos_24_companies.png')
plt.savefig(fig_out, dpi=150, bbox_inches='tight')
print(f'Figure -> {fig_out}')
