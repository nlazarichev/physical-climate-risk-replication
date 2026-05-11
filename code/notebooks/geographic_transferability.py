"""
Section 6.4 — Geographic transferability.

Reproduces paper claim: 24-company OOS broken down by region.
- US Harvey n=8: median ratio 1.2× (within 0.4-2.5×)
- Thailand n=8: median ratio 0.78× (within 0.4-2.5×)
- Other (single observations): case studies, not statistical
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

df = pd.read_csv(os.path.join(ROOT, "data/processed/oos_24_companies.csv"))
df['ratio'] = df['ECL_paper_M'] / df['actual_loss_M']

# group by region archetype
def archetype(event):
    if event == 'Harvey':   return 'US Harvey (Gulf tropical-cyclone surge)'
    if event == 'Thailand': return 'Thailand (monsoonal industrial-park)'
    return 'Other (single events)'

df['archetype'] = df['event'].map(archetype)

print(f"\n{'Archetype':<48} {'n':>3} {'median ratio':>14} {'range':>12}  {'within 0.4-2.5×':>17}")
print('-' * 100)
for arch, sub in df.groupby('archetype'):
    n = len(sub)
    med = sub['ratio'].median()
    lo, hi = sub['ratio'].min(), sub['ratio'].max()
    within = ((sub['ratio'] >= 0.4) & (sub['ratio'] <= 2.5)).sum()
    print(f"{arch:<48} {n:>3} {med:>14.2f}× {lo:>5.2f}-{hi:<5.2f}  {within}/{n} ({100*within/n:.0f}%)")

print('-' * 100)
total_within = ((df['ratio'] >= 0.4) & (df['ratio'] <= 2.5)).sum()
print(f"{'TOTAL':<48} {len(df):>3} {df['ratio'].median():>14.2f}× "
      f"{df['ratio'].min():>5.2f}-{df['ratio'].max():<5.2f}  "
      f"{total_within}/{len(df)} ({100*total_within/len(df):.0f}%)")

print(f"\nPaper §6.4: US Harvey n=8 median 1.2×; Thailand n=8 median 0.78×")
print(f"Ours:       US Harvey n=8 median {df[df['event']=='Harvey']['ratio'].median():.2f}×; "
      f"Thailand n=8 median {df[df['event']=='Thailand']['ratio'].median():.2f}×")

out = os.path.join(ROOT, "params", "geographic_transferability.csv")
gt = df.groupby('archetype').agg(
    n=('company', 'count'),
    median_ratio=('ratio', 'median'),
    mean_ratio=('ratio', 'mean'),
    min_ratio=('ratio', 'min'),
    max_ratio=('ratio', 'max'),
    within_band=('ratio', lambda x: ((x >= 0.4) & (x <= 2.5)).sum())
).reset_index()
gt.to_csv(out, index=False)
print(f"\nSaved -> {out}")
