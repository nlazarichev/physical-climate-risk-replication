"""
Regenerate duration_curves.png (Fig. 4) using canonical calibration.

Reads parameters from params/unified_params.csv (output of unified_calibration.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

H_MIN = 0.05
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT_DIR = os.path.join(ROOT, "figures")
PARAMS_CSV = os.path.join(ROOT, "params", "unified_params.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# Read canonical params
params = pd.read_csv(PARAMS_CSV)['value'].values
B, Q0, a0, nu0, lambda_a, lambda_Q, delta_nu = params
tau0 = 48.0
F_min = 0.0
F_s = 8.0


def duration_richards(h, tau):
    ratio = tau / tau0
    a_eff = a0 * np.exp(-lambda_a * ratio)
    Q_eff = max(Q0 * np.exp(-lambda_Q * ratio), 0.001)
    nu_eff = max(nu0 + delta_nu * np.log(1 + ratio), 0.01)
    h_norm = (h - F_min) / (F_s - F_min)
    return 1.0 / (1.0 + Q_eff * np.exp(-B * (h_norm - a_eff))) ** (1.0 / nu_eff)


h = np.linspace(H_MIN, 5.0, 300)

tau_values = [0, 36, 120, 288]
labels = [r'$\tau = 0$ (instantaneous)', r'$\tau = 36$\,h (Sandy)',
          r'$\tau = 120$\,h (Katrina)', r'$\tau = 288$\,h (Harvey)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(7, 4.5))

for tau, label, color in zip(tau_values, labels, colors):
    y = duration_richards(h, tau)
    ax.plot(h, y, color=color, linewidth=2, label=label)

dr_0 = duration_richards(np.array([1.0]), 0)[0]
dr_max = duration_richards(np.array([1.0]), 288)[0]
amp = dr_max / dr_0

ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.annotate('', xy=(1.0, dr_0), xytext=(1.0, dr_max),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(1.05, (dr_0 + dr_max) / 2, f'{amp:.1f}' + r'$\times$',
        fontsize=9, color='gray', va='center')

ax.set_xlabel('Inundation depth (m)', fontsize=11)
ax.set_ylabel('Damage ratio', fontsize=11)
ax.set_xlim(left=H_MIN, right=5.0)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'duration_curves.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved -> {out_path}")
print(f"  amplification at 1m: {amp:.3f}x  (DR at tau=0: {dr_0:.3f}, tau=288h: {dr_max:.3f})")
