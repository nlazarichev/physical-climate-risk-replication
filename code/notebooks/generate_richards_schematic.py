"""
Regenerate richards_schematic.png (Fig. 3) for revision v3.

Change vs submitted: curves start at H_MIN=0.05m (R3 fix).
Uses illustrative parameters (not the calibrated ones) to show the
S-shaped Richards curve with annotated key features.

Run: cd physical-climate-risk && PYTHONPATH=. python revision_v3/notebooks/generate_richards_schematic.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

H_MIN = 0.05
OUT_DIR = os.path.join(os.path.dirname(__file__), "../Content/Images")
os.makedirs(OUT_DIR, exist_ok=True)

# Illustrative parameters for a clean schematic
B = 3.5
Q = 1.0
a = 0.45   # inflection point (normalised)
nu = 1.0
F_min = 0.0
F_s = 5.0  # normalisation ceiling


def richards(h, B=B, Q=Q, a=a, nu=nu, F_min=F_min, F_s=F_s):
    h_norm = (h - F_min) / (F_s - F_min)
    return 1.0 / (1.0 + Q * np.exp(-B * (h_norm - a))) ** (1.0 / nu)


h = np.linspace(H_MIN, 5.0, 500)
y = richards(h)

# Key points for annotations
h_infl = a * (F_s - F_min) + F_min          # inflection depth in metres
y_infl = richards(h_infl)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(h, y, 'steelblue', linewidth=2.5)

# ── Asymptotes ──────────────────────────────────────────────────────────────
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(4.85, 1.03, r'$K = 1$ (total loss)', ha='right', fontsize=9, color='gray')
ax.text(4.85, -0.05, r'$A = 0$ (no damage)', ha='right', fontsize=9, color='gray')

# ── Inflection point ─────────────────────────────────────────────────────────
ax.plot(h_infl, y_infl, 'o', color='steelblue', markersize=7, zorder=5)
ax.annotate(
    r'inflection point $a$' '\n' r'($\approx$50\% damage depth)',
    xy=(h_infl, y_infl),
    xytext=(h_infl + 0.8, y_infl - 0.18),
    arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.2),
    fontsize=9, color='dimgray',
)

# ── Steepness annotation (tangent at inflection) ─────────────────────────────
# Slope at inflection: dy/dh = B/(4 * (F_s - F_min)) for nu=1, Q=1
slope = B / (4 * (F_s - F_min))
dh = 0.6
h_tan = np.array([h_infl - dh, h_infl + dh])
y_tan = y_infl + slope * (h_tan - h_infl)
ax.plot(h_tan, y_tan, 'tomato', linewidth=1.5, linestyle='--', alpha=0.85)
ax.text(h_infl + dh + 0.05, y_tan[-1] + 0.03,
        r'slope $\propto B$', fontsize=9, color='tomato')

# ── Region labels ─────────────────────────────────────────────────────────────
ax.text(0.4, 0.07, 'negligible\ndamage regime', fontsize=8.5,
        color='#555', ha='center', style='italic')
ax.text(h_infl, 0.87, 'saturation', fontsize=8.5,
        color='#555', ha='center', style='italic')

# ── h_min marker ─────────────────────────────────────────────────────────────
ax.axvline(H_MIN, color='darkgreen', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(H_MIN + 0.05, 0.62,
        r'$h_{\min} = 0.05$\,m', fontsize=8.5, color='darkgreen', va='center')

# ── Non-zero floor note ───────────────────────────────────────────────────────
y_at_hmin = richards(H_MIN)
ax.annotate(
    fr'$\mathrm{{DF}}(h_{{\min}}) \approx {y_at_hmin:.2f}$',
    xy=(H_MIN, y_at_hmin),
    xytext=(0.6, y_at_hmin + 0.09),
    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.0),
    fontsize=8.5, color='darkgreen',
)

ax.set_xlabel('Inundation depth $h$ (m)', fontsize=11)
ax.set_ylabel('Damage ratio $\\mathrm{DF}$', fontsize=11)
ax.set_xlim(H_MIN - 0.1, 5.0)
ax.set_ylim(-0.08, 1.12)
ax.grid(True, alpha=0.25)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'richards_schematic.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out_path}")
