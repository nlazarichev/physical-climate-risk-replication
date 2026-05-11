"""
Regenerate DF.png (Fig. 4) for revision v3.

Change vs v2: x-axis starts at H_MIN=0.1m, not 0.
Caption footnote added in LaTeX: "Curves start at h=0.1m; the Richards function
is not defined at h=0 as it represents the minimum inundation threshold."

Run: cd physical-climate-risk && PYTHONPATH=. python revision_v3/notebooks/generate_df_png.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

H_MIN = 0.1  # fig 4 uses illustrative params; 0.1m is cleaner threshold for caption
OUT_DIR = os.path.join(os.path.dirname(__file__), "../Content/Images")
os.makedirs(OUT_DIR, exist_ok=True)


def generalised_logistic(x, A=0, K=1, C=1, B=0.4, Q=1, nu=1, a=-10):
    return A + (K - A) / (C + Q * np.exp(-B * (x - a))) ** (1 / nu)


x = np.linspace(H_MIN, 15, 400)
y = generalised_logistic(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, "b-", linewidth=2)
ax.set_xlabel("Hazard intensity")
ax.set_ylabel("Damage ratio")
ax.set_title("Generalised logistic (Richards) damage function")
ax.set_xlim(left=H_MIN)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.annotate(
    r"$h \geq h_{\min}$",
    xy=(H_MIN, generalised_logistic(H_MIN)),
    xytext=(H_MIN + 0.5, 0.1),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=9, color="gray",
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "DF.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {os.path.join(OUT_DIR, 'DF.png')}")
