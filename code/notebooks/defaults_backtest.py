"""
Defaults backtest: Does our predicted DR correlate with real financial distress?

Data: IHP registrations by ZIP (311 ZIPs, Harvey)
  - pct_destroyed: % of registrations where property was destroyed
  - pct_sba_approved: % that needed SBA disaster loan (financial distress proxy)
  - mean_water_depth_m: average reported flood depth

Test: predict DR per ZIP → correlate with actual distress metrics.
If our model works, predicted DR should correlate with:
  1. % properties destroyed
  2. % SBA loans approved (people who needed emergency financing)
  3. Average FEMA damage amount

Run: cd physical-climate-risk && PYTHONPATH=. python notebooks/defaults_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr

from damage_functions.duration_richards import DurationRichards, DurationRichardsParams

os.makedirs("notebooks/figures", exist_ok=True)

# ============================================================================
# Load data
# ============================================================================

ihp = pd.read_csv("data/processed/ihp_harvey_by_zip.csv")
print(f"IHP data: {len(ihp)} ZIPs")
print(f"Columns: {list(ihp.columns)}")

# filter: need valid water depth
ihp = ihp[ihp["mean_water_depth_m"] > 0].copy()
print(f"After depth filter: {len(ihp)} ZIPs")

# load calibrated model
params_df = pd.read_csv("params/unified_params.csv")
theta = params_df["value"].values
model = DurationRichards(DurationRichardsParams.from_vector(theta))

HARVEY_DURATION = 288  # hours

# ============================================================================
# Predictions
# ============================================================================

# predict DR for each ZIP using mean water depth
ihp["predicted_dr"] = ihp["mean_water_depth_m"].apply(
    lambda d: float(model(d, HARVEY_DURATION))
)

# also predict with USACE (no duration) for comparison
usace_depth = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.4])
usace_dr = np.array([0.08, 0.18, 0.27, 0.35, 0.43, 0.48, 0.55])

ihp["predicted_dr_usace"] = ihp["mean_water_depth_m"].apply(
    lambda d: float(np.interp(d, usace_depth, usace_dr, right=usace_dr[-1]))
)

# JRC Residential for comparison
jrc_depth = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
jrc_dr = np.array([0.00, 0.25, 0.40, 0.50, 0.57, 0.67, 0.74, 0.79, 0.82])

ihp["predicted_dr_jrc"] = ihp["mean_water_depth_m"].apply(
    lambda d: float(np.interp(d, jrc_depth, jrc_dr, right=jrc_dr[-1]))
)

# ============================================================================
# Distress proxies
# ============================================================================

# normalize damage amount to [0, 1] for correlation
ihp["norm_damage_amt"] = ihp["mean_flood_damage_amt"] / ihp["mean_flood_damage_amt"].max()

# composite distress score: weighted average of multiple proxies
ihp["distress_score"] = (
    0.4 * ihp["pct_destroyed"] / max(ihp["pct_destroyed"].max(), 0.001) +
    0.3 * ihp["pct_sba_approved"] / max(ihp["pct_sba_approved"].max(), 0.001) +
    0.3 * ihp["norm_damage_amt"]
)

# ============================================================================
# Correlation analysis
# ============================================================================

distress_metrics = {
    "% Destroyed": "pct_destroyed",
    "% SBA Approved": "pct_sba_approved",
    "% Flood Damage": "pct_flood_damage",
    "Mean Damage ($)": "mean_flood_damage_amt",
    "Mean IHP ($)": "mean_ihp_amount",
    "Distress Score": "distress_score",
}

models_to_test = {
    "Duration-Richards (τ=288h)": "predicted_dr",
    "USACE Reference": "predicted_dr_usace",
    "JRC Residential": "predicted_dr_jrc",
    "Raw Depth (m)": "mean_water_depth_m",
}

print("\n" + "=" * 80)
print("CORRELATION: Predicted DR vs Real Financial Distress")
print("=" * 80)

results = {}
for model_name, pred_col in models_to_test.items():
    results[model_name] = {}
    print(f"\n  {model_name}:")
    for metric_name, metric_col in distress_metrics.items():
        valid = ihp[[pred_col, metric_col]].dropna()
        if len(valid) < 10:
            continue
        r_pearson, p_pearson = pearsonr(valid[pred_col], valid[metric_col])
        r_spearman, p_spearman = spearmanr(valid[pred_col], valid[metric_col])
        results[model_name][metric_name] = {
            "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman,
        }
        sig = "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else ""
        print(f"    vs {metric_name:<20}: "
              f"Spearman r={r_spearman:.3f}{sig}, "
              f"Pearson r={r_pearson:.3f}")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# --- 1. Scatter: predicted DR vs % SBA approved ---
ax1 = fig.add_subplot(gs[0, 0])
for model_name, pred_col, color, marker in [
    ("Ours (τ=288h)", "predicted_dr", "darkred", "o"),
    ("USACE", "predicted_dr_usace", "green", "s"),
    ("JRC", "predicted_dr_jrc", "steelblue", "D"),
]:
    ax1.scatter(ihp[pred_col], ihp["pct_sba_approved"],
                s=20, alpha=0.5, color=color, marker=marker, label=model_name)
ax1.set_xlabel("Predicted Damage Ratio")
ax1.set_ylabel("% SBA Loans Approved (distress proxy)")
ax1.set_title("Predicted DR vs Financial Distress")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- 2. Scatter: predicted DR vs % destroyed ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(ihp["predicted_dr"], ihp["pct_destroyed"],
            s=ihp["n_registrations"] / 10, alpha=0.5, color="darkred",
            edgecolors="white", linewidth=0.3)
r, p = spearmanr(ihp["predicted_dr"], ihp["pct_destroyed"])
ax2.set_xlabel("Predicted DR (Duration-Richards)")
ax2.set_ylabel("% Properties Destroyed")
ax2.set_title(f"Predicted DR vs Destruction Rate (ρ={r:.3f}, p={p:.1e})")
ax2.grid(True, alpha=0.3)

# --- 3. Scatter: predicted DR vs mean damage amount ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(ihp["predicted_dr"], ihp["mean_flood_damage_amt"] / 1000,
            s=20, alpha=0.5, color="darkred")
r, p = spearmanr(ihp["predicted_dr"], ihp["mean_flood_damage_amt"])
ax3.set_xlabel("Predicted DR (Duration-Richards)")
ax3.set_ylabel("Mean FEMA Damage Amount ($K)")
ax3.set_title(f"Predicted DR vs Actual Damage (ρ={r:.3f})")
ax3.grid(True, alpha=0.3)

# --- 4. Correlation heatmap ---
ax4 = fig.add_subplot(gs[1, 0])

corr_matrix = np.zeros((len(models_to_test), len(distress_metrics)))
for i, (mn, _) in enumerate(models_to_test.items()):
    for j, (dn, _) in enumerate(distress_metrics.items()):
        if mn in results and dn in results[mn]:
            corr_matrix[i, j] = results[mn][dn]["spearman_r"]

im = ax4.imshow(corr_matrix, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
ax4.set_xticks(range(len(distress_metrics)))
ax4.set_xticklabels([n[:12] for n in distress_metrics.keys()],
                      fontsize=7, rotation=45, ha="right")
ax4.set_yticks(range(len(models_to_test)))
ax4.set_yticklabels(list(models_to_test.keys()), fontsize=8)

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        ax4.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                 fontsize=8, color="black" if abs(corr_matrix[i,j]) < 0.4 else "white")

ax4.set_title("Spearman Correlation Matrix")
plt.colorbar(im, ax=ax4, shrink=0.8)

# --- 5. Bar chart: mean Spearman r across distress metrics ---
ax5 = fig.add_subplot(gs[1, 1])
mean_corrs = {}
for mn in models_to_test:
    if mn in results:
        corrs = [results[mn][dn]["spearman_r"]
                 for dn in distress_metrics if dn in results[mn]]
        mean_corrs[mn] = np.mean(corrs) if corrs else 0

colors_model = ["darkred", "green", "steelblue", "gray"]
bars = ax5.bar(range(len(mean_corrs)),
               list(mean_corrs.values()),
               color=colors_model[:len(mean_corrs)],
               edgecolor="white", alpha=0.8)
ax5.set_xticks(range(len(mean_corrs)))
ax5.set_xticklabels([n.split("(")[0].strip() for n in mean_corrs.keys()],
                      fontsize=8, rotation=30, ha="right")
ax5.set_ylabel("Mean Spearman ρ")
ax5.set_title("Average Correlation with Financial Distress")
for bar, v in zip(bars, mean_corrs.values()):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax5.grid(True, alpha=0.3, axis="y")

# --- 6. Depth distribution of IHP ZIPs ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(ihp["mean_water_depth_m"], bins=30, color="steelblue",
         edgecolor="white", alpha=0.8)
ax6.set_xlabel("Mean Water Depth (m)")
ax6.set_ylabel("Number of ZIPs")
ax6.set_title(f"Flood Depth Distribution ({len(ihp)} ZIPs)")
ax6.grid(True, alpha=0.3)

# --- 7. Quintile analysis: sort ZIPs by predicted DR, show actual distress ---
ax7 = fig.add_subplot(gs[2, 0])
ihp_sorted = ihp.sort_values("predicted_dr")
n_q = 5
q_size = len(ihp_sorted) // n_q

quintile_stats = []
for q in range(n_q):
    start = q * q_size
    end = start + q_size if q < n_q - 1 else len(ihp_sorted)
    chunk = ihp_sorted.iloc[start:end]
    quintile_stats.append({
        "quintile": f"Q{q+1}",
        "mean_pred_dr": chunk["predicted_dr"].mean(),
        "mean_pct_destroyed": chunk["pct_destroyed"].mean(),
        "mean_pct_sba": chunk["pct_sba_approved"].mean(),
        "mean_damage": chunk["mean_flood_damage_amt"].mean(),
    })

qs = pd.DataFrame(quintile_stats)
x = np.arange(n_q)
width = 0.35

ax7.bar(x - width/2, qs["mean_pct_sba"] * 100, width,
        label="% SBA Approved", color="darkred", alpha=0.7)
ax7.bar(x + width/2, qs["mean_pct_destroyed"] * 100, width,
        label="% Destroyed", color="steelblue", alpha=0.7)
ax7.set_xticks(x)
ax7.set_xticklabels([f"Q{i+1}\n(DR={qs.iloc[i]['mean_pred_dr']:.2f})"
                      for i in range(n_q)], fontsize=8)
ax7.set_ylabel("Percentage (%)")
ax7.set_title("Quintile Analysis: Predicted DR → Actual Distress")
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3, axis="y")

# --- 8. Quintile: damage amount ---
ax8 = fig.add_subplot(gs[2, 1])
ax8.bar(x, qs["mean_damage"] / 1000, color="darkorange", alpha=0.8)
ax8.set_xticks(x)
ax8.set_xticklabels([f"Q{i+1}" for i in range(n_q)])
ax8.set_ylabel("Mean FEMA Damage ($K)")
ax8.set_title("Mean Damage Amount by Predicted DR Quintile")
ax8.grid(True, alpha=0.3, axis="y")

for i, v in enumerate(qs["mean_damage"] / 1000):
    ax8.text(i, v + 0.2, f"${v:.1f}K", ha="center", fontsize=9)

# --- 9. Summary ---
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")

# find best model
best_model = max(mean_corrs, key=mean_corrs.get)
best_corr = mean_corrs[best_model]

summary = (
    f"Defaults Backtest Summary\n"
    f"{'═' * 40}\n"
    f"Data: {len(ihp)} Harvey ZIPs (IHP)\n"
    f"Distress proxies: % destroyed,\n"
    f"  % SBA approved, damage amount\n\n"
    f"Best predictor: {best_model}\n"
    f"  Mean Spearman ρ = {best_corr:.3f}\n\n"
    f"Quintile Monotonicity Test\n"
    f"{'─' * 40}\n"
    f"Q1→Q5 SBA rate:  {qs.iloc[0]['mean_pct_sba']*100:.1f}% → "
    f"{qs.iloc[-1]['mean_pct_sba']*100:.1f}%\n"
    f"Q1→Q5 destroyed: {qs.iloc[0]['mean_pct_destroyed']*100:.2f}% → "
    f"{qs.iloc[-1]['mean_pct_destroyed']*100:.2f}%\n"
    f"Q1→Q5 damage:    ${qs.iloc[0]['mean_damage']/1000:.1f}K → "
    f"${qs.iloc[-1]['mean_damage']/1000:.1f}K\n\n"
    f"Conclusion:\n"
    f"{'─' * 40}\n"
)

# check monotonicity
sba_monotonic = all(qs["mean_pct_sba"].iloc[i] <= qs["mean_pct_sba"].iloc[i+1]
                     for i in range(n_q - 1))
destr_monotonic = all(qs["mean_pct_destroyed"].iloc[i] <= qs["mean_pct_destroyed"].iloc[i+1]
                       for i in range(n_q - 1))

if sba_monotonic and destr_monotonic:
    summary += "✓ PASS: Perfect monotonicity\n  across all quintiles"
elif sba_monotonic or destr_monotonic:
    summary += "◐ PARTIAL: Monotonic for\n  some distress metrics"
else:
    summary += "? MIXED: Check quintile\n  patterns visually"

ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
         fontsize=9.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.suptitle("Defaults Backtest: Does Predicted DR Predict Real Financial Distress?",
             fontsize=14, y=1.01)
plt.savefig("figures/14_defaults_backtest.png",
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → notebooks/figures/14_defaults_backtest.png")
