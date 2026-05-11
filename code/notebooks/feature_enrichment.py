"""
Feature-enriched damage model: D(depth) → D(depth, building_features).

Shows how adding building characteristics reduces prediction error.

Models compared (all on Harvey 70/30 split):
  M0: D(depth) — Richards, depth only
  M1: D(depth, occupancy) — separate curves by building type
  M2: D(depth, occupancy, floors) — + number of floors
  M3: D(depth, occupancy, floors, basement) — + basement indicator
  M4: Gradient Boosting on all features — upper bound on achievable accuracy

Run: cd physical-climate-risk && PYTHONPATH=. python notebooks/feature_enrichment.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

os.makedirs("notebooks/figures", exist_ok=True)
np.random.seed(42)


# ============================================================================
# Load Harvey data with all features
# ============================================================================

df = pd.read_csv("data/raw/fema_harvey_claims.csv", low_memory=False)

# compute damage ratio
for col in ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
            "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
            "waterDepth", "numberOfFloorsInTheInsuredBuilding",
            "basementEnclosureCrawlspaceType", "buildingDamageAmount",
            "occupancyType", "buildingDescriptionCode"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["depth_m"] = df["waterDepth"] * 0.0254
df["total_paid"] = df["amountPaidOnBuildingClaim"].fillna(0) + df["amountPaidOnContentsClaim"].fillna(0)
df["total_coverage"] = df["totalBuildingInsuranceCoverage"].fillna(0) + df["totalContentsInsuranceCoverage"].fillna(0)
df["damage_ratio"] = (df["total_paid"] / df["total_coverage"]).clip(0, 1)

# filters
mask = (
    (df["waterDepth"] > 0) &
    (df["total_coverage"] > 1000) &
    (df["total_paid"] > 0) &
    (df["damage_ratio"] > 0) &
    (df["depth_m"] <= 6)
)
df = df[mask].copy()

# clean features
df["floors"] = df["numberOfFloorsInTheInsuredBuilding"].fillna(1).clip(1, 4).astype(int)
df["has_basement"] = (df["basementEnclosureCrawlspaceType"].fillna(0) > 0).astype(int)
df["occupancy"] = df["occupancyType"].fillna(1).astype(int)
df["is_residential"] = (df["occupancy"].isin([1, 2, 3])).astype(int)
df["primary_residence"] = df["primaryResidenceIndicator"].map({True: 1, False: 0}).fillna(0).astype(int)

print(f"Dataset: {len(df):,} claims with features")
print(f"Floors distribution: {dict(df['floors'].value_counts().sort_index())}")
print(f"Basement: {df['has_basement'].mean():.1%} have basement")
print(f"Occupancy types: {dict(df['occupancy'].value_counts().head(5))}")


# ============================================================================
# Train/test split
# ============================================================================

idx = np.random.permutation(len(df))
n_train = int(0.7 * len(df))
train = df.iloc[idx[:n_train]].copy()
test = df.iloc[idx[n_train:]].copy()

print(f"\nTrain: {len(train):,}, Test: {len(test):,}")


# ============================================================================
# Richards helper (raw parameterization)
# ============================================================================

def richards(xi, B, Q, midpoint, nu):
    """Raw Richards: D = 1 / (1 + Q*exp(-B*(x - midpoint)))^(1/nu)."""
    exp_term = np.clip(-B * (np.asarray(xi, dtype=float) - midpoint), -50, 50)
    denom = (1.0 + max(Q, 0.001) * np.exp(exp_term)) ** (1.0 / max(nu, 0.01))
    return np.clip(1.0 / denom, 0, 1)


def fit_richards(depth, dr, x0=None):
    """Fit Richards on data, return (B, Q, midpoint, nu)."""
    if x0 is None:
        x0 = [1.0, 1.0, 2.0, 0.5]

    def obj(theta):
        B, Q, mid, nu = theta
        if B <= 0 or Q <= 0 or nu <= 0.01:
            return 1e10
        pred = richards(depth, B, Q, mid, nu)
        return np.mean((dr - pred) ** 2)

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 10000})
    return res.x


def evaluate(y_true, y_pred):
    """Compute MAE, RMSE, R²."""
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ============================================================================
# M0: D(depth) — Richards, depth only
# ============================================================================

print("\n--- M0: D(depth) ---")
theta_m0 = fit_richards(train["depth_m"].values, train["damage_ratio"].values)
pred_m0_test = richards(test["depth_m"].values, *theta_m0)
pred_m0_train = richards(train["depth_m"].values, *theta_m0)
metrics_m0 = evaluate(test["damage_ratio"].values, pred_m0_test)
print(f"  Test: MAE={metrics_m0['mae']:.4f}, RMSE={metrics_m0['rmse']:.4f}, R²={metrics_m0['r2']:.4f}")


# ============================================================================
# M1: D(depth, occupancy) — separate Richards per occupancy type
# ============================================================================

print("\n--- M1: D(depth, occupancy) ---")
occ_map = {1: "Single Family", 2: "Multi Family", 3: "Other Resid", 4: "Non-Resid"}
occ_thetas = {}

for occ, label in occ_map.items():
    sub = train[train["occupancy"] == occ]
    if len(sub) < 100:
        occ_thetas[occ] = theta_m0  # fallback to global
        continue
    occ_thetas[occ] = fit_richards(sub["depth_m"].values, sub["damage_ratio"].values)
    print(f"  {label} (n={len(sub):,}): B={occ_thetas[occ][0]:.3f}, "
          f"mid={occ_thetas[occ][2]:.3f}")

pred_m1_test = np.zeros(len(test))
for occ in test["occupancy"].unique():
    mask = test["occupancy"] == occ
    theta = occ_thetas.get(occ, theta_m0)
    pred_m1_test[mask.values] = richards(test.loc[mask, "depth_m"].values, *theta)

metrics_m1 = evaluate(test["damage_ratio"].values, pred_m1_test)
print(f"  Test: MAE={metrics_m1['mae']:.4f}, RMSE={metrics_m1['rmse']:.4f}, R²={metrics_m1['r2']:.4f}")


# ============================================================================
# M2: D(depth, occupancy, floors) — separate per (occupancy × floors)
# ============================================================================

print("\n--- M2: D(depth, occupancy, floors) ---")
strat_thetas = {}

for occ in [1, 2, 3, 4]:
    for floors in [1, 2, 3]:
        sub = train[(train["occupancy"] == occ) & (train["floors"] == floors)]
        key = (occ, floors)
        if len(sub) < 50:
            strat_thetas[key] = occ_thetas.get(occ, theta_m0)
            continue
        strat_thetas[key] = fit_richards(sub["depth_m"].values, sub["damage_ratio"].values)

pred_m2_test = np.zeros(len(test))
for i, row in test.iterrows():
    key = (row["occupancy"], min(row["floors"], 3))
    theta = strat_thetas.get(key, theta_m0)
    pred_m2_test[test.index.get_loc(i)] = float(richards(row["depth_m"], *theta))

metrics_m2 = evaluate(test["damage_ratio"].values, pred_m2_test)
print(f"  Test: MAE={metrics_m2['mae']:.4f}, RMSE={metrics_m2['rmse']:.4f}, R²={metrics_m2['r2']:.4f}")


# ============================================================================
# M3: D(depth, occupancy, floors, basement) — add basement
# ============================================================================

print("\n--- M3: D(depth, occupancy, floors, basement) ---")
strat3_thetas = {}

for occ in [1, 2, 3, 4]:
    for floors in [1, 2, 3]:
        for bsmt in [0, 1]:
            sub = train[(train["occupancy"] == occ) &
                        (train["floors"] == floors) &
                        (train["has_basement"] == bsmt)]
            key = (occ, floors, bsmt)
            if len(sub) < 30:
                strat3_thetas[key] = strat_thetas.get((occ, floors), theta_m0)
                continue
            strat3_thetas[key] = fit_richards(
                sub["depth_m"].values, sub["damage_ratio"].values)

pred_m3_test = np.zeros(len(test))
for i, row in test.iterrows():
    key = (row["occupancy"], min(row["floors"], 3), row["has_basement"])
    theta = strat3_thetas.get(key, strat_thetas.get(
        (row["occupancy"], min(row["floors"], 3)),
        theta_m0))
    pred_m3_test[test.index.get_loc(i)] = float(richards(row["depth_m"], *theta))

metrics_m3 = evaluate(test["damage_ratio"].values, pred_m3_test)
print(f"  Test: MAE={metrics_m3['mae']:.4f}, RMSE={metrics_m3['rmse']:.4f}, R²={metrics_m3['r2']:.4f}")


# ============================================================================
# M4: Gradient Boosting — upper bound on achievable accuracy
# ============================================================================

print("\n--- M4: Gradient Boosting (all features) ---")

from sklearn.ensemble import GradientBoostingRegressor

feature_cols = ["depth_m", "occupancy", "floors", "has_basement",
                "is_residential", "primary_residence"]

X_train = train[feature_cols].values
y_train = train["damage_ratio"].values
X_test = test[feature_cols].values
y_test = test["damage_ratio"].values

gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42,
)
gb.fit(X_train, y_train)
pred_m4_test = np.clip(gb.predict(X_test), 0, 1)

metrics_m4 = evaluate(y_test, pred_m4_test)
print(f"  Test: MAE={metrics_m4['mae']:.4f}, RMSE={metrics_m4['rmse']:.4f}, R²={metrics_m4['r2']:.4f}")

# feature importance
importances = gb.feature_importances_
print(f"  Feature importance:")
for fname, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"    {fname:>20}: {imp:.3f}")


# ============================================================================
# Summary table
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Feature Enrichment Impact")
print("=" * 70)

models = [
    ("M0: D(depth)", metrics_m0),
    ("M1: + occupancy", metrics_m1),
    ("M2: + floors", metrics_m2),
    ("M3: + basement", metrics_m3),
    ("M4: GBM (upper bound)", metrics_m4),
]

print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAE Δ vs M0':>12}")
print("─" * 65)
m0_mae = metrics_m0["mae"]
for name, m in models:
    delta = (m["mae"] - m0_mae) / m0_mae * 100
    print(f"{name:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f} {delta:>+11.1f}%")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# --- 1. MAE comparison bar chart ---
ax1 = fig.add_subplot(gs[0, 0])
model_names = [n for n, _ in models]
maes = [m["mae"] for _, m in models]
colors_bar = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]
bars = ax1.bar(range(len(models)), maes, color=colors_bar, edgecolor="white")
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels([n.split(":")[0] for n in model_names], fontsize=9)
ax1.set_ylabel("MAE (out-of-sample)")
ax1.set_title("MAE Improvement with Building Features")
for i, (bar, mae) in enumerate(zip(bars, maes)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{mae:.3f}", ha="center", fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# --- 2. R² comparison ---
ax2 = fig.add_subplot(gs[0, 1])
r2s = [m["r2"] for _, m in models]
bars2 = ax2.bar(range(len(models)), r2s, color=colors_bar, edgecolor="white")
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels([n.split(":")[0] for n in model_names], fontsize=9)
ax2.set_ylabel("R² (out-of-sample)")
ax2.set_title("Explained Variance")
ax2.axhline(0, color="black", linewidth=0.5)
for i, (bar, r2) in enumerate(zip(bars2, r2s)):
    ax2.text(bar.get_x() + bar.get_width()/2,
             max(bar.get_height(), 0) + 0.005,
             f"{r2:.3f}", ha="center", fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# --- 3. Feature importance (GBM) ---
ax3 = fig.add_subplot(gs[0, 2])
sorted_idx = np.argsort(importances)
ax3.barh(range(len(feature_cols)),
         importances[sorted_idx],
         color="steelblue", alpha=0.8)
ax3.set_yticks(range(len(feature_cols)))
ax3.set_yticklabels([feature_cols[i] for i in sorted_idx])
ax3.set_xlabel("Feature Importance")
ax3.set_title("GBM Feature Importance")
ax3.grid(True, alpha=0.3, axis="x")

# --- 4. Predicted vs observed: M0 vs M4 ---
ax4 = fig.add_subplot(gs[1, 0])
idx_plot = np.random.choice(len(test), min(3000, len(test)), replace=False)
ax4.scatter(y_test[idx_plot], pred_m0_test[idx_plot],
            s=3, alpha=0.2, color="red", label="M0: depth only")
ax4.scatter(y_test[idx_plot], pred_m4_test[idx_plot],
            s=3, alpha=0.2, color="blue", label="M4: GBM")
ax4.plot([0, 1], [0, 1], "k--", linewidth=1)
ax4.set_xlabel("Observed DR")
ax4.set_ylabel("Predicted DR")
ax4.set_title("M0 vs M4: Predicted vs Observed")
ax4.legend(fontsize=8)
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.grid(True, alpha=0.3)

# --- 5. Curves by occupancy type ---
ax5 = fig.add_subplot(gs[1, 1])
xi = np.linspace(0, 6, 200)
occ_colors = {1: "blue", 2: "red", 3: "green", 4: "orange"}

for occ, label in occ_map.items():
    theta = occ_thetas.get(occ, theta_m0)
    D = richards(xi, *theta)
    n_test = (test["occupancy"] == occ).sum()
    ax5.plot(xi, D, color=occ_colors[occ], linewidth=2,
             label=f"{label} (n={n_test:,})")

ax5.plot(xi, richards(xi, *theta_m0), "k--", linewidth=1.5,
         alpha=0.5, label="All buildings")
ax5.set_xlabel("Flood Depth (m)")
ax5.set_ylabel("Damage Ratio")
ax5.set_title("M1: Separate Curves by Building Type")
ax5.legend(fontsize=8)
ax5.set_ylim(-0.05, 1.05)
ax5.grid(True, alpha=0.3)

# --- 6. Curves by floors ---
ax6 = fig.add_subplot(gs[1, 2])
floor_colors = {1: "darkred", 2: "darkorange", 3: "darkgreen"}

for floors in [1, 2, 3]:
    # single family + this floor count
    key = (1, floors)
    theta = strat_thetas.get(key, theta_m0)
    D = richards(xi, *theta)
    n_sub = ((test["occupancy"] == 1) & (test["floors"] == floors)).sum()
    ax6.plot(xi, D, color=floor_colors[floors], linewidth=2,
             label=f"Single Family, {floors}F (n={n_sub:,})")

ax6.set_xlabel("Flood Depth (m)")
ax6.set_ylabel("Damage Ratio")
ax6.set_title("M2: Single Family by Floor Count")
ax6.legend(fontsize=8)
ax6.set_ylim(-0.05, 1.05)
ax6.grid(True, alpha=0.3)

# --- 7. Residual reduction ---
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(y_test - pred_m0_test, bins=50, alpha=0.4, color="red",
         density=True, label=f"M0 (MAE={metrics_m0['mae']:.3f})")
ax7.hist(y_test - pred_m4_test, bins=50, alpha=0.4, color="blue",
         density=True, label=f"M4 (MAE={metrics_m4['mae']:.3f})")
ax7.set_xlabel("Residual")
ax7.set_ylabel("Density")
ax7.set_title("Residual Distribution: M0 vs M4")
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# --- 8. MAE by depth bin: M0 vs M1 vs M4 ---
ax8 = fig.add_subplot(gs[2, 1])
depth_bins = np.linspace(0, 6, 13)
bin_idx = np.clip(np.digitize(test["depth_m"].values, depth_bins) - 1, 0, 11)

for model_name, pred, color in [
    ("M0: depth only", pred_m0_test, "red"),
    ("M1: + occupancy", pred_m1_test, "orange"),
    ("M4: GBM", pred_m4_test, "blue"),
]:
    bin_maes = []
    bin_centers = []
    for b in range(12):
        mask = bin_idx == b
        if mask.sum() < 20:
            continue
        bin_mae = np.mean(np.abs(y_test[mask] - pred[mask]))
        bin_maes.append(bin_mae)
        bin_centers.append(depth_bins[b] + 0.25)
    ax8.plot(bin_centers, bin_maes, "o-", color=color, linewidth=2,
             markersize=5, label=model_name)

ax8.set_xlabel("Flood Depth (m)")
ax8.set_ylabel("MAE")
ax8.set_title("MAE by Depth Bin")
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# --- 9. Summary text ---
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")

summary = (
    f"Feature Enrichment Summary\n"
    f"{'═' * 40}\n"
    f"Dataset: {len(df):,} Harvey claims\n"
    f"Train: {len(train):,} / Test: {len(test):,}\n\n"
    f"{'Model':<22} {'MAE':>6} {'R²':>7} {'Δ MAE':>7}\n"
    f"{'─' * 45}\n"
)
for name, m in models:
    delta = (m["mae"] - m0_mae) / m0_mae * 100
    summary += f"{name:<22} {m['mae']:.3f}  {m['r2']:>6.3f}  {delta:>+6.1f}%\n"
summary += (
    f"\n{'─' * 45}\n"
    f"Depth alone explains {max(metrics_m0['r2'], 0)*100:.0f}% of variance\n"
    f"+ features adds {max((metrics_m4['r2'] - metrics_m0['r2']), 0)*100:.0f}pp\n"
    f"MAE reduction: {(m0_mae - metrics_m4['mae'])/m0_mae*100:.0f}%\n\n"
    f"Irreducible noise:\n"
    f"  GBM MAE = {metrics_m4['mae']:.3f} is likely\n"
    f"  near the floor for this dataset\n"
    f"  (binary DR at 0 or 1 creates noise)\n"
)
ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
         fontsize=9.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.suptitle("Feature Enrichment: How Building Characteristics Reduce Prediction Error",
             fontsize=14, y=1.01)
plt.savefig("figures/12_feature_enrichment.png",
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → notebooks/figures/12_feature_enrichment.png")
