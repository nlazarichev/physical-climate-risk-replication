"""
Depth-dependent κ(D): improving transfer learning from residential to commercial.

Constant κ:    D_commercial = D_residential × κ         (same ratio at all depths)
Proposed κ(D): D_commercial = D_residential × κ(D)      (ratio increases with severity)

κ(D) = κ₀ + (1 - κ₀) · D^γ

Physics:
  - At low D (shallow flood): commercial buildings barely affected
    (concrete floors, elevated equipment) → κ ≈ κ₀ ≈ 0
  - At high D (deep flood): everything destroyed equally → κ → 1.0
  - γ controls how fast commercial "catches up" to residential

Calibration: use facility-level data from 3 event study companies.

Run: cd physical-climate-risk && PYTHONPATH=. python notebooks/kappa_calibration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

from damage_functions.duration_richards import DurationRichards, DurationRichardsParams

os.makedirs("notebooks/figures", exist_ok=True)

params_df = pd.read_csv("params/unified_params.csv")
model = DurationRichards(DurationRichardsParams.from_vector(params_df["value"].values))

OPEX_R_BASE = 0.5
OPEX_R_SCALE = 3.0
OPEX_ALPHA = 1.5


# ============================================================================
# κ models
# ============================================================================

def kappa_const(D, kappa_0):
    """Constant κ (current approach)."""
    return np.full_like(np.asarray(D, dtype=float), kappa_0)


def kappa_depth(D, kappa_0, gamma):
    """Depth-dependent κ: κ(D) = κ₀ + (1-κ₀)·D^γ.

    At D=0: κ = κ₀ (base resilience — near 0 for industrial)
    At D=1: κ = 1.0 (total destruction is universal)
    γ > 1: slow onset, fast convergence at high D
    γ < 1: fast onset (commercial catches up quickly)
    γ = 1: linear
    """
    D = np.asarray(D, dtype=float)
    return kappa_0 + (1.0 - kappa_0) * np.clip(D, 0, 1) ** gamma


def compute_loss_with_kappa(facilities, kappa_func, kappa_params, duration_h=288):
    """Compute total loss using a κ function."""
    total = 0.0
    details = []

    for f in facilities:
        dr_resid = float(model(f["depth_m"], duration_h))
        kappa_val = float(kappa_func(dr_resid, *kappa_params))
        dr_adj = dr_resid * kappa_val
        capex_net = dr_adj * f["value_M"] * (1 - f["insurance"])
        opex = capex_net * (OPEX_R_BASE + OPEX_R_SCALE * (max(dr_adj, 0.001) ** OPEX_ALPHA))
        loss = capex_net + opex
        total += loss
        details.append({
            "name": f["name"],
            "depth": f["depth_m"],
            "dr_resid": dr_resid,
            "kappa": kappa_val,
            "dr_adj": dr_adj,
            "loss_M": loss,
        })

    return total, details


# ============================================================================
# Event study data
# ============================================================================

companies = {
    "Sysco": {
        "sector": "distribution", "actual_loss_M": 7,
        "facilities": [
            {"name": "HQ", "value_M": 200, "depth_m": 0.5, "duration_h": 168, "insurance": 0.7},
            {"name": "DC Greens", "value_M": 80, "depth_m": 1.5, "duration_h": 240, "insurance": 0.3},
            {"name": "DC #2", "value_M": 60, "depth_m": 0.8, "duration_h": 120, "insurance": 0.3},
        ],
        "color": "#2196F3",
    },
    "NRG": {
        "sector": "energy", "actual_loss_M": 55,
        "facilities": [
            {"name": "WA Parish", "value_M": 2000, "depth_m": 1.2, "duration_h": 240, "insurance": 0.6},
            {"name": "Cedar Bayou", "value_M": 500, "depth_m": 1.8, "duration_h": 288, "insurance": 0.5},
            {"name": "San Jacinto", "value_M": 300, "depth_m": 0.8, "duration_h": 168, "insurance": 0.5},
            {"name": "HQ", "value_M": 50, "depth_m": 0.3, "duration_h": 72, "insurance": 0.8},
            {"name": "Reliant", "value_M": 100, "depth_m": 0.5, "duration_h": 120, "insurance": 0.4},
        ],
        "color": "#FF9800",
    },
    "HCA": {
        "sector": "healthcare", "actual_loss_M": 100,
        "facilities": [
            {"name": "East Houston", "value_M": 80, "depth_m": 1.83, "duration_h": 336, "insurance": 0.5},
            {"name": "Bayshore", "value_M": 120, "depth_m": 0.6, "duration_h": 168, "insurance": 0.6},
            {"name": "Clear Lake", "value_M": 150, "depth_m": 0.3, "duration_h": 96, "insurance": 0.6},
            {"name": "Houston NW", "value_M": 100, "depth_m": 0.5, "duration_h": 120, "insurance": 0.5},
            {"name": "Cypress", "value_M": 90, "depth_m": 0.4, "duration_h": 96, "insurance": 0.6},
            {"name": "Woman's Hosp", "value_M": 100, "depth_m": 0.2, "duration_h": 72, "insurance": 0.7},
            {"name": "Other (8)", "value_M": 500, "depth_m": 0.3, "duration_h": 72, "insurance": 0.6},
        ],
        "color": "#4CAF50",
    },
}


# ============================================================================
# Calibrate κ(D) jointly on all 3 companies
# ============================================================================

def joint_objective(params, kappa_func):
    """Sum of squared log-errors across all companies."""
    total = 0.0
    for name, c in companies.items():
        pred, _ = compute_loss_with_kappa(c["facilities"], kappa_func, params)
        actual = c["actual_loss_M"]
        if pred > 0 and actual > 0:
            # log-space error (handles order-of-magnitude differences)
            total += (np.log(pred) - np.log(actual)) ** 2
    return total


# --- Constant κ ---
print("Calibrating constant κ...")
res_const = minimize(
    lambda p: joint_objective(p, kappa_const),
    x0=[0.05],
    bounds=[(0.001, 1.0)],
    method="L-BFGS-B",
)
kappa_const_opt = res_const.x[0]
print(f"  κ_const = {kappa_const_opt:.4f}")

# --- Depth-dependent κ(D) ---
print("\nCalibrating κ(D) = κ₀ + (1-κ₀)·D^γ...")
res_depth = minimize(
    lambda p: joint_objective(p, kappa_depth),
    x0=[0.01, 2.0],
    bounds=[(0.0001, 0.5), (0.5, 10.0)],
    method="L-BFGS-B",
)
kappa0_opt, gamma_opt = res_depth.x
print(f"  κ₀ = {kappa0_opt:.4f}, γ = {gamma_opt:.2f}")


# ============================================================================
# Compare results
# ============================================================================

print(f"\n{'=' * 80}")
print(f"{'Company':<10} {'Actual':>8} {'Raw':>8} {'κ=const':>8} {'κ(D)':>8} "
      f"{'Ratio_c':>8} {'Ratio_d':>8}")
print("─" * 80)

results_comparison = []
for name, c in companies.items():
    loss_raw, _ = compute_loss_with_kappa(c["facilities"], kappa_const, [1.0])
    loss_const, det_c = compute_loss_with_kappa(c["facilities"], kappa_const, [kappa_const_opt])
    loss_depth, det_d = compute_loss_with_kappa(c["facilities"], kappa_depth, [kappa0_opt, gamma_opt])
    actual = c["actual_loss_M"]

    r_const = loss_const / max(actual, 0.01)
    r_depth = loss_depth / max(actual, 0.01)

    results_comparison.append({
        "name": name, "actual": actual,
        "raw": loss_raw, "const": loss_const, "depth_dep": loss_depth,
        "ratio_const": r_const, "ratio_depth": r_depth,
        "details_const": det_c, "details_depth": det_d,
        "color": c["color"],
    })

    print(f"{name:<10} {actual:>7.0f}M {loss_raw:>7.0f}M {loss_const:>7.1f}M "
          f"{loss_depth:>7.1f}M {r_const:>7.1f}x {r_depth:>7.1f}x")

# log-error comparison
logerr_const = sum((np.log(r["const"]) - np.log(r["actual"]))**2 for r in results_comparison)
logerr_depth = sum((np.log(r["depth_dep"]) - np.log(r["actual"]))**2 for r in results_comparison)
improvement = (1 - logerr_depth / max(logerr_const, 1e-10)) * 100

print(f"\nLog-squared error: const={logerr_const:.3f}, κ(D)={logerr_depth:.3f}")
print(f"Improvement: {improvement:+.1f}%")


# ============================================================================
# Show κ(D) curve and facility-level detail
# ============================================================================

print(f"\n--- Facility-level κ values ---")
for r in results_comparison:
    print(f"\n  {r['name']}:")
    for dc, dd in zip(r["details_const"], r["details_depth"]):
        print(f"    {dc['name']:<20} DR_res={dc['dr_resid']:.3f} "
              f"κ_const={dc['kappa']:.4f} κ(D)={dd['kappa']:.4f} "
              f"({'↑' if dd['kappa'] > dc['kappa'] else '↓'} "
              f"{abs(dd['kappa']-dc['kappa'])/max(dc['kappa'],0.001)*100:.0f}%)")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# --- 1. κ(D) curve ---
ax1 = fig.add_subplot(gs[0, 0])
D_range = np.linspace(0, 1, 200)

ax1.plot(D_range, kappa_const(D_range, kappa_const_opt),
         "r--", linewidth=2, label=f"Constant κ={kappa_const_opt:.3f}")
ax1.plot(D_range, kappa_depth(D_range, kappa0_opt, gamma_opt),
         "b-", linewidth=2.5,
         label=f"κ(D) = {kappa0_opt:.3f} + {1-kappa0_opt:.3f}·D^{gamma_opt:.1f}")
ax1.fill_between(D_range,
                  kappa_const(D_range, kappa_const_opt),
                  kappa_depth(D_range, kappa0_opt, gamma_opt),
                  alpha=0.15, color="blue")

# mark facilities
for r in results_comparison:
    for d in r["details_depth"]:
        ax1.scatter(d["dr_resid"], d["kappa"], s=40,
                    color=r["color"], edgecolors="white", zorder=5)

ax1.set_xlabel("Residential Damage Ratio D")
ax1.set_ylabel("κ (sector adjustment)")
ax1.set_title("κ(D): Depth-Dependent Sector Transfer")
ax1.legend(fontsize=9)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# --- 2. Predicted vs Actual: 3 models ---
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(3)
width = 0.2

for i, r in enumerate(results_comparison):
    ax2.bar(i - width, r["const"], width, color=r["color"], alpha=0.4,
            edgecolor="black", label="κ=const" if i == 0 else "")
    ax2.bar(i, r["depth_dep"], width, color=r["color"], alpha=0.8,
            edgecolor="black", label="κ(D)" if i == 0 else "")
    ax2.bar(i + width, r["actual"], width, color="gray", alpha=0.5,
            edgecolor="black", hatch="//", label="Actual" if i == 0 else "")

ax2.set_xticks(x)
ax2.set_xticklabels([r["name"] for r in results_comparison])
ax2.set_ylabel("Loss ($M)")
ax2.set_title("Loss Prediction: const κ vs κ(D) vs Actual")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# --- 3. Pred/Actual ratios ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(x - 0.15, [r["ratio_const"] for r in results_comparison], 0.3,
        label="κ=const", color="lightcoral", alpha=0.7)
ax3.bar(x + 0.15, [r["ratio_depth"] for r in results_comparison], 0.3,
        label="κ(D)", color="steelblue", alpha=0.8)
ax3.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
ax3.axhspan(0.5, 2.0, alpha=0.1, color="green", label="Acceptable")
ax3.set_xticks(x)
ax3.set_xticklabels([r["name"] for r in results_comparison])
ax3.set_ylabel("Predicted / Actual")
ax3.set_title(f"Accuracy: κ(D) improves by {improvement:+.0f}%")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# --- 4. Effective DR: residential vs commercial ---
ax4 = fig.add_subplot(gs[1, 0])
depth_range = np.linspace(0, 5, 200)

# residential
dr_resid = np.array([float(model(d, 288)) for d in depth_range])
ax4.plot(depth_range, dr_resid, "k-", linewidth=2, label="Residential (FEMA)")

# commercial with κ(D)
dr_commercial = dr_resid * kappa_depth(dr_resid, kappa0_opt, gamma_opt)
ax4.plot(depth_range, dr_commercial, "b-", linewidth=2.5,
         label=f"Commercial κ(D) [κ₀={kappa0_opt:.3f}, γ={gamma_opt:.1f}]")

# commercial with κ=const
dr_const = dr_resid * kappa_const_opt
ax4.plot(depth_range, dr_const, "r--", linewidth=1.5,
         label=f"Commercial κ=const={kappa_const_opt:.3f}")

ax4.fill_between(depth_range, dr_const, dr_commercial,
                  alpha=0.15, color="blue", label="κ(D) vs κ=const gap")

ax4.set_xlabel("Flood Depth (m)")
ax4.set_ylabel("Damage Ratio")
ax4.set_title("Effective Damage: Residential vs Commercial")
ax4.legend(fontsize=8)
ax4.set_ylim(-0.05, 1.05)
ax4.grid(True, alpha=0.3)

# --- 5. κ by sector (with depth dependence) ---
ax5 = fig.add_subplot(gs[1, 1])

sectors_demo = {
    "Residential": (1.0, 1.0),       # κ always 1
    "Hospitality (SME)": (0.15, 1.5),
    "Healthcare": (kappa0_opt, gamma_opt),
    "Manufacturing": (0.005, 3.0),
    "Energy/Infra": (0.001, 4.0),
}

for sector, (k0, g) in sectors_demo.items():
    kappa_curve = kappa_depth(D_range, k0, g)
    ax5.plot(D_range, kappa_curve, linewidth=2, label=f"{sector} (κ₀={k0:.3f})")

ax5.set_xlabel("Residential Damage Ratio D")
ax5.set_ylabel("κ(D)")
ax5.set_title("Sector κ(D) Families (illustrative)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- 6. Summary ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

summary = (
    f"κ(D) Calibration Results\n"
    f"{'═' * 40}\n\n"
    f"Model: κ(D) = κ₀ + (1-κ₀)·D^γ\n"
    f"  κ₀ = {kappa0_opt:.4f} (base resilience)\n"
    f"  γ  = {gamma_opt:.2f} (convergence speed)\n\n"
    f"vs constant κ = {kappa_const_opt:.4f}\n\n"
    f"Log-error improvement: {improvement:+.0f}%\n\n"
    f"Physical interpretation:\n"
    f"{'─' * 40}\n"
    f"At D=0.1 (shallow): κ={float(kappa_depth(0.1, kappa0_opt, gamma_opt)):.3f}\n"
    f"  → commercial barely affected\n"
    f"At D=0.5 (moderate): κ={float(kappa_depth(0.5, kappa0_opt, gamma_opt)):.3f}\n"
    f"  → commercial starts taking damage\n"
    f"At D=0.9 (severe): κ={float(kappa_depth(0.9, kappa0_opt, gamma_opt)):.3f}\n"
    f"  → converging to residential\n"
    f"At D=1.0 (total): κ=1.0\n"
    f"  → destruction is universal\n\n"
    f"Key insight:\n"
    f"{'─' * 40}\n"
    f"Commercial buildings ARE more resilient\n"
    f"at low depths (concrete, elevation),\n"
    f"but this advantage erodes at high\n"
    f"severity. κ(D) captures this naturally.\n"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

plt.suptitle("Transfer Learning: κ(D) Depth-Dependent Sector Adjustment",
             fontsize=14, y=1.01)
plt.savefig("figures/19_kappa_depth.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → notebooks/figures/19_kappa_depth.png")
