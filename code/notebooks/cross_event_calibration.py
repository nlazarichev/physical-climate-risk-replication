"""
Cross-event calibration: Sandy vs Harvey vs Katrina.

Fits duration-dependent Richards jointly on 3 events
with different duration profiles.

Duration assignments (from FEMA/literature):
  Sandy (2012):   ~24-48h (tidal surge, fast recession)
  Harvey (2017):  ~240-336h (2+ weeks rain flooding)
  Katrina (2005): ~72-168h (levee breach, pump-out over days)

Run: cd physical-climate-risk && PYTHONPATH=. python notebooks/cross_event_calibration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

from damage_functions.duration_richards import (
    DurationRichards, DurationRichardsParams, predict_from_vector
)

os.makedirs("notebooks/figures", exist_ok=True)

# ============================================================================
# Load all three events
# ============================================================================

events = {
    "Sandy": {
        "file": "data/processed/fema_sandy_calibration.csv",
        "duration_h": 36,      # typical: 24-48h tidal surge
        "color": "steelblue",
        "marker": "o",
    },
    "Harvey": {
        "file": "data/processed/fema_harvey_calibration.csv",
        "duration_h": 288,     # typical: ~12 days rain flooding
        "color": "darkred",
        "marker": "s",
    },
    "Katrina": {
        "file": "data/processed/fema_katrina_calibration.csv",
        "duration_h": 120,     # typical: ~5 days (levee breach + pump-out)
        "color": "darkorange",
        "marker": "D",
    },
}

# Bin each event
n_bins = 30

event_data = {}
for name, cfg in events.items():
    cal = pd.read_csv(cfg["file"])
    depth = cal["depth_m"].values
    dr = cal["damage_ratio"].values

    edges = np.linspace(0, 10, n_bins + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, n_bins - 1)

    bin_d = np.array([depth[idx == i].mean() for i in range(n_bins)])
    bin_dr = np.array([dr[idx == i].mean() for i in range(n_bins)])
    bin_n = np.array([(idx == i).sum() for i in range(n_bins)])

    valid = bin_n > 20
    event_data[name] = {
        "depth": bin_d[valid],
        "dr": bin_dr[valid],
        "count": bin_n[valid],
        "sigma": np.maximum(
            np.array([dr[idx == i].std() for i in range(n_bins)])[valid]
            / np.sqrt(bin_n[valid]),
            0.02
        ),
        "duration_h": cfg["duration_h"],
        "n_total": len(cal),
        "dr_median": np.median(dr),
    }
    print(f"{name}: {len(cal):,} claims, {valid.sum()} bins, "
          f"DR median={np.median(dr):.3f}, assigned τ={cfg['duration_h']}h")

# ============================================================================
# Joint calibration: fit 7 parameters to all 3 events simultaneously
# ============================================================================

def joint_log_likelihood(theta):
    """Log-likelihood across all events."""
    ll = 0.0
    for name, data in event_data.items():
        try:
            pred = predict_from_vector(theta, data["depth"], data["duration_h"])
        except (ValueError, FloatingPointError):
            return np.inf
        residuals = data["dr"] - pred
        ll += np.sum((residuals / data["sigma"]) ** 2)
    return ll


def joint_log_prior(theta):
    """Prior on duration-Richards parameters."""
    B, Q0, a0, nu0, lam_a, lam_Q, d_nu = theta

    if B <= 0.01 or Q0 <= 0 or a0 <= 0 or a0 >= 1 or nu0 <= 0:
        return np.inf
    if lam_a < 0 or lam_Q < 0 or d_nu < 0:
        return np.inf
    if B > 15 or Q0 > 10 or nu0 > 5 or lam_a > 5 or lam_Q > 5 or d_nu > 3:
        return np.inf

    lp = 0.0
    lp += 0.5 * ((B - 3.0) / 2.0) ** 2
    lp += 0.5 * ((Q0 - 1.0) / 1.0) ** 2
    lp += 0.5 * ((a0 - 0.45) / 0.15) ** 2
    lp += 0.5 * ((nu0 - 0.4) / 0.3) ** 2
    lp += 0.5 * ((lam_a - 0.5) / 0.5) ** 2
    lp += 0.5 * ((lam_Q - 0.3) / 0.3) ** 2
    lp += 0.5 * ((d_nu - 0.3) / 0.3) ** 2
    return lp


def objective(theta):
    return joint_log_likelihood(theta) + joint_log_prior(theta)


print("\n--- Joint optimization (Nelder-Mead) ---")

# multi-start to avoid local minima
best_result = None
best_obj = np.inf

starts = [
    [2.0, 0.8, 0.45, 0.4, 0.5, 0.3, 0.3],
    [3.0, 1.5, 0.35, 0.3, 0.8, 0.5, 0.5],
    [1.5, 0.5, 0.50, 0.5, 0.3, 0.2, 0.2],
    [4.0, 1.0, 0.40, 0.6, 0.4, 0.4, 0.4],
    [1.0, 2.0, 0.30, 0.2, 1.0, 0.6, 0.6],
]

for i, x0 in enumerate(starts):
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8})
    print(f"  Start {i+1}: obj={res.fun:.2f}, converged={res.success}")
    if res.fun < best_obj:
        best_obj = res.fun
        best_result = res

theta_best = best_result.x
param_names = DurationRichardsParams.PARAM_NAMES

print(f"\n--- Best fit ---")
for name, val in zip(param_names, theta_best):
    print(f"  {name:>5} = {val:.4f}")

# create model
best_params = DurationRichardsParams.from_vector(theta_best)
model = DurationRichards(best_params)

# compute per-event MAE
print(f"\nPer-event fit quality:")
for name, data in event_data.items():
    pred = model(data["depth"], data["duration_h"])
    mae = np.mean(np.abs(data["dr"] - pred))
    r2 = 1 - np.sum((data["dr"] - pred)**2) / np.sum((data["dr"] - data["dr"].mean())**2)
    print(f"  {name:>10}: MAE={mae:.4f}, R²={r2:.4f} (τ={data['duration_h']}h)")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# --- 1. All three events: data + fitted curves ---
ax1 = fig.add_subplot(gs[0, 0:2])

xi_plot = np.linspace(0, 8, 200)

for name, cfg in events.items():
    data = event_data[name]
    tau = data["duration_h"]
    color = cfg["color"]
    marker = cfg["marker"]

    # data points
    ax1.scatter(data["depth"], data["dr"], s=30, color=color, marker=marker,
                alpha=0.6, label=f'{name} data (τ={tau}h)', zorder=4)

    # fitted curve
    D_fit = model(xi_plot, tau)
    days = tau / 24
    ax1.plot(xi_plot, D_fit, color=color, linewidth=2.5,
             label=f'{name} fit ({days:.0f}d)')

# USACE reference
usace_depth = np.array([0, 1, 2, 3, 4, 5, 8]) * 0.3048
usace_dr = np.array([0.08, 0.18, 0.27, 0.35, 0.43, 0.48, 0.55])
ax1.scatter(usace_depth, usace_dr, s=100, c='lime', edgecolors='darkgreen',
            marker='D', zorder=5, label='USACE reference (instant)')
D_instant = model(xi_plot, 0)
ax1.plot(xi_plot, D_instant, 'g--', linewidth=1.5, alpha=0.5,
         label='Model at τ=0 (instant)')

ax1.set_xlabel("Flood Depth (m)", fontsize=12)
ax1.set_ylabel("Damage Ratio", fontsize=12)
ax1.set_title("Cross-Event Fit: Duration-Dependent Richards", fontsize=13)
ax1.legend(fontsize=8, ncol=2, loc="lower right")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# --- 2. Summary ---
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")

summary = (
    f"Cross-Event Calibration\n"
    f"{'═' * 38}\n"
    f"Events: Sandy + Harvey + Katrina\n"
    f"Total claims: {sum(d['n_total'] for d in event_data.values()):,}\n\n"
    f"Duration-Richards Parameters\n"
    f"{'─' * 38}\n"
)
for pname, val in zip(param_names, theta_best):
    summary += f"  {pname:>5} = {val:.4f}\n"
summary += f"\nPer-Event Quality\n{'─' * 38}\n"
for name, data in event_data.items():
    pred = model(data["depth"], data["duration_h"])
    mae = np.mean(np.abs(data["dr"] - pred))
    r2 = 1 - np.sum((data["dr"] - pred)**2) / np.sum((data["dr"] - data["dr"].mean())**2)
    summary += f"  {name:>8}: MAE={mae:.3f} R²={r2:.3f}\n"

summary += (
    f"\nDuration Effect\n"
    f"{'─' * 38}\n"
    f"  DR at 1m, τ=0h:   {float(model(1.0, 0)):.3f}\n"
    f"  DR at 1m, τ=36h:  {float(model(1.0, 36)):.3f}\n"
    f"  DR at 1m, τ=288h: {float(model(1.0, 288)):.3f}\n"
    f"  Ratio 288h/0h:    {float(model(1.0, 288))/max(float(model(1.0, 0)), 0.001):.1f}x\n"
)
ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
         fontsize=9.5, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# --- 3. Duration family of curves ---
ax3 = fig.add_subplot(gs[1, 0])
model.plot_duration_family(
    durations_h=[0, 12, 36, 72, 120, 288, 504],
    ax=ax3,
)

# --- 4. Residuals by event ---
ax4 = fig.add_subplot(gs[1, 1])
for name, cfg in events.items():
    data = event_data[name]
    pred = model(data["depth"], data["duration_h"])
    residuals = data["dr"] - pred
    ax4.scatter(data["depth"], residuals, s=15, color=cfg["color"],
                alpha=0.6, label=name)
ax4.axhline(0, color="black", linewidth=0.5)
ax4.set_xlabel("Flood Depth (m)")
ax4.set_ylabel("Residual (observed - predicted)")
ax4.set_title("Residuals by Event")
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- 5. Duration effect at fixed depths ---
ax5 = fig.add_subplot(gs[1, 2])
durations_range = np.linspace(0, 400, 200)
for depth_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
    D_tau = np.array([float(model(depth_val, t)) for t in durations_range])
    ax5.plot(durations_range / 24, D_tau, linewidth=2, label=f"depth={depth_val}m")

# mark events
for name, cfg in events.items():
    tau = event_data[name]["duration_h"]
    ax5.axvline(tau / 24, color=cfg["color"], linestyle=":", alpha=0.5)
    ax5.text(tau / 24 + 0.3, 0.02, name, color=cfg["color"], fontsize=8, rotation=90)

ax5.set_xlabel("Duration (days)")
ax5.set_ylabel("Damage Ratio")
ax5.set_title("Duration Effect at Fixed Depths")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

plt.suptitle(
    f"Cross-Event Calibration: {sum(d['n_total'] for d in event_data.values()):,} "
    f"FEMA Claims (Sandy + Harvey + Katrina)",
    fontsize=14, y=1.01
)
plt.savefig("figures/09_cross_event_calibration.png",
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → notebooks/figures/09_cross_event_calibration.png")

# save parameters
pd.DataFrame({
    "parameter": param_names,
    "value": theta_best,
}).to_csv("data/duration_richards_params.csv", index=False)
print("Parameters → data/duration_richards_params.csv")
