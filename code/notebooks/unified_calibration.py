"""
Unified calibration: D(depth, duration) on FEMA NFIP claims + USACE anchor.

Reads:  data/processed/fema_*_calibration.csv
Writes: params/unified_params.csv
        figures/10_unified_calibration.png

Run from package root:  PYTHONPATH=code python code/notebooks/unified_calibration.py
"""

import sys, os

# package root = ../.. from this file (code/notebooks/unified_calibration.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "code"))

H_MIN = 0.05  # min flood depth for plotting (m)
FIG_DIR = os.path.join(ROOT, "figures")
PARAMS_DIR = os.path.join(ROOT, "params")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize, differential_evolution

from damage_functions.duration_richards import (
    DurationRichards, DurationRichardsParams, predict_from_vector
)


# ============================================================================
# Load + prepare data
# ============================================================================

DATA_DIR = os.path.join(ROOT, "data", "processed")
# Per-event depth caps chosen to match each event's empirical data range:
# Sandy 99.9th pct = 3.35m, Harvey 99.9th pct = 8.66m, Katrina 99.9th pct = 9.14m.
# We use slightly wider caps (Sandy/Harvey 4m, Katrina 10m) to retain the tail
# while excluding extreme outliers. This includes 99.91% of valid claims.
# A uniform 6m cap (previous default) discarded the Katrina levee-breach signal
# above 6m (1,216 claims, 0.9% of valid Katrina); event-tailored caps preserve it.
events_cfg = {
    "Sandy": {"file": os.path.join(DATA_DIR, "fema_sandy_calibration.csv"), "duration_h": 36,
              "color": "steelblue", "max_depth": 4.0},
    "Harvey": {"file": os.path.join(DATA_DIR, "fema_harvey_calibration.csv"), "duration_h": 288,
               "color": "darkred", "max_depth": 4.0},
    "Katrina": {"file": os.path.join(DATA_DIR, "fema_katrina_calibration.csv"), "duration_h": 120,
                "color": "darkorange", "max_depth": 10.0},
}

# USACE anchor (instantaneous, τ=0)
usace_depth = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.4]) # meters
usace_dr = np.array([0.08, 0.18, 0.27, 0.35, 0.43, 0.48, 0.55])

TARGET_BINS = 12  # equal bins per event


def prepare_event(name, cfg):
    """Load and bin one event, return equal-sized bin arrays."""
    cal = pd.read_csv(cfg["file"])

    # apply depth cap
    cal = cal[cal["depth_m"] <= cfg["max_depth"]]

    depth = cal["depth_m"].values
    dr = cal["damage_ratio"].values

    # bin
    edges = np.linspace(0, cfg["max_depth"], TARGET_BINS + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, TARGET_BINS - 1)

    bin_d = np.array([depth[idx == i].mean() if (idx == i).sum() > 0 else np.nan
                      for i in range(TARGET_BINS)])
    bin_dr = np.array([dr[idx == i].mean() if (idx == i).sum() > 0 else np.nan
                       for i in range(TARGET_BINS)])
    bin_std = np.array([dr[idx == i].std() if (idx == i).sum() > 1 else 0.2
                        for i in range(TARGET_BINS)])
    bin_n = np.array([(idx == i).sum() for i in range(TARGET_BINS)])

    valid = bin_n > 10
    return {
        "depth": bin_d[valid],
        "dr": bin_dr[valid],
        "std": bin_std[valid],
        "sigma": np.maximum(bin_std[valid] / np.sqrt(bin_n[valid]), 0.015),
        "count": bin_n[valid],
        "duration_h": cfg["duration_h"],
        "n_total": len(cal),
        "n_original": len(pd.read_csv(cfg["file"])),
        "dr_median": np.median(dr),
    }


event_data = {}
for name, cfg in events_cfg.items():
    event_data[name] = prepare_event(name, cfg)
    d = event_data[name]
    print(f"{name}: {d['n_total']:,} claims (≤{cfg['max_depth']}m), "
          f"{len(d['depth'])} bins, DR median={d['dr_median']:.3f}")

# add USACE as pseudo-event at τ=0
event_data["USACE"] = {
    "depth": usace_depth,
    "dr": usace_dr,
    "sigma": np.full_like(usace_dr, 0.03),  # tight — high confidence in USACE
    "duration_h": 0,
    "n_total": 9,
    "dr_median": np.median(usace_dr),
}
print(f"USACE: {len(usace_depth)} points at τ=0h (anchor)")


# ============================================================================
# Calibration
# ============================================================================

USACE_WEIGHT = 3.0  # upweight USACE anchor (it's the only τ=0 data)

def objective(theta):
    """Weighted sum of squared errors across all events."""
    total = 0.0

    for name, data in event_data.items():
        try:
            pred = predict_from_vector(theta, data["depth"], data["duration_h"])
        except:
            return 1e10

        residuals = data["dr"] - pred
        weight = USACE_WEIGHT if name == "USACE" else 1.0
        total += weight * np.sum((residuals / data["sigma"]) ** 2)

    # soft prior
    B, Q0, a0, nu0, lam_a, lam_Q, d_nu = theta
    total += 0.5 * ((B - 3.0) / 3.0) ** 2
    total += 0.5 * ((Q0 - 1.0) / 1.0) ** 2
    total += 0.5 * ((a0 - 0.5) / 0.2) ** 2
    total += 0.5 * ((nu0 - 0.5) / 0.3) ** 2

    return total


print("\n--- Differential Evolution (global optimizer) ---")

bounds = [
    (0.5, 12),    # B
    (0.05, 5),    # Q_0
    (0.1, 0.9),   # a_0
    (0.1, 3.0),   # nu_0
    (0.0, 3.0),   # lambda_a
    (0.0, 3.0),   # lambda_Q
    (0.0, 2.0),   # delta_nu
]

result = differential_evolution(
    objective, bounds,
    seed=42, maxiter=1000, tol=1e-10,
    polish=True, workers=1,
)

theta_best = result.x
param_names = DurationRichardsParams.PARAM_NAMES

print(f"Converged: {result.success}, obj={result.fun:.2f}")
print(f"\nParameters:")
for pname, val in zip(param_names, theta_best):
    print(f"  {pname:>5} = {val:.4f}")

# create model
best_params = DurationRichardsParams.from_vector(theta_best)
model = DurationRichards(best_params)

# per-event quality
print(f"\nPer-event fit quality:")
for name, data in event_data.items():
    pred = model(data["depth"], data["duration_h"])
    mae = np.mean(np.abs(data["dr"] - pred))
    ss_res = np.sum((data["dr"] - pred) ** 2)
    ss_tot = np.sum((data["dr"] - data["dr"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"  {name:>10}: MAE={mae:.4f}, R²={r2:.4f} (τ={data['duration_h']}h)")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

xi_plot = np.linspace(H_MIN, 7, 200)

# --- 1. Main: all events + fitted curves ---
ax1 = fig.add_subplot(gs[0, :2])

event_colors = {"Sandy": "steelblue", "Harvey": "darkred",
                "Katrina": "darkorange", "USACE": "green"}
event_markers = {"Sandy": "o", "Harvey": "s", "Katrina": "D", "USACE": "^"}

for name, data in event_data.items():
    tau = data["duration_h"]
    c = event_colors[name]
    m = event_markers[name]

    ax1.scatter(data["depth"], data["dr"], s=50, color=c, marker=m,
                alpha=0.7, edgecolors="white", linewidth=0.5,
                label=f'{name} (τ={tau}h)', zorder=4)

    D_fit = model(xi_plot, tau)
    days = tau / 24
    ls = "--" if name == "USACE" else "-"
    lw = 1.5 if name == "USACE" else 2.5
    ax1.plot(xi_plot, D_fit, color=c, linewidth=lw, linestyle=ls,
             label=f'{name} fit ({days:.0f}d)' if tau > 0 else f'{name} fit (instant)')

ax1.set_xlabel("Flood Depth (m)", fontsize=12)
ax1.set_ylabel("Damage Ratio", fontsize=12)
ax1.set_title("Unified Duration-Richards: One Model, All Events", fontsize=13)
ax1.legend(fontsize=8, ncol=2, loc="lower right")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, alpha=0.3)

# --- 2. Summary panel ---
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")

total_claims = sum(d["n_total"] for d in event_data.values())
summary = (
    f"Unified Calibration\n"
    f"{'═' * 38}\n"
    f"Events: USACE + Sandy + Harvey + Katrina\n"
    f"Total: {total_claims:,} data points\n"
    f"Depth cap: ≤ 6m\n"
    f"USACE weight: {USACE_WEIGHT}x (τ=0 anchor)\n\n"
    f"Duration-Richards Parameters\n"
    f"{'─' * 38}\n"
)
for pname, val in zip(param_names, theta_best):
    summary += f"  {pname:>5} = {val:.4f}\n"
summary += f"\nPer-Event Quality\n{'─' * 38}\n"
for name, data in event_data.items():
    pred = model(data["depth"], data["duration_h"])
    mae = np.mean(np.abs(data["dr"] - pred))
    ss_res = np.sum((data["dr"] - pred) ** 2)
    ss_tot = np.sum((data["dr"] - data["dr"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    summary += f"  {name:>8}: MAE={mae:.3f} R²={r2:.3f}\n"

dr_0h = float(model(1.0, 0))
dr_36h = float(model(1.0, 36))
dr_120h = float(model(1.0, 120))
dr_288h = float(model(1.0, 288))
summary += (
    f"\nDuration Effect at 1m depth\n"
    f"{'─' * 38}\n"
    f"  τ=0h (instant):  DR={dr_0h:.3f}\n"
    f"  τ=36h (Sandy):   DR={dr_36h:.3f}\n"
    f"  τ=120h (Katrina):DR={dr_120h:.3f}\n"
    f"  τ=288h (Harvey): DR={dr_288h:.3f}\n"
    f"  Ratio max/min:   {dr_288h/max(dr_0h, 0.001):.1f}x\n"
)
ax2.text(0.05, 0.95, summary, transform=ax2.transAxes,
         fontsize=9, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# --- 3. Duration family ---
ax3 = fig.add_subplot(gs[1, 0])
model.plot_duration_family(
    durations_h=[0, 12, 36, 72, 120, 288, 504],
    ax=ax3,
)
# mark events on family
for name, data in event_data.items():
    if name == "USACE":
        continue
    tau = data["duration_h"]
    D_at_2m = float(model(2.0, tau))
    ax3.plot(2.0, D_at_2m, marker=event_markers[name], color=event_colors[name],
             markersize=10, zorder=5, markeredgecolor="white")

# --- 4. Residuals ---
ax4 = fig.add_subplot(gs[1, 1])
for name, data in event_data.items():
    pred = model(data["depth"], data["duration_h"])
    residuals = data["dr"] - pred
    ax4.scatter(data["depth"], residuals, s=25, color=event_colors[name],
                marker=event_markers[name], alpha=0.7, label=name)
ax4.axhline(0, color="black", linewidth=0.5)
ax4.set_xlabel("Flood Depth (m)")
ax4.set_ylabel("Residual")
ax4.set_title("Residuals by Event")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# --- 5. Duration sensitivity at fixed depths ---
ax5 = fig.add_subplot(gs[1, 2])
dur_range = np.linspace(0, 400, 200)
for d_val, ls in [(0.5, ":"), (1.0, "--"), (2.0, "-"), (3.0, "-"), (5.0, "-")]:
    D_tau = np.array([float(model(d_val, t)) for t in dur_range])
    ax5.plot(dur_range / 24, D_tau, linewidth=2, linestyle=ls,
             label=f"depth={d_val}m")
for name in ["Sandy", "Harvey", "Katrina"]:
    tau = event_data[name]["duration_h"]
    ax5.axvline(tau / 24, color=event_colors[name], linestyle=":", alpha=0.5)
    ax5.text(tau / 24, 0.02, name, color=event_colors[name],
             fontsize=8, rotation=90, va="bottom")
ax5.set_xlabel("Duration (days)")
ax5.set_ylabel("Damage Ratio")
ax5.set_title("Duration Effect at Fixed Depths")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- 6. Parameter sensitivity (effective params vs duration) ---
ax6 = fig.add_subplot(gs[2, 0])
tau_range = np.linspace(0, 500, 200)
a_eff = np.array([model.effective_params(t)[1] for t in tau_range])
Q_eff = np.array([model.effective_params(t)[0] for t in tau_range])
nu_eff = np.array([model.effective_params(t)[2] for t in tau_range])

ax6.plot(tau_range / 24, a_eff / best_params.a_0, "b-", linewidth=2,
         label="a(τ)/a₀ (inflection shift)")
ax6.plot(tau_range / 24, Q_eff / max(best_params.Q_0, 0.001), "r--", linewidth=2,
         label="Q(τ)/Q₀ (onset speed)")
ax6.plot(tau_range / 24, nu_eff / max(best_params.nu_0, 0.001), "g:", linewidth=2,
         label="ν(τ)/ν₀ (asymmetry)")

for name in ["Sandy", "Harvey", "Katrina"]:
    tau = event_data[name]["duration_h"]
    ax6.axvline(tau / 24, color=event_colors[name], linestyle=":", alpha=0.4)
ax6.set_xlabel("Duration (days)")
ax6.set_ylabel("Relative parameter value")
ax6.set_title("How Duration Modifies Richards Shape")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# --- 7. 3D surface: D(depth, duration) ---
ax7 = fig.add_subplot(gs[2, 1])
depth_grid = np.linspace(0, 6, 60)
dur_grid = np.linspace(0, 350, 60)
D_surface = np.zeros((60, 60))
for i, d in enumerate(depth_grid):
    for j, t in enumerate(dur_grid):
        D_surface[j, i] = float(model(d, t))

im = ax7.contourf(depth_grid, dur_grid / 24, D_surface,
                    levels=20, cmap="YlOrRd")
plt.colorbar(im, ax=ax7, label="Damage Ratio")

# mark events
for name in ["Sandy", "Harvey", "Katrina"]:
    data = event_data[name]
    med_depth = np.median(data["depth"])
    ax7.plot(med_depth, data["duration_h"] / 24, marker=event_markers[name],
             color="white", markersize=12, markeredgecolor="black", zorder=5)
    ax7.annotate(name, (med_depth, data["duration_h"] / 24),
                 textcoords="offset points", xytext=(8, 0), fontsize=9,
                 color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

ax7.set_xlabel("Flood Depth (m)")
ax7.set_ylabel("Duration (days)")
ax7.set_title("Unified Damage Surface D(depth, τ)")
ax7.grid(True, alpha=0.2)

# --- 8. Comparison: with vs without duration ---
ax8 = fig.add_subplot(gs[2, 2])

# model at τ=0 (no duration effect)
D_nodur = model(xi_plot, 0)

for name in ["Sandy", "Harvey", "Katrina"]:
    data = event_data[name]
    tau = data["duration_h"]
    D_withdur = model(xi_plot, tau)

    ax8.plot(xi_plot, D_withdur, color=event_colors[name], linewidth=2,
             label=f'{name} (τ={tau}h)')

ax8.plot(xi_plot, D_nodur, "k--", linewidth=1.5, alpha=0.5,
         label="No duration (τ=0)")
ax8.fill_between(xi_plot, D_nodur, model(xi_plot, 288),
                  alpha=0.1, color="red", label="Duration gap (0→288h)")

ax8.set_xlabel("Flood Depth (m)")
ax8.set_ylabel("Damage Ratio")
ax8.set_title("Impact of Duration Extension")
ax8.legend(fontsize=8)
ax8.set_ylim(-0.05, 1.05)
ax8.grid(True, alpha=0.3)

plt.suptitle(
    f"Unified D(depth, duration): {total_claims:,} Claims + USACE Anchor",
    fontsize=14, y=1.01
)
fig_path = os.path.join(FIG_DIR, "10_unified_calibration.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFigure  → {fig_path}")

params_path = os.path.join(PARAMS_DIR, "unified_params.csv")
pd.DataFrame({
    "parameter": param_names,
    "value": theta_best,
}).to_csv(params_path, index=False)
print(f"Params  → {params_path}")
