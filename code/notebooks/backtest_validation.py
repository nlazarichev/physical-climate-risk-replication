"""
Out-of-sample validation of Duration-Richards.

Three validation strategies:
  1. Leave-one-event-out: train on 2 events → predict 3rd
  2. Random 70/30 split within each event
  3. Comparison with baselines (linear, USACE-only, no-duration Richards)

Run: cd physical-climate-risk && PYTHONPATH=. python notebooks/backtest_validation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution

from damage_functions.duration_richards import (
    DurationRichards, DurationRichardsParams, predict_from_vector
)

os.makedirs("notebooks/figures", exist_ok=True)
np.random.seed(42)

# ============================================================================
# Load raw claim-level data (not binned — for proper splitting)
# ============================================================================

events_cfg = {
    "Sandy":   {"file": "data/processed/fema_sandy_calibration.csv",   "duration_h": 36,
                "color": "steelblue"},
    "Harvey":  {"file": "data/processed/fema_harvey_calibration.csv",  "duration_h": 288,
                "color": "darkred"},
    "Katrina": {"file": "data/processed/fema_katrina_calibration.csv", "duration_h": 120,
                "color": "darkorange"},
}

MAX_DEPTH = 6.0  # same cap as unified calibration

raw_data = {}
for name, cfg in events_cfg.items():
    df = pd.read_csv(cfg["file"])
    df = df[df["depth_m"] <= MAX_DEPTH].copy()
    raw_data[name] = {
        "depth": df["depth_m"].values,
        "dr": df["damage_ratio"].values,
        "duration_h": cfg["duration_h"],
        "color": cfg["color"],
    }
    print(f"{name}: {len(df):,} claims")

# USACE anchor (always in training)
usace_depth = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.4])
usace_dr = np.array([0.08, 0.18, 0.27, 0.35, 0.43, 0.48, 0.55])

# ============================================================================
# Helper: bin data for calibration (optimizer needs binned, not 60K raw points)
# ============================================================================

def bin_data(depth, dr, n_bins=15, min_count=20):
    edges = np.linspace(0, MAX_DEPTH, n_bins + 1)
    idx = np.clip(np.digitize(depth, edges) - 1, 0, n_bins - 1)

    bin_d, bin_dr, bin_sigma = [], [], []
    for i in range(n_bins):
        mask = idx == i
        if mask.sum() < min_count:
            continue
        bin_d.append(depth[mask].mean())
        bin_dr.append(dr[mask].mean())
        bin_sigma.append(max(dr[mask].std() / np.sqrt(mask.sum()), 0.015))

    return np.array(bin_d), np.array(bin_dr), np.array(bin_sigma)


# ============================================================================
# Calibration function
# ============================================================================

BOUNDS = [
    (0.5, 12),    # B
    (0.05, 5),    # Q_0
    (0.1, 0.9),   # a_0
    (0.1, 3.0),   # nu_0
    (0.0, 3.0),   # lambda_a
    (0.0, 3.0),   # lambda_Q
    (0.0, 2.0),   # delta_nu
]


def calibrate(train_events, usace_weight=3.0):
    """Calibrate duration-Richards on training events + USACE anchor."""

    # prepare training bins
    train_bins = {}
    for name, data in train_events.items():
        bd, bdr, bsig = bin_data(data["depth"], data["dr"])
        train_bins[name] = {
            "depth": bd, "dr": bdr, "sigma": bsig,
            "duration_h": data["duration_h"],
        }

    # add USACE
    train_bins["USACE"] = {
        "depth": usace_depth, "dr": usace_dr,
        "sigma": np.full_like(usace_dr, 0.03),
        "duration_h": 0,
    }

    def objective(theta):
        total = 0.0
        for name, bins in train_bins.items():
            try:
                pred = predict_from_vector(theta, bins["depth"], bins["duration_h"])
            except:
                return 1e10
            residuals = bins["dr"] - pred
            w = usace_weight if name == "USACE" else 1.0
            total += w * np.sum((residuals / bins["sigma"]) ** 2)

        # soft prior
        B, Q0, a0, nu0, la, lq, dn = theta
        total += 0.3 * ((B - 3) / 3) ** 2
        total += 0.3 * ((Q0 - 1) / 1) ** 2
        total += 0.3 * ((a0 - 0.5) / 0.2) ** 2
        total += 0.3 * ((nu0 - 0.5) / 0.3) ** 2
        return total

    res = differential_evolution(objective, BOUNDS, seed=42,
                                  maxiter=500, tol=1e-8, polish=True)
    params = DurationRichardsParams.from_vector(res.x)
    model = DurationRichards(params)
    return model, res.x


def evaluate(model, depth, dr_true, duration_h):
    """Compute metrics on a dataset."""
    pred = model(depth, duration_h)
    residuals = dr_true - pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((dr_true - dr_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
    bias = np.mean(residuals)
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "n": len(dr_true)}


# ============================================================================
# BASELINE MODELS
# ============================================================================

def linear_model(depth, dr_true):
    """Simple linear: DR = a + b * depth, capped at [0, 1]."""
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(depth, dr_true, 1)
    def predict(d, tau=None):
        return np.clip(coeffs[0] + coeffs[1] * np.asarray(d), 0, 1)
    return predict


def usace_lookup(depth, tau=None):
    """USACE table lookup with linear interpolation."""
    return np.interp(np.asarray(depth), usace_depth, usace_dr,
                     left=usace_dr[0], right=usace_dr[-1])


# ============================================================================
# TEST 1: Leave-one-event-out
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: Leave-One-Event-Out Cross-Validation")
print("=" * 70)

loo_results = {}

for test_name in ["Sandy", "Harvey", "Katrina"]:
    train_names = [n for n in raw_data if n != test_name]
    train_events = {n: raw_data[n] for n in train_names}

    print(f"\n  Train: {', '.join(train_names)} → Test: {test_name}")

    # calibrate on train
    model, theta = calibrate(train_events)

    # evaluate on test (raw claim-level data)
    test = raw_data[test_name]
    metrics = evaluate(model, test["depth"], test["dr"], test["duration_h"])

    # baselines on test
    # linear trained on train data
    all_train_depth = np.concatenate([raw_data[n]["depth"] for n in train_names])
    all_train_dr = np.concatenate([raw_data[n]["dr"] for n in train_names])
    linear = linear_model(all_train_depth, all_train_dr)
    metrics_linear = evaluate(
        type('', (), {'__call__': lambda self, d, t=None: linear(d)})(),
        test["depth"], test["dr"], test["duration_h"]
    )

    # USACE (no duration, no calibration needed)
    metrics_usace = evaluate(
        type('', (), {'__call__': lambda self, d, t=None: usace_lookup(d)})(),
        test["depth"], test["dr"], test["duration_h"]
    )

    loo_results[test_name] = {
        "duration_richards": metrics,
        "linear": metrics_linear,
        "usace": metrics_usace,
        "theta": theta,
    }

    print(f"  Duration-Richards: MAE={metrics['mae']:.4f}, "
          f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, "
          f"bias={metrics['bias']:.4f}")
    print(f"  Linear baseline:  MAE={metrics_linear['mae']:.4f}, "
          f"RMSE={metrics_linear['rmse']:.4f}, R²={metrics_linear['r2']:.4f}")
    print(f"  USACE lookup:     MAE={metrics_usace['mae']:.4f}, "
          f"RMSE={metrics_usace['rmse']:.4f}, R²={metrics_usace['r2']:.4f}")


# ============================================================================
# TEST 2: Random 70/30 split within each event
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: Random 70/30 Split Within Each Event")
print("=" * 70)

N_SPLITS = 5
split_results = {name: [] for name in raw_data}

for split_i in range(N_SPLITS):
    train_events_split = {}
    test_sets = {}

    for name, data in raw_data.items():
        n = len(data["depth"])
        idx = np.random.permutation(n)
        n_train = int(0.7 * n)

        train_events_split[name] = {
            "depth": data["depth"][idx[:n_train]],
            "dr": data["dr"][idx[:n_train]],
            "duration_h": data["duration_h"],
        }
        test_sets[name] = {
            "depth": data["depth"][idx[n_train:]],
            "dr": data["dr"][idx[n_train:]],
            "duration_h": data["duration_h"],
        }

    model, _ = calibrate(train_events_split)

    for name, test in test_sets.items():
        metrics = evaluate(model, test["depth"], test["dr"], test["duration_h"])
        split_results[name].append(metrics)

print(f"\n  Results over {N_SPLITS} random splits:")
for name in raw_data:
    maes = [m["mae"] for m in split_results[name]]
    rmses = [m["rmse"] for m in split_results[name]]
    r2s = [m["r2"] for m in split_results[name]]
    print(f"  {name:>10}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}, "
          f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}, "
          f"R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}")


# ============================================================================
# TEST 3: Train on Sandy+USACE only → predict Harvey & Katrina
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: Train on Sandy + USACE → Predict Harvey & Katrina")
print("=" * 70)

model_sandy, _ = calibrate({"Sandy": raw_data["Sandy"]})

for test_name in ["Harvey", "Katrina"]:
    test = raw_data[test_name]
    m = evaluate(model_sandy, test["depth"], test["dr"], test["duration_h"])
    m_usace = evaluate(
        type('', (), {'__call__': lambda self, d, t=None: usace_lookup(d)})(),
        test["depth"], test["dr"], test["duration_h"]
    )
    improvement = (m_usace["mae"] - m["mae"]) / m_usace["mae"] * 100
    print(f"  {test_name}: Duration-Richards MAE={m['mae']:.4f}, "
          f"USACE MAE={m_usace['mae']:.4f}, "
          f"improvement={improvement:+.1f}%")


# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# --- 1. LOO: comparison bar chart ---
ax1 = fig.add_subplot(gs[0, 0])
test_events = list(loo_results.keys())
x = np.arange(len(test_events))
width = 0.25

maes_dr = [loo_results[e]["duration_richards"]["mae"] for e in test_events]
maes_lin = [loo_results[e]["linear"]["mae"] for e in test_events]
maes_us = [loo_results[e]["usace"]["mae"] for e in test_events]

ax1.bar(x - width, maes_us, width, label="USACE lookup", color="green", alpha=0.7)
ax1.bar(x, maes_lin, width, label="Linear", color="gray", alpha=0.7)
ax1.bar(x + width, maes_dr, width, label="Duration-Richards", color="darkred", alpha=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(test_events)
ax1.set_ylabel("MAE (out-of-sample)")
ax1.set_title("Leave-One-Event-Out: MAE Comparison")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# --- 2. LOO: predicted vs observed scatter ---
ax2 = fig.add_subplot(gs[0, 1])
for test_name, res in loo_results.items():
    test = raw_data[test_name]
    theta = res["theta"]
    params = DurationRichardsParams.from_vector(theta)
    model = DurationRichards(params)

    # subsample for plotting (too many points)
    idx = np.random.choice(len(test["depth"]), min(2000, len(test["depth"])), replace=False)
    pred = model(test["depth"][idx], test["duration_h"])
    ax2.scatter(test["dr"][idx], pred, s=3, alpha=0.3,
                color=events_cfg[test_name]["color"], label=test_name)

ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")
ax2.set_xlabel("Observed DR")
ax2.set_ylabel("Predicted DR")
ax2.set_title("LOO: Predicted vs Observed")
ax2.legend(fontsize=8)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

# --- 3. LOO: residual distributions ---
ax3 = fig.add_subplot(gs[0, 2])
for test_name, res in loo_results.items():
    test = raw_data[test_name]
    theta = res["theta"]
    params = DurationRichardsParams.from_vector(theta)
    model = DurationRichards(params)
    pred = model(test["depth"], test["duration_h"])
    residuals = test["dr"] - pred
    ax3.hist(residuals, bins=50, alpha=0.4, density=True,
             color=events_cfg[test_name]["color"], label=test_name)
    ax3.axvline(np.mean(residuals), color=events_cfg[test_name]["color"],
                linewidth=1.5, linestyle="--")

ax3.set_xlabel("Residual (observed - predicted)")
ax3.set_ylabel("Density")
ax3.set_title("LOO: Residual Distributions")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- 4. 70/30 split: MAE stability ---
ax4 = fig.add_subplot(gs[1, 0])
for i, name in enumerate(raw_data):
    maes = [m["mae"] for m in split_results[name]]
    ax4.boxplot(maes, positions=[i], widths=0.5,
                boxprops=dict(color=events_cfg[name]["color"]),
                medianprops=dict(color=events_cfg[name]["color"]))
ax4.set_xticks(range(len(raw_data)))
ax4.set_xticklabels(list(raw_data.keys()))
ax4.set_ylabel("MAE (test set)")
ax4.set_title(f"70/30 Split Stability ({N_SPLITS} runs)")
ax4.grid(True, alpha=0.3, axis="y")

# --- 5. LOO curves: each trained model vs all data ---
ax5 = fig.add_subplot(gs[1, 1:])
xi_plot = np.linspace(0, 6, 200)

for test_name, res in loo_results.items():
    theta = res["theta"]
    params = DurationRichardsParams.from_vector(theta)
    model = DurationRichards(params)

    # plot curve for each event's duration
    for ev_name, ev_data in raw_data.items():
        tau = ev_data["duration_h"]
        D = model(xi_plot, tau)
        ls = "-" if ev_name == test_name else ":"
        lw = 2.5 if ev_name == test_name else 1
        alpha = 1.0 if ev_name == test_name else 0.3
        color = events_cfg[test_name]["color"]

        if ev_name == test_name:
            ax5.plot(xi_plot, D, color=color, linewidth=lw, linestyle=ls,
                     alpha=alpha,
                     label=f"Trained w/o {test_name}, predicting {test_name}")

    # data for test event
    test = raw_data[test_name]
    bd, bdr, _ = bin_data(test["depth"], test["dr"])
    ax5.scatter(bd, bdr, s=50, color=events_cfg[test_name]["color"],
                edgecolors="white", zorder=4, alpha=0.8)

ax5.set_xlabel("Flood Depth (m)")
ax5.set_ylabel("Damage Ratio")
ax5.set_title("LOO: Each Model Predicting the Held-Out Event")
ax5.legend(fontsize=8)
ax5.set_ylim(-0.05, 1.05)
ax5.grid(True, alpha=0.3)

# --- 6. Summary table ---
ax6 = fig.add_subplot(gs[2, 0:2])
ax6.axis("off")

header = f"{'Test Event':>12} │ {'Method':>20} │ {'MAE':>7} │ {'RMSE':>7} │ {'R²':>7} │ {'Bias':>7} │ {'N':>7}"
sep = "─" * len(header)
lines = [
    "OUT-OF-SAMPLE VALIDATION RESULTS",
    "═" * 50,
    "",
    "Test 1: Leave-One-Event-Out",
    sep,
    header,
    sep,
]

for test_name, res in loo_results.items():
    for method_name, method_key in [("Duration-Richards", "duration_richards"),
                                      ("Linear baseline", "linear"),
                                      ("USACE lookup", "usace")]:
        m = res[method_key]
        lines.append(
            f"{test_name:>12} │ {method_name:>20} │ {m['mae']:>7.4f} │ "
            f"{m['rmse']:>7.4f} │ {m['r2']:>7.4f} │ {m['bias']:>+7.4f} │ {m['n']:>7,}"
        )
    lines.append(sep)

lines.extend([
    "",
    "Test 2: Random 70/30 Split (mean ± std over 5 runs)",
    sep,
])
for name in raw_data:
    maes = [m["mae"] for m in split_results[name]]
    rmses = [m["rmse"] for m in split_results[name]]
    r2s = [m["r2"] for m in split_results[name]]
    lines.append(
        f"{name:>12} │ {'Duration-Richards':>20} │ "
        f"{np.mean(maes):.4f}±{np.std(maes):.3f} │ "
        f"{np.mean(rmses):.4f}±{np.std(rmses):.3f} │ "
        f"{np.mean(r2s):.4f}±{np.std(r2s):.3f}"
    )

ax6.text(0.02, 0.95, "\n".join(lines), transform=ax6.transAxes,
         fontsize=8, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# --- 7. Improvement over baselines ---
ax7 = fig.add_subplot(gs[2, 2])

improvements = []
labels_imp = []
for test_name in loo_results:
    mae_dr = loo_results[test_name]["duration_richards"]["mae"]
    mae_usace = loo_results[test_name]["usace"]["mae"]
    mae_lin = loo_results[test_name]["linear"]["mae"]

    imp_vs_usace = (mae_usace - mae_dr) / mae_usace * 100
    imp_vs_linear = (mae_lin - mae_dr) / mae_lin * 100

    improvements.append(imp_vs_usace)
    labels_imp.append(f"{test_name}\nvs USACE")
    improvements.append(imp_vs_linear)
    labels_imp.append(f"{test_name}\nvs Linear")

colors_imp = []
for v in improvements:
    colors_imp.append("green" if v > 0 else "red")

ax7.barh(range(len(improvements)), improvements, color=colors_imp, alpha=0.7)
ax7.set_yticks(range(len(improvements)))
ax7.set_yticklabels(labels_imp, fontsize=8)
ax7.set_xlabel("MAE Improvement (%)")
ax7.set_title("Duration-Richards vs Baselines")
ax7.axvline(0, color="black", linewidth=0.5)
ax7.grid(True, alpha=0.3, axis="x")

for i, v in enumerate(improvements):
    ax7.text(v + (1 if v > 0 else -1), i, f"{v:+.1f}%",
             va="center", fontsize=8, fontweight="bold")

plt.suptitle("Out-of-Sample Validation: Duration-Richards on 182K FEMA Claims",
             fontsize=14, y=1.01)
plt.savefig("figures/11_validation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved → notebooks/figures/11_validation.png")
