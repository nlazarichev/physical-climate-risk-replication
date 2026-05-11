"""
Empirical κ calibration from 82,650 NFIP non-residential claims.

Replaces company-based κ (N=24) with claims-based κ (N=82K).
Grouping: (numberOfFloors, coverageBand) — observable, physically motivated.

Physical logic:
  - floors: fraction of building above flood water → lower damage ratio
  - coverage: proxy for building size, construction quality, floor elevation

κ = median(DR_commercial / DR_residential) at each depth bin, then
averaged across depths (weighted by N per bin) to get a stable estimate.

Company validations (24 events) become out-of-sample tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

try:
    from .richards_damage import richards_damage, DEFAULT_PARAMS
except ImportError:
    from richards_damage import richards_damage, DEFAULT_PARAMS


DATA = Path(__file__).parent.parent / "data" / "raw"

# ─────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────

@dataclass
class KappaCell:
    """Calibration result for one (floors, coverage) cell."""
    floors_group: str
    coverage_band: str
    kappa: float
    kappa_std: float      # std across depth bins
    n_claims: int
    depth_bins_used: int


def calibrate_kappa_from_claims(
    claims_path: Path = None,
    depth_bins: list = None,
    min_claims_per_bin: int = 20,
) -> dict:
    """
    Calibrate κ per (floors, coverage) group from NFIP commercial claims.

    For each group and depth bin:
      κ_ij = median(DR_commercial) / DR_residential(depth_midpoint, τ=0)

    Then κ per group = weighted average of κ_ij across depth bins.

    Returns:
        dict with 'lookup' (nested dict floors→coverage→KappaCell),
        'table' (DataFrame), 'n_total' (total claims used)
    """
    if claims_path is None:
        claims_path = DATA / "fema_nfip_commercial_claims.parquet"

    if depth_bins is None:
        depth_bins = [0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 6.0]

    df = pd.read_parquet(claims_path)

    # Compute damage ratio
    df["payout"] = df["amountPaidOnBuildingClaim"].fillna(0) + df["amountPaidOnContentsClaim"].fillna(0)
    df["coverage"] = df["totalBuildingInsuranceCoverage"].fillna(0) + df["totalContentsInsuranceCoverage"].fillna(0)
    df["DR"] = np.clip(df["payout"] / df["coverage"].replace(0, np.nan), 0, 1.5)
    df["depth_m"] = df["waterDepth"] * 0.0254  # inches → meters

    # Quality filter
    clean = df[
        (df["coverage"] > 1000) &
        (df["DR"] > 0) & (df["DR"] <= 1.5) &
        (df["depth_m"] > 0) & (df["depth_m"] <= 6)
    ].copy()

    # Grouping variables
    floors = clean["numberOfFloorsInTheInsuredBuilding"].fillna(1).clip(1, 4)
    clean["floors_group"] = np.where(floors <= 1, "1-story",
                            np.where(floors <= 2, "2-story", "3+"))

    clean["coverage_band"] = pd.cut(
        clean["coverage"],
        bins=[0, 50_000, 200_000, 500_000, np.inf],
        labels=["<50K", "50-200K", "200-500K", ">500K"],
    )

    clean["depth_bin"] = pd.cut(clean["depth_m"], bins=depth_bins)

    # Calibrate κ per (floors, coverage) group
    results = []

    for floors_g in ["1-story", "2-story", "3+"]:
        for cov_b in ["<50K", "50-200K", "200-500K", ">500K"]:
            mask = (clean["floors_group"] == floors_g) & (clean["coverage_band"] == cov_b)
            subset = clean[mask]

            if len(subset) < min_claims_per_bin:
                continue

            kappas_by_depth = []
            weights = []

            for _, grp in subset.groupby("depth_bin", observed=True):
                if len(grp) < min_claims_per_bin:
                    continue

                depth_mid = (grp["depth_bin"].iloc[0].left + grp["depth_bin"].iloc[0].right) / 2
                dr_residential = richards_damage(depth_mid, tau=0)

                if dr_residential < 0.05:
                    continue  # skip near-zero residential DR (unstable ratio)

                dr_commercial = grp["DR"].median()
                kappa_bin = dr_commercial / dr_residential
                kappas_by_depth.append(kappa_bin)
                weights.append(len(grp))

            if not kappas_by_depth:
                continue

            # Weighted average across depth bins
            kappas_arr = np.array(kappas_by_depth)
            weights_arr = np.array(weights, dtype=float)
            kappa_avg = np.average(kappas_arr, weights=weights_arr)
            kappa_std = np.sqrt(np.average((kappas_arr - kappa_avg) ** 2, weights=weights_arr))

            results.append(KappaCell(
                floors_group=floors_g,
                coverage_band=cov_b,
                kappa=float(np.clip(kappa_avg, 0.05, 3.0)),
                kappa_std=float(kappa_std),
                n_claims=int(mask.sum()),
                depth_bins_used=len(kappas_by_depth),
            ))

    # Build lookup table
    lookup = {}
    for cell in results:
        if cell.floors_group not in lookup:
            lookup[cell.floors_group] = {}
        lookup[cell.floors_group][cell.coverage_band] = cell

    # DataFrame for display
    table = pd.DataFrame([
        {
            "floors": c.floors_group,
            "coverage": c.coverage_band,
            "kappa": c.kappa,
            "kappa_std": c.kappa_std,
            "n_claims": c.n_claims,
            "depth_bins": c.depth_bins_used,
        }
        for c in results
    ])

    return {
        "lookup": lookup,
        "table": table,
        "n_total": len(clean),
        "cells": results,
    }


def get_kappa_from_claims(
    floors: int,
    coverage_usd: float,
    lookup: dict,
) -> float:
    """
    Look up claims-calibrated κ for a given building.

    Args:
        floors: number of stories (1, 2, 3+)
        coverage_usd: building insurance coverage or replacement value ($)
        lookup: from calibrate_kappa_from_claims()['lookup']

    Returns:
        κ value
    """
    if floors <= 1:
        fg = "1-story"
    elif floors <= 2:
        fg = "2-story"
    else:
        fg = "3+"

    if coverage_usd < 50_000:
        cb = "<50K"
    elif coverage_usd < 200_000:
        cb = "50-200K"
    elif coverage_usd < 500_000:
        cb = "200-500K"
    else:
        cb = ">500K"

    cell = lookup.get(fg, {}).get(cb)
    if cell is not None:
        return cell.kappa

    # Fallback: try same floors, any coverage
    if fg in lookup:
        cells = list(lookup[fg].values())
        return float(np.median([c.kappa for c in cells]))

    return 0.50  # conservative default


# ─────────────────────────────────────────────────
# OOS Validation on 24 companies
# ─────────────────────────────────────────────────

# Each company mapped to (floors, coverage_usd) for lookup
# floors = typical number of stories at the affected facility
# coverage_usd = estimated replacement value per building (for band assignment)
COMPANY_BUILDING_PROFILES = [
    # (name, floors, coverage_usd, actual_loss_M, h, tau, sector_type, financials_dict, severity, hazard)
    # Harvey 2017
    ("NRG Energy", 1, 600_000, 75, 1.8, 288, "power_plant",
     dict(V=600e6, rev=15e6, OL=0.55, emp=2000, ins=0.55, sc=0.15, cm=0.30), 0.6, "clean_flood"),
    ("HCA Healthcare", 3, 600_000, 50, 0.8, 288, "hospital",
     dict(V=150e6, rev=12e6, OL=0.70, emp=8000, ins=0.60, sc=0.10, cm=0.25), 0.7, "clean_flood"),
    ("LyondellBasell", 1, 600_000, 200, 1.5, 288, "petrochemical",
     dict(V=2000e6, rev=30e6, OL=0.50, emp=3000, ins=0.50, sc=0.20, cm=0.05), 0.7, "clean_flood"),
    ("ExxonMobil", 1, 600_000, 170, 1.0, 288, "petrochemical",
     dict(V=3000e6, rev=50e6, OL=0.45, emp=5000, ins=0.60, sc=0.10, cm=0.03), 0.6, "clean_flood"),
    ("CenterPoint", 1, 600_000, 125, 1.0, 288, "power_plant",
     dict(V=1000e6, rev=10e6, OL=0.60, emp=3000, ins=0.40, sc=0.05, cm=0.25), 0.7, "clean_flood"),
    ("Olin Corp", 1, 600_000, 43, 1.5, 288, "petrochemical",
     dict(V=300e6, rev=5e6, OL=0.55, emp=800, ins=0.50, sc=0.15, cm=0.05), 0.6, "clean_flood"),
    ("Huntsman", 1, 600_000, 50, 1.8, 288, "petrochemical",
     dict(V=500e6, rev=8e6, OL=0.50, emp=1500, ins=0.50, sc=0.15, cm=0.05), 0.6, "clean_flood"),
    ("Phillips 66 Harvey", 1, 600_000, 60, 1.5, 288, "petrochemical",
     dict(V=500e6, rev=8e6, OL=0.45, emp=800, ins=0.55, sc=0.10, cm=0.03), 0.5, "clean_flood"),
    # KZN 2022
    ("Toyota SA", 1, 600_000, 361, 2.5, 96, "manufacturing",
     dict(V=600e6, rev=6.8e6, OL=0.75, emp=8000, ins=0.50, sc=0.20, cm=0.15, comp=150), 0.8, "mudslide"),
    # Helene 2024
    ("Baxter", 2, 600_000, 200, 2.0, 72, "manufacturing",
     dict(V=400e6, rev=10e6, OL=0.65, emp=2500, ins=0.50, sc=0.10, cm=0.05, comp=250), 0.7, "clean_flood"),
    # Michigan 2022
    ("Abbott", 1, 600_000, 15, 0.5, 6, "manufacturing",
     dict(V=200e6, rev=3e6, OL=0.55, emp=800, ins=0.70, sc=0.05, cm=0.02), 0.3, "clean_flood"),
    # Thailand 2011
    ("Western Digital", 1, 600_000, 275, 1.8, 720, "manufacturing",
     dict(V=500e6, rev=12e6, OL=0.60, emp=28000, ins=0.40, sc=0.50, cm=0.20, comp=30), 0.9, "clean_flood"),
    ("Nikon", 2, 600_000, 140, 2.0, 1200, "manufacturing",
     dict(V=300e6, rev=5e6, OL=0.55, emp=3000, ins=0.30, sc=0.30, cm=0.10, comp=30), 0.9, "clean_flood"),
    ("Honda", 1, 600_000, 500, 2.3, 1200, "manufacturing",
     dict(V=800e6, rev=4e6, OL=0.75, emp=4000, ins=0.35, sc=0.60, cm=0.15, comp=30), 0.9, "clean_flood"),
    ("Canon", 1, 600_000, 260, 2.5, 1000, "manufacturing",
     dict(V=400e6, rev=8e6, OL=0.55, emp=5000, ins=0.35, sc=0.30, cm=0.10, comp=30), 0.9, "clean_flood"),
    ("Minebea", 2, 600_000, 120, 2.0, 1000, "manufacturing",
     dict(V=400e6, rev=4e6, OL=0.60, emp=20000, ins=0.30, sc=0.30, cm=0.05, comp=20), 0.9, "clean_flood"),
    ("Pioneer", 1, 600_000, 100, 1.5, 1200, "manufacturing",
     dict(V=200e6, rev=3e6, OL=0.50, emp=3000, ins=0.35, sc=0.20, cm=0.05, comp=30), 0.9, "clean_flood"),
    ("Rohm", 1, 600_000, 67, 1.5, 1200, "manufacturing",
     dict(V=150e6, rev=2e6, OL=0.55, emp=2000, ins=0.35, sc=0.15, cm=0.05, comp=30), 0.9, "clean_flood"),
    ("Panasonic", 1, 600_000, 170, 1.5, 1000, "manufacturing",
     dict(V=400e6, rev=6e6, OL=0.55, emp=6000, ins=0.35, sc=0.25, cm=0.10, comp=30), 0.9, "clean_flood"),
    ("Toshiba", 1, 600_000, 260, 2.5, 1000, "manufacturing",
     dict(V=500e6, rev=8e6, OL=0.55, emp=8000, ins=0.35, sc=0.30, cm=0.10, comp=30), 0.9, "clean_flood"),
    # Sandy 2012
    ("Con Edison", 1, 600_000, 460, 4.2, 108, "power_plant",
     dict(V=2000e6, rev=30e6, OL=0.65, emp=5000, ins=0.40, sc=0.05, cm=0.30), 0.7, "storm_surge"),
    # Florence 2018
    ("Int'l Paper", 1, 600_000, 35, 1.0, 96, "manufacturing",
     dict(V=400e6, rev=6e6, OL=0.50, emp=2000, ins=0.55, sc=0.10, cm=0.05), 0.5, "clean_flood"),
    # Phillips 66 Ida
    ("Phillips 66 Ida", 1, 600_000, 400, 1.5, 60, "petrochemical",
     dict(V=2500e6, rev=40e6, OL=0.50, emp=900, ins=0.45, sc=0.05, cm=0.03), 0.7, "clean_flood"),
    # Desmond 2015
    ("United Biscuits", 2, 600_000, 30, 1.2, 2880, "manufacturing",
     dict(V=150e6, rev=3e6, OL=0.55, emp=1000, ins=0.50, sc=0.10, cm=0.10, comp=200), 0.6, "clean_flood"),
]


def calibrate_corporate_resilience(calibration: dict = None) -> dict:
    """
    Two-stage calibration:
      Stage 1: κ_claims from 82K NFIP claims (building-level vulnerability)
      Stage 2: ρ_corporate from 24 companies (corporate resilience factor)

    κ_total = κ_claims(floors, coverage) × ρ_corporate

    NFIP claims capture small/medium commercial buildings. Large corporate
    facilities have additional resilience (insurance, BCP, redundancy, rerouting)
    not present in NFIP data. ρ_corporate captures this gap.

    ρ is calibrated as: median(actual_loss / ECL_with_kappa_claims) across
    24 company validations. Then validated via leave-one-out.
    """
    try:
        from .enhanced_damage import (
            compute_enhanced_ecl, SectorFinancials, ExposureProfile,
        )
    except ImportError:
        from enhanced_damage import (
            compute_enhanced_ecl, SectorFinancials, ExposureProfile,
        )

    if calibration is None:
        calibration = calibrate_kappa_from_claims()

    lookup = calibration["lookup"]

    # Compute ECL with raw claims-based κ for each company
    raw_ratios = []
    for entry in COMPANY_BUILDING_PROFILES:
        name, floors, cov_usd, actual_M, h, tau, sector, p, sev, hazard = entry
        kappa_claims = get_kappa_from_claims(floors, cov_usd, lookup)

        fin = SectorFinancials(
            daily_revenue=p["rev"], operating_leverage=p["OL"],
            employees_exposed=p["emp"], avg_daily_comp=p.get("comp", 300),
            insurance_coverage=p["ins"], kappa=kappa_claims)
        exp = ExposureProfile(
            V_exposed=p["V"], financials=fin, sector_type=sector,
            supply_concentration=p["sc"], customer_mobility=p["cm"],
            hazard_type=hazard)

        ecl = compute_enhanced_ecl(h, tau, exp, sev)
        ecl_M = ecl["ecl_total"] / 1e6
        raw_ratios.append(actual_M / ecl_M if ecl_M > 0 else 1.0)

    # ρ_corporate = median of (actual / ECL_raw)
    rho = float(np.median(raw_ratios))

    # Leave-one-out cross-validation for stability
    loo_rhos = []
    for i in range(len(raw_ratios)):
        loo = [r for j, r in enumerate(raw_ratios) if j != i]
        loo_rhos.append(float(np.median(loo)))

    return {
        "rho_corporate": rho,
        "rho_std": float(np.std(loo_rhos)),
        "rho_range": (float(min(loo_rhos)), float(max(loo_rhos))),
        "n_companies": len(raw_ratios),
        "raw_ratios": raw_ratios,
    }


def get_kappa_total(floors: int, coverage_usd: float,
                     lookup: dict, rho_corporate: float) -> float:
    """
    Total κ = κ_claims(floors, coverage) × ρ_corporate.

    Stage 1 (claims, N=82K): building-level vulnerability from NFIP.
    Stage 2 (corporate, N=24): resilience factor for large corporates.
    """
    return get_kappa_from_claims(floors, coverage_usd, lookup) * rho_corporate


def run_oos_validation(calibration: dict = None) -> pd.DataFrame:
    """
    Out-of-sample validation: use two-stage κ to predict
    losses for 24 companies, compare to actual disclosed losses.

    Returns DataFrame with predicted ECL, actual, ratio for each company.
    """
    try:
        from .enhanced_damage import (
            compute_enhanced_ecl, SectorFinancials, ExposureProfile,
            SECTOR_DOWNTIME,
        )
    except ImportError:
        from enhanced_damage import (
            compute_enhanced_ecl, SectorFinancials, ExposureProfile,
            SECTOR_DOWNTIME,
        )

    if calibration is None:
        calibration = calibrate_kappa_from_claims()

    lookup = calibration["lookup"]

    # Stage 2: calibrate ρ_corporate
    corp = calibrate_corporate_resilience(calibration)
    rho = corp["rho_corporate"]

    results = []

    for i, entry in enumerate(COMPANY_BUILDING_PROFILES):
        name, floors, cov_usd, actual_M, h, tau, sector, p, sev, hazard = entry

        kappa_claims = get_kappa_from_claims(floors, cov_usd, lookup)

        # Leave-one-out: recalibrate ρ without this company
        loo_ratios = [r for j, r in enumerate(corp["raw_ratios"]) if j != i]
        rho_loo = float(np.median(loo_ratios))

        kappa_total = kappa_claims * rho_loo

        fin = SectorFinancials(
            daily_revenue=p["rev"], operating_leverage=p["OL"],
            employees_exposed=p["emp"], avg_daily_comp=p.get("comp", 300),
            insurance_coverage=p["ins"], kappa=kappa_total)
        exp = ExposureProfile(
            V_exposed=p["V"], financials=fin, sector_type=sector,
            supply_concentration=p["sc"], customer_mobility=p["cm"],
            hazard_type=hazard)

        ecl = compute_enhanced_ecl(h, tau, exp, sev)
        ecl_M = ecl["ecl_total"] / 1e6
        ratio = ecl_M / actual_M if actual_M > 0 else 0

        results.append({
            "company": name,
            "floors": floors,
            "kappa_claims": kappa_claims,
            "rho_loo": rho_loo,
            "kappa_total": kappa_total,
            "ecl_M": ecl_M,
            "actual_M": actual_M,
            "ratio": ratio,
            "downtime_d": ecl["downtime_days"],
        })

    return pd.DataFrame(results), corp


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 90)
    print("κ CALIBRATION FROM 82K NFIP COMMERCIAL CLAIMS")
    print("=" * 90)

    cal = calibrate_kappa_from_claims()

    print(f"\nTotal clean claims: {cal['n_total']:,}")
    print(f"\nCalibrated κ lookup table:")
    print(f"\n{'Floors':>10s} | {'Coverage':>10s} | {'κ':>6s} | {'κ_std':>6s} | {'N claims':>8s} | {'Depth bins':>10s}")
    print("-" * 65)
    for _, row in cal["table"].iterrows():
        print(f"  {row['floors']:>8s} | {row['coverage']:>10s} | {row['kappa']:5.3f} | "
              f"{row['kappa_std']:5.3f} | {row['n_claims']:>8,} | {row['depth_bins']:>10}")

    # Stage 2: Corporate resilience
    print(f"\n{'=' * 90}")
    print("STAGE 2: CORPORATE RESILIENCE FACTOR (ρ)")
    print("=" * 90)

    corp = calibrate_corporate_resilience(cal)
    print(f"\n  ρ_corporate = {corp['rho_corporate']:.3f}")
    print(f"  LOO std:      {corp['rho_std']:.4f}")
    print(f"  LOO range:    [{corp['rho_range'][0]:.3f}, {corp['rho_range'][1]:.3f}]")
    print(f"  Interpretation: large corporates absorb {(1-corp['rho_corporate'])*100:.0f}% of "
          f"NFIP-equivalent damage through insurance, BCP, redundancy")

    # OOS Validation (leave-one-out on ρ)
    print(f"\n{'=' * 90}")
    print("OUT-OF-SAMPLE VALIDATION (leave-one-out on ρ)")
    print("κ_total = κ_claims(floors) × ρ_corporate(LOO)")
    print("=" * 90)

    oos, _ = run_oos_validation(cal)

    print(f"\n{'Company':>22s} | {'Fl':>2s} | {'κ_cl':>5s} | {'ρ_LOO':>5s} | {'κ_tot':>5s} | {'ECL':>8s} | {'Actual':>7s} | {'Ratio':>5s}")
    print("-" * 85)
    for _, r in oos.iterrows():
        print(f"  {r['company']:>20s} | {r['floors']:>2.0f} | {r['kappa_claims']:.2f} | "
              f"{r['rho_loo']:.3f} | {r['kappa_total']:.3f} | "
              f"${r['ecl_M']:>6.1f}M | ${r['actual_M']:>5.0f}M | {r['ratio']:.2f}x")

    ratios = oos["ratio"].values
    within_25 = np.sum((ratios >= 0.4) & (ratios <= 2.5))
    within_20 = np.sum((ratios >= 0.5) & (ratios <= 2.0))
    median_ratio = float(np.median(ratios))
    mae_log = float(np.mean(np.abs(np.log(ratios))))

    print(f"\n  N = {len(ratios)} companies (ALL out-of-sample via LOO)")
    print(f"  Within 0.4-2.5×: {within_25}/{len(ratios)} ({within_25/len(ratios):.0%})")
    print(f"  Within 0.5-2.0×: {within_20}/{len(ratios)} ({within_20/len(ratios):.0%})")
    print(f"  Median ratio: {median_ratio:.2f}×")
    print(f"  Mean |log(ratio)|: {mae_log:.2f}")
