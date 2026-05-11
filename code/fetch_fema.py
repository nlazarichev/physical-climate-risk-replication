"""
Fetch FEMA NFIP flood insurance claims via OpenFEMA API.

Downloads claims for specified flood events, computes damage ratios,
and saves clean calibration data.

Usage:
    cd physical-climate-risk
    PYTHONPATH=. python data/fetch_fema.py

Output:
    data/fema_harvey_claims.csv      — raw claims
    data/fema_harvey_calibration.csv — clean (depth_m, damage_ratio) pairs
"""

import json
import time
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd


BASE_URL = "https://www.fema.gov/api/open/v2/FimaNfipClaims"

FIELDS = [
    "dateOfLoss",
    "yearOfLoss",
    "reportedZipCode",
    "waterDepth",               # inches
    "floodWaterDuration",       # hours (NEW — we need this for duration model)
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage",
    "buildingDescriptionCode",
    "occupancyType",
    "latitude",
    "longitude",
    "countyCode",
    "censusTract",
    "state",
    "floodEvent",
    "basementEnclosureCrawlspaceType",
    "numberOfFloorsInTheInsuredBuilding",
    "primaryResidenceIndicator",
    "buildingDamageAmount",
    "contentsDamageAmount",
    "causeOfDamage",
]


def fetch_event(event_name: str, max_records: int = 250_000) -> pd.DataFrame:
    """Fetch all claims for a given flood event via pagination."""
    select = ",".join(FIELDS)
    page_size = 10_000
    all_records = []
    skip = 0

    # first, get total count
    filter_str = f"floodEvent eq '{event_name}'"
    count_params = urlencode({
        "$inlinecount": "allpages",
        "$top": "1",
        "$filter": filter_str,
    })
    count_url = f"{BASE_URL}?{count_params}"
    print(f"Querying count for '{event_name}'...")
    resp = json.loads(urlopen(Request(count_url), timeout=60).read())
    total = resp.get("metadata", {}).get("count", 0)
    print(f"  Total records: {total:,}")

    fetch_total = min(total, max_records)

    while skip < fetch_total:
        params = urlencode({
            "$top": str(page_size),
            "$skip": str(skip),
            "$filter": filter_str,
            "$select": select,
            "$orderby": "waterDepth desc",
        })
        url = f"{BASE_URL}?{params}"
        print(f"  Fetching {skip:,}–{min(skip + page_size, fetch_total):,} "
              f"of {fetch_total:,}...")

        for attempt in range(5):
            try:
                resp = json.loads(urlopen(Request(url), timeout=120).read())
                records = resp.get("FimaNfipClaims", [])
                break
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"  Error at skip={skip} (attempt {attempt+1}/5): {e}, "
                      f"waiting {wait}s...")
                time.sleep(wait)
                records = None
        if records is None:
            print(f"  FAILED at skip={skip}, stopping pagination for this event")
            break

        if not records:
            break

        all_records.extend(records)
        skip += page_size
        time.sleep(0.5)  # be polite to API

    df = pd.DataFrame(all_records)
    print(f"  Fetched {len(df):,} records total")
    return df


def compute_damage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute damage ratio from FEMA claims data.

    damage_ratio = total_paid / total_coverage

    Filter criteria (matches paper Table 1 caption exactly):
      - waterDepth > 0 (must have recorded flood depth)
      - total_paid > 0 (must have actual claim)
      - damage_ratio ∈ (0, 1.5] (reasonable range; >1 possible due to
        replacement cost vs actual cash value)
      - total_coverage > 0 (denominator non-zero — required for division)
    """
    out = df.copy()

    # convert types
    numeric_cols = [
        "waterDepth", "floodWaterDuration",
        "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
        "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage",
        "buildingDamageAmount", "contentsDamageAmount",
        "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # depth in meters (from inches)
    out["depth_m"] = out["waterDepth"] * 0.0254

    # duration in hours
    out["duration_h"] = out["floodWaterDuration"]

    # total paid and coverage
    out["total_paid"] = (
        out["amountPaidOnBuildingClaim"].fillna(0)
        + out["amountPaidOnContentsClaim"].fillna(0)
    )
    out["total_coverage"] = (
        out["totalBuildingInsuranceCoverage"].fillna(0)
        + out["totalContentsInsuranceCoverage"].fillna(0)
    )

    # damage ratio
    out["damage_ratio"] = out["total_paid"] / out["total_coverage"]

    # building damage ratio (building only — more relevant for structural damage)
    out["building_dr"] = (
        out["amountPaidOnBuildingClaim"].fillna(0)
        / out["totalBuildingInsuranceCoverage"].replace(0, np.nan)
    )

    # filter
    mask = (
        (out["waterDepth"] > 0)
        & (out["total_coverage"] > 0)
        & (out["total_paid"] > 0)
        & (out["damage_ratio"] > 0)
        & (out["damage_ratio"] <= 1.5)
    )
    clean = out[mask].copy()

    # cap DR at 1.0 for calibration
    clean["damage_ratio_capped"] = clean["damage_ratio"].clip(upper=1.0)
    clean["building_dr_capped"] = clean["building_dr"].clip(upper=1.0)

    print(f"\nCleaning: {len(df):,} → {len(clean):,} records "
          f"({len(clean)/len(df)*100:.1f}%)")
    print(f"  Depth range: {clean['depth_m'].min():.2f}m – {clean['depth_m'].max():.2f}m")
    print(f"  DR range: {clean['damage_ratio_capped'].min():.3f} – "
          f"{clean['damage_ratio_capped'].max():.3f}")
    print(f"  DR median: {clean['damage_ratio_capped'].median():.3f}")

    if "duration_h" in clean.columns and clean["duration_h"].notna().sum() > 0:
        dur = clean["duration_h"].dropna()
        print(f"  Duration range: {dur.min():.0f}h – {dur.max():.0f}h "
              f"(n={len(dur):,})")

    return clean


def make_calibration_data(clean: pd.DataFrame) -> pd.DataFrame:
    """Extract (depth_m, damage_ratio) pairs for Richards calibration.

    Also includes duration and building type for stratified calibration.
    """
    cols = ["depth_m", "damage_ratio_capped", "building_dr_capped"]
    optional = ["duration_h", "occupancyType", "buildingDescriptionCode",
                "latitude", "longitude", "reportedZipCode"]

    for c in optional:
        if c in clean.columns:
            cols.append(c)

    cal = clean[cols].dropna(subset=["depth_m", "damage_ratio_capped"])
    cal = cal.rename(columns={
        "damage_ratio_capped": "damage_ratio",
        "building_dr_capped": "building_damage_ratio",
    })

    return cal


def summary_by_depth(cal: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Bin by depth and compute mean DR per bin — for quick visualization."""
    cal = cal.copy()
    cal["depth_bin"] = pd.cut(cal["depth_m"], bins=bins)
    summary = cal.groupby("depth_bin", observed=True).agg(
        depth_mean=("depth_m", "mean"),
        damage_ratio_mean=("damage_ratio", "mean"),
        damage_ratio_median=("damage_ratio", "median"),
        damage_ratio_std=("damage_ratio", "std"),
        count=("damage_ratio", "count"),
    ).reset_index()
    return summary


# ============================================================================

if __name__ == "__main__":
    # package-relative paths
    pkg_root = Path(__file__).resolve().parent.parent
    raw_dir = pkg_root / "data" / "raw"
    proc_dir = pkg_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    events = {
        "Hurricane Harvey": "fema_harvey",
        "Hurricane Sandy": "fema_sandy",
        "Hurricane Katrina": "fema_katrina",
    }
    MAX_PER_EVENT = 250_000  # full pull — Sandy ~145k, Harvey ~92k, Katrina ~208k

    for event_name, prefix in events.items():
        raw_path = raw_dir / f"{prefix}_claims.csv"
        cal_path = proc_dir / f"{prefix}_calibration.csv"
        summary_path = proc_dir / f"{prefix}_summary.csv"

        if raw_path.exists():
            print(f"\n{raw_path} already exists, loading...")
            df = pd.read_csv(raw_path)
        else:
            print(f"\nFetching {event_name}...")
            df = fetch_event(event_name, max_records=MAX_PER_EVENT)
            df.to_csv(raw_path, index=False)
            print(f"  Saved to {raw_path}")

        # clean and compute DRs
        clean = compute_damage_ratios(df)

        # calibration data
        cal = make_calibration_data(clean)
        cal.to_csv(cal_path, index=False)
        print(f"  Calibration data ({len(cal):,} rows) → {cal_path}")

        # summary by depth bins
        summary = summary_by_depth(cal)
        summary.to_csv(summary_path, index=False)
        print(f"  Summary ({len(summary)} bins) → {summary_path}")

    print("\nDone. Ready for Bayesian calibration.")
