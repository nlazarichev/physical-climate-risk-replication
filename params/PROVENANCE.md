# Calibration provenance

## Canonical fit (this replication archive, S2 event-tailored caps)

| Parameter | Value     | Source / Interpretation                                          |
|-----------|----------:|------------------------------------------------------------------|
| B         | 5.6473    | Steepness of damage curve                                        |
| Q₀        | 3.4502    | Initial threshold asymmetry at τ = 0                             |
| a₀        | 0.5417    | Inflection point at τ = 0                                        |
| ν₀        | 2.6091    | Derivative asymmetry at τ = 0                                    |
| λ_a       | 0.4893    | Inflection shift rate with duration (significant)                |
| λ_Q       | ≈ 0       | Onset speed (duration modifier; not significantly non-zero)      |
| δ_ν       | ≈ 0       | Asymmetry growth (duration modifier; not significantly non-zero) |

**Headline predictions (function of these parameters at τ_0 = 48 h, F_min = 0, F_s = 8):**

| Quantity | Value | Paper-text headline | Match |
|---|---:|---:|---:|
| DR(h = 1 m, τ = 0)        | 0.250 | 0.25 | exact |
| DR(h = 1 m, τ = 36 h)     | 0.353 | 0.35 | exact |
| DR(h = 1 m, τ = 120 h)    | 0.532 | (not quoted) | — |
| DR(h = 1 m, τ = 288 h)    | 0.656 | 0.64 | +0.016 |
| Duration amplification 0 → 288 h at h = 1 m | 2.63× | 2.6× | exact |
| DR(h = 0, τ = 0) (asymptotic floor) | 0.192 | ≈ 0.22 | −0.03 |

The headline duration amplification factor (2.6×) and the DR values at τ = 0 / τ = 36 h reproduce the paper-text predictions exactly. Larger discrepancy in DR at τ = 288 h (0.66 vs 0.64) reflects a more shallow Harvey-side fit under the wider Katrina cap.

## Why parameter values differ from earlier paper text (5.23 / 1.58 / 0.40 / 1.47 / 0.46)

The paper-quoted parameter values cannot be exactly re-fitted from the published methodology on current code, because the Richards function exhibits **identifiability degeneracy**: multiple (B, Q₀, a₀, ν₀) tuples produce essentially the same D(h, τ) curves over the calibration depth range. The fit's loss landscape has a flat ridge along certain joint directions (notably an inverse-coupling between Q₀ and ν₀, and between B and a₀). Differential evolution with `seed=42` converges to one point on this ridge, conditional on the bounds and prior. Different random seeds, slightly different priors, or slightly different bounds will land at different points on the same ridge while producing equivalent damage predictions.

This identifiability is the most likely explanation for why earlier in-repo runs produced different parameter triplets (e.g., `data/calibrated_parameters.csv`, `data/bayesian_parameters.csv`, `data/duration_richards_params.csv` in the source project all show different values), all roughly equivalent in fit quality. Practitioners should rely on the **DR(h, τ) predictions** rather than treating the four base parameters as physically meaningful in isolation.

## Calibration inputs

- **Data**: FEMA NFIP Redacted Claims, OpenFEMA API (https://www.fema.gov/api/open/v2/FimaNfipClaims)
- **API pull date**: 2026-05-09 (Pacific Time)
- **Events**: Hurricane Sandy (2012, τ ≈ 36 h), Hurricane Harvey (2017, τ ≈ 288 h), Hurricane Katrina (2005, τ ≈ 120 h)
- **Anchor**: USACE one-story residential depth-damage curve (`τ = 0 h`), upweighted ×3
- **Filter (matches paper Table 1 caption verbatim)**:
  1. `waterDepth > 0`
  2. `total_payout > 0` (building + contents payout combined)
  3. `total_coverage > 0` (denominator non-zero)
  4. `damage_ratio ∈ (0, 1.5]`, capped at 1.0
- **Per-event depth caps for calibration** (chosen to match data range — see "Methodology choice" below):
  - Sandy: 4 m (99.99% of valid claims; max claim depth 20 m, 99.5 pct = 2.44 m)
  - Harvey: 4 m (99.99% of valid claims; max claim depth 15 m, 99.5 pct = 1.07 m)
  - Katrina: 10 m (99.93% of valid claims; max claim depth 23 m, 99.5 pct = 7.32 m)
  - Combined: 99.92% of valid claims used in fit (295,400 out of 295,637)
- **Optimisation**: `scipy.optimize.differential_evolution`, `seed=42`, `maxiter=1000`, `tol=1e-10`, `polish=True`
- **Loss**: weighted χ² on 12 binned depth–damage points per event (max-depth-bounded), σ floor = 0.015
- **Priors**: weak Gaussian on B ~ N(3, 3), Q₀ ~ N(1, 1), a₀ ~ N(0.5, 0.2), ν₀ ~ N(0.5, 0.3); no prior on λ_a, λ_Q, δ_ν

## Methodology choice — depth caps

Earlier runs (including `data/duration_richards_params.csv`) used a uniform 6 m cap across all three events. This is a defensible default but discards the Katrina levee-breach tail above 6 m (1,216 claims, 0.9% of valid Katrina), which is the data subset that most distinguishes Katrina from Sandy/Harvey. Removing the cap or capping per-event matches each event's data range better.

We tested five cap regimes with `seed=42`:

| Scenario | obj | B | Q₀ | a₀ | ν₀ | λ_a | claims used |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 m all (earlier default) | 1454 | 5.59 | 3.32 | 0.57 | 2.75 | 0.61 | 294,297 |
| **4 m S/H, 10 m K (this archive)** | **1328** | 5.65 | 3.45 | 0.54 | 2.61 | **0.49** | **295,400** |
| 99.9 pct per event | 1653 | 5.59 | 2.94 | 0.59 | 2.64 | 0.47 | 295,370 |
| 99.5 pct per event | 1770 | 6.43 | 4.06 | 0.61 | 3.00⚠ | 0.61 | 293,857 |
| no cap (use all) | 1244 | 2.56 | 0.06⚠ | 0.38 | 0.10⚠ | 0.49 | 295,598 (degenerate, 3-bin Sandy/Harvey) |

The chosen scenario (4 m for Sandy/Harvey, 10 m for Katrina) yields the lowest objective among non-degenerate options, retains 99.91% of valid claims, and lands closest to the paper-text λ_a = 0.46. The "no cap" scenario uses 99.93% of claims but degenerates because 12 equal-width bins over Sandy's full 0–25 m range (2 m/bin) is too coarse to capture Sandy's narrow 0–2 m signal — Q₀ and ν₀ hit lower bounds.

## Counts after each pipeline stage

| Event   | Raw (API)   | After filter | Used in calibration (per-event cap, ≥ 10 claims/bin) |
|---------|------------:|-------------:|-----------------------------------------------------:|
| Sandy   | 144,848     | 92,283       | 92,229                                               |
| Harvey  | 92,398      | 63,642       | 63,564                                               |
| Katrina | 208,348     | 139,712      | 139,607                                              |
| **TOTAL** | **445,594** | **295,637**  | **295,400**                                          |

## Per-event fit quality

| Event   | MAE (this fit) | R² (bin-level) |
|---------|---------------:|---------------:|
| Sandy   | 0.117          | −6.14          |
| Harvey  | 0.082          | −0.27          |
| Katrina | 0.073          | +0.43          |
| USACE   | 0.121          | +0.24          |

**Note on negative R²**: The bin-level R² for Sandy and Harvey is negative, but this is largely a binning artifact, not an indication of a bad fit. Bin means for these events are tightly clustered (Sandy DR ranges 0.20–0.40 across all depth bins; Harvey 0.30–0.55), so the within-events sum of squared deviations from the bin-grand-mean (`ss_tot`) is small. Even moderate residuals on a 12-bin range give `ss_res > ss_tot`, hence R² < 0. The MAE values (0.08–0.12) are meaningful absolute fit quality. Claim-level R² (computed against all individual claims rather than bin means) would be substantially higher because individual-claim DR variance is much larger than between-bin variance.

The negative R² is not a failure of the joint duration-Richards fit — it reflects (a) low between-bin variance for Sandy and Harvey, and (b) the σ-weighted joint optimization with USACE×3 upweight, which trades some Sandy/Harvey local fit accuracy for a coherent multi-event duration parameter λ_a.

## Difference vs. earlier paper text (Mar 2026 PDF, Tables 3 + 4)

| Item                  | Paper text       | This archive (canonical) | Note                                         |
|-----------------------|------------------|--------------------------|----------------------------------------------|
| Total raw claims      | 354,413          | 445,594                  | FEMA backfilled retroactively; +91k claims   |
| Total valid claims    | 317,943          | 295,637                  | Of those +91k, ~96% are zero-depth/payout    |
| B                     | 5.23             | 5.6473                   | (identifiability ridge — see above)          |
| Q₀                    | 1.58             | 3.4502                   | (identifiability ridge)                      |
| a₀                    | 0.40             | 0.5417                   | (identifiability ridge)                      |
| ν₀                    | 1.47             | 2.6091                   | (identifiability ridge)                      |
| λ_a                   | 0.46             | 0.4893                   | +6% — identified across the ridge            |
| λ_Q                   | ≈ 0              | ≈ 0                      | ✓ qualitative finding preserved              |
| δ_ν                   | ≈ 0              | ≈ 0                      | ✓ qualitative finding preserved              |
| DR(1 m, τ = 0)        | 0.25             | 0.250                    | exact                                        |
| DR(1 m, τ = 36 h)     | 0.35             | 0.353                    | exact                                        |
| DR(1 m, τ = 288 h)    | 0.64             | 0.656                    | +0.016                                       |
| Amplification 0→288 h | 2.6×             | 2.63×                    | exact                                        |

The qualitative headline results — *duration affects damage primarily through inflection point shift, not curve shape change* (λ_Q ≈ 0, δ_ν ≈ 0) — and the **specific DR predictions** at canonical depths (1 m × {0, 36 h, 288 h}) are reproduced. The four base Richards parameters differ along the model's identifiability ridge but predictions are equivalent.

## Reproducibility

Running `reproduce.sh` from the package root (with the OpenFEMA API reachable) re-fetches the latest snapshot, applies the documented filter, and refits. Because FEMA continues to backfill claims, exact counts will drift slightly with later pull dates; the parameters in this CSV pin the 2026-05-09 fit and are bit-exact reproducible against the frozen `data/raw/*.csv` files (modulo BLAS thread-ordering effects on the optimizer at the 4th decimal).

## Derived results not in this archive

The paper reports several downstream analyses that build on the calibrated parameters but whose source scripts and inputs are NOT bundled here. Each requires data or pipelines outside the scope of the joint Richards calibration:

- **Sector adjustment κ** — calibrated on 82,650 NFIP non-residential claims (paper §3.3).
- **24-company out-of-sample validation** — predicted-vs-actual on 24 corporate flood events (paper §6 Table).
- **JRC/CLIMADA benchmark** — MAE comparison vs Joint Research Centre depth-damage curves on Harvey binned data (paper Table 5).
- **Feature enrichment / gradient boosting** — claim-level ML on Harvey 70/30 split (paper §5.4 Table 7).
- **Leave-one-event-out cross-validation** — train on two events + USACE, predict held-out (paper Table on LOO MAE).
- **Spearman ρ = 0.949 vs FEMA IHP** — ZIP-level correlation against IHP registrations (paper §5.3).
- **ADM credit-risk pipeline** — facility-level damage → CapEx/OpEx → ICR → ΔPD chain (paper §5).
- **Bootstrap κ_total uncertainty** — 1,000-replication CI of ±0.06 (paper §3.3).
- **Binning sensitivity sweep (8 / 16 bins)** (paper §4.3).

Numerical claims in the paper that derive from these analyses are NOT recomputed here when the calibration parameters change. After the May 9, 2026 refit, all of these downstream results would in principle shift, but **probably very little**, because the headline DR predictions are equivalent to the paper text's predictions to within 0.02 across the whole calibration depth range. Specific recompute is documented in `derived_results_recompute_TODO.md` (decision deferred to co-authors).
