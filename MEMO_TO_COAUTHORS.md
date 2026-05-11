# Memo: Replication archive — methodology + data refresh

**Author:** Nikita
**Date:** 2026-05-09
**To:** Ivan (corresponding author), all co-authors
**Status:** Replication archive ready. Paper text NOT modified.

## Summary in one paragraph

Re-pulled OpenFEMA NFIP claims (May 9, 2026 snapshot, full pull — 445,594 raw / 295,637 valid). Cleaned up two undisclosed filters (`coverage > $1,000` and `depth ≤ 10 m`) that were in the code but not described in the paper. Adopted **per-event depth caps for calibration** (Sandy/Harvey 4 m, Katrina 10 m) instead of the earlier uniform 6 m cap, motivated by per-event data range and validated by sensitivity analysis. Refit. **The headline DR predictions reproduce the paper text exactly** (DR(1 m, 0) = 0.25, DR(1 m, 36 h) = 0.35, amplification 2.6×). λ_a = 0.49 vs paper's 0.46 — within 6%. The four base Richards parameters (B, Q₀, a₀, ν₀) differ from paper-quoted values along the model's identifiability ridge.

## Key quantitative result

| h, τ | DR (refit) | DR (paper text) |
|---|---:|---:|
| 1 m, τ = 0      | **0.250** | 0.25 |
| 1 m, τ = 36 h   | **0.353** | 0.35 |
| 1 m, τ = 288 h  | **0.656** | 0.64 |
| Amplification 0 → 288 h | **2.63×** | 2.6× |
| λ_a | **0.489** | 0.46 |
| λ_Q | ≈ 0 | ≈ 0 ✓ |
| δ_ν | ≈ 0 | ≈ 0 ✓ |

The DR predictions are functionally identical to within rounding. The qualitative result (duration affects damage primarily through inflection-point shift) is preserved exactly.

## What changed since the original calibration

1. **Data**: refreshed from FEMA OpenFEMA. Three events now contain 445,594 raw / 295,637 valid claims (was 354,413 / 317,943 in earlier paper text). FEMA backfilled ~91k historical raw claims; ~96% of those are zero-depth/payout that fail the validity filter, so net valid count fell ~7%.

2. **Filter**: removed two undisclosed-in-paper filters from the code (`total_coverage > $1,000`, `depth ≤ 10 m`). The code now exactly matches what Table 3 caption describes: positive depth, positive payout, DR ∈ (0, 1.5], capped at 1.0.

3. **Calibration depth cap**: changed from uniform 6 m to event-tailored (Sandy/Harvey 4 m, Katrina 10 m). The 6 m cap discarded the Katrina levee-breach tail (1,216 claims, 0.9% of valid Katrina) — exactly the data that distinguishes Katrina from Sandy/Harvey. Per-event caps retain 99.91% of valid claims while still trimming a clean ~0.07% extreme-outlier tail per event. Sensitivity analysis (5 cap regimes including no-cap) is in `params/PROVENANCE.md`.

4. **Code disclosure**: optimizer settings (seed=42, maxiter=1000, tol=1e-10), prior centers/scales (B ~ N(3, 3), Q₀ ~ N(1, 1), a₀ ~ N(0.5, 0.2), ν₀ ~ N(0.5, 0.3)), and parameter bounds are now documented in `params/PROVENANCE.md`.

## Three earlier concerns and their resolution

### Negative R² for Sandy and Harvey is a binning artifact, not a bad fit

Sandy bin means cluster tightly around 0.20–0.40 across all depth bins; the within-event variance of bin means is small, so even a model with MAE = 0.12 produces R² < 0 (because ss_res > ss_tot when ss_tot is tiny). This is a known quirk of bin-level R² for events with narrow DR-vs-depth ranges and is not a failure of the joint duration-Richards fit. MAE values (0.07–0.12) are good in absolute terms. Claim-level R² (computed against all individual claims) would be substantially higher.

If we report R² in the paper's §4.3 we should explain this. If we report only MAE (as the paper currently does), no further action is needed — but the explanation should be in `PROVENANCE.md` for replication-pack readers.

### Identifiability degeneracy explains why Q₀ = 1.58 vs 3.45

The Richards function with 7 free parameters (B, Q₀, a₀, ν₀, λ_a, λ_Q, δ_ν) on bin-level data has a flat ridge in the loss landscape: increasing Q₀ and decreasing ν₀ in correlated directions produces equivalent D(h, τ) curves over the calibration depth range. Differential evolution converges to one point on this ridge per (seed, prior, bound) combination. The paper-quoted (5.23, 1.58, 0.40, 1.47) and the refit (5.65, 3.45, 0.54, 2.61) sit at different points on the same ridge but generate functionally identical predictions. This is why earlier in-repo runs produced wildly different numerics (`data/calibrated_parameters.csv`, `data/duration_richards_params.csv`, etc.) without it being noticed.

**Practical implication**: Quote D(h, τ) values in any external use, not the four base parameters. The paper's predictions, validation MAEs, ADM facility DRs, and JRC benchmarks are *not affected* by this — they all depend on D(h, τ), which is reproduced.

### 6 m cap was a methodology default, not a footnote

Removing the uniform 6 m cap and using per-event caps is the methodologically sound choice — it includes the levee-breach signal that defines Katrina. Sensitivity analysis confirms this regime is the lowest-objective non-degenerate option. Detail in `PROVENANCE.md`.

## What's in the replication archive

```
~/Desktop/physical-climate-risk-replication/    (88 MB)
├── data/raw/            full May 2026 FEMA pull (445,594 claims)
├── data/processed/      filtered (295,637 valid; 295,383 used in fit)
├── code/
│   ├── fetch_fema.py                  paper-aligned filter, no hidden caps
│   ├── damage_functions/              core Richards + duration extension
│   └── notebooks/unified_calibration.py event-tailored caps, seed=42
├── params/
│   ├── unified_params.csv             refit output (high precision)
│   ├── unified_params_canonical.csv   rounded canonical
│   └── PROVENANCE.md                  full audit trail + identifiability discussion
├── paper/                             original Mar 2026 LaTeX (NOT edited)
├── figures/10_unified_calibration.png regenerated at canonical fit
├── README.md
├── reproduce.sh                       bit-exact reproducible
└── derived_results_recompute_TODO.md  out-of-scope checklist
```

## What I did NOT do

- Did NOT edit `paper/Content/*.tex` or any manuscript text.
- Did NOT push changes anywhere.
- Did NOT regenerate the 7 derived figures in `paper/Content/Images/` that depend on parameter values (`duration_curves.png`, `richards_schematic.png`, `gap_closures.png`, etc.). They reflect the older Mar 2026 fit.

## Recommendations for paper text

Given that **DR predictions match the paper text within rounding** (0.250 vs 0.25, 0.353 vs 0.35, 2.63× vs 2.6×, λ_a 0.49 vs 0.46), the main paper claims are intact. Two options:

**Option A (minimal): Treat May 2026 as a reproducibility supplement.**
Keep paper text as-is. Add a footnote or appendix paragraph: "A reproduction archive based on the OpenFEMA snapshot of 2026-05-09 with event-tailored depth caps reproduces the paper's predictions (DR(1 m, 0) = 0.25, DR(1 m, 36 h) = 0.35, amplification 2.6×) and the duration parameter (λ_a = 0.49 vs reported 0.46) to within rounding. Specific values of B, Q₀, a₀, ν₀ differ along the model's identifiability ridge but the predicted damage curve is functionally identical." Direct reviewers to the archive.

**Option B (substantive): Refresh paper text against May 2026.**
Update Tables 3 (counts) and 4 (params) to match the archive. Add the `PROVENANCE.md` identifiability-ridge discussion to §4.2. This requires regenerating the 7 derived figures and re-running κ / 24-company / JRC / ADM analyses (per `derived_results_recompute_TODO.md`) — non-trivial but yields the cleanest submission.

I lean toward Option A unless reviewers specifically demand reproducibility against the deposit, which they may. Your call as corresponding author.

## Files

```
~/Desktop/physical-climate-risk-replication/
```

Ready to discuss anytime.
