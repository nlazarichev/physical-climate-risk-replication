# Overleaf upload package — A duration-dependent flood depth-damage function

Drop-in folder for Overleaf. Compiles with `pdflatex main.tex` (Elsevier `elsarticle` class, default).

## Structure

```
physical-climate-risk-overleaf/
├── main.tex                                ← entry point
├── highlights.tex                           ← 5 bullet points, max 85 chars each
├── cover_letter.tex                         ← response to 3 reviewers (R1, R2, R3)
├── README.md                                ← this file
│
├── Common/
│   └── bibliography.bib                     ← Elsevier-Harvard format, 60+ entries
│
└── Content/
    ├── abstract.tex                          ← Background / Methods / Results / Conclusions
    ├── introduction.tex                      ← §1 with §1.1 Contribution, §1.2 Data req
    ├── litReview.tex                         ← §2 Related Work (with 9 new citations)
    ├── symbols.tex                           ← Table 1 notation
    │
    ├── Methods/
    │   ├── framework.tex                     ← §3 framework + Fig pipeline_scheme
    │   ├── damage_function.tex                ← §3.1 Eq.1 Richards (B(x-a) form)
    │   ├── duration_extension.tex             ← §3.2 Eq.2 + duration-curves figure
    │   ├── sector_adjustment.tex              ← §3.3 κ + uncertainty paragraph
    │   └── pd_translation.tex                 ← §3.4 ICR-based credit translation
    │
    ├── empirical_calibration.tex             ← §4 Tables 3+4+5+6+7+8
    │
    ├── ResultsDiscussion/
    │   ├── adm.tex                           ← §5 ADM application:
    │   │                                       §5.1 exposure, §5.2 calibrated DR, §5.3 OpEx,
    │   │                                       §5.4 credit translation,
    │   │                                       §5.5 τ sensitivity (NEW),
    │   │                                       §5.6 calibrated vs crude,
    │   │                                       §5.7 insurance sensitivity (moved from appendix)
    │   ├── harvey_event_study.tex             ← §6 24-company OOS:
    │   │                                       §6.4 Discussion + Geographic transferability + Excluded events
    │   └── discussion.tex                     ← §7 Discussion:
    │                                            limitations, identification of τ vs covariates,
    │                                            predictive use, parameter uncertainty (200-rep bootstrap),
    │                                            CLIMADA comparison, hazard characterisation
    │
    ├── conclusion.tex                         ← §8 Conclusion + 3 findings + duration assignment + scope
    └── appendix_insurance.tex                 ← LEGACY: content moved to §5.7; not included in main.tex
```

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or in Overleaf: project compiles automatically with the default settings (Latexmk).

## Authoring conventions

- Custom macros defined in main.tex preamble:
  - `\DF` = $\text{DF}$ (damage function symbol)
  - `\ECL` = $\mathbb{E}[\text{CL}]$ (expected climate loss)
  - `\ICR` = ICR (interest coverage ratio)
- Cross-references to revised passages are listed in `cover_letter.tex` for each
  R-numbered reviewer comment (e.g. R1.1 → Section 7 Limitations; R2.9 → Section 6.4
  Geographic transferability paragraph).

## Cover letter

`cover_letter.tex` responds to 3 reviewers point-by-point covering 22 distinct comments
(R1.1–1.4, R2.1–2.9, R3.1–3.9). It compiles separately:

```bash
pdflatex cover_letter.tex
```

The PDF version `cover_letter.pdf` (~70 KB, 11 pages) is the canonical signed-off
version; this `.tex` is provided for editor-side convenience.

## Highlights

`highlights.tex` is a standalone 5-bullet file (≤85 chars/bullet, Elsevier requirement).
Compiles to a 1-page PDF. Submitted alongside the main manuscript.

## Numerics

Calibration parameters used throughout the paper:
- $B = 5.23$, $Q_0 = 1.58$, $a_0 = 0.40$, $\nu_0 = 1.47$, $\lambda_a = 0.46$
- DR(1m, τ=288h) = 0.64; amplification 2.6×

These are the **submitted-paper** values from the Mar 2026 OpenFEMA snapshot (317,943 valid claims).
The accompanying replication archive (`physical-climate-risk-replication/`) re-fits the same
methodology against a May 2026 OpenFEMA snapshot (295,637 valid claims) and obtains parameters
along the model's identifiability ridge (5.65/3.45/0.54/2.61/0.49) that produce **functionally
equivalent D(h, τ) predictions** at h ≈ 1 m. See `params/PROVENANCE.md` in the replication
archive for the full audit trail.

## Reviewer-introduced sections (cross-referenced in cover_letter.tex)

- §5.5 Sensitivity to τ assignment (Reviewer 1.3)
- §5.7 Insurance coverage sensitivity (moved from Appendix per cover-letter summary point 1)
- §6.4 Geographic transferability + Excluded events (Reviewer 2.9)
- §7 Identification of τ vs confounded covariates (Reviewer 1.2)
- §7 Predictive use and how to assign τ in practice (Reviewer 1.3)
- §7 Parameter uncertainty — 200-replicate bootstrap (Reviewer 2.3)
- §7 CLIMADA comparison (Reviewer 2.6)
- §7 Hazard characterisation — velocity discussion (Reviewer 3.9)
- 9 new citations in §2 (Reviewer 2.1: A–I)
- Section 2 opening intensity-measure terminology (Reviewer 3.4)
- Fig 3 caption: DF≈0.22 explanation (Reviewer 3.5)
- Figs 4 + 6 captions: h_min = 0.05 m (Reviewer 3.6)
- Eq 5 E[CL] notation fix (Reviewer 3.7)
- Conclusion × → "times" (Reviewer 3.8)
- "Exclusively" → "primarily" globally (Reviewer 2.2)
- CRediT authorship statement (NHR requirement)
- Ethics statement, Funding, Data availability, AI declaration (NHR end-matter)
