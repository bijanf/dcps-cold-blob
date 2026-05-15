# P6 — key findings ready for the paper

Generated 2026-05-15 morning on PIK after the overnight + Tier 1 (IC ensemble,
story figure, flex-fit) + Tier 2D (member ensemble) + Tier 2E (multi-basin)
runs. See `summary.md` for the per-realization table and per-cohort
descriptive statistics; this file is the **narrative** for the manuscript.

## Headline (one-sentence verdict)

**DLESyM v1 reproduces the Quiescence-Signature *spatial pattern* in the North
Atlantic (`Q_X = +0.27`, `p_perm < 0.001`) but the registered parametric form
`(1+τ·EKE)^(-1/2)` with `τ=144` fails its goodness-of-fit ceiling
(`χ²_ν = 5.59`, ceiling 5.0) → registered verdict = `falsified`.**

## The four numbers that decide the verdict

| metric | primary (30-yr, atmos=0, ocean=0) | registered threshold | passes? |
|---|---|---|---|
| `Q_X` | +0.272 | ≥ 0.20 | ✓ |
| `p_perm` | 0.001 | < 0.01 | ✓ |
| `1/3 ≤ τ̂/144 ≤ 3` | τ̂/144 = 0.26 | ratio in [0.33, 3] | ✗ (off by ~4×) |
| `χ²_ν ≤ 5` | 5.59 | ≤ 5.0 | ✗ (12% over) |

Two thresholds pass, two fail → registered branch: **falsified by chi²**.
Per `OPERATIONS.md §4`, action = "narrow the regime-of-validity in
`\cref{si:deriv-regime}`".

## Robustness — IC ensemble (Tier 1A, 8 ICs 1985→2020)

All 7 new ICs **also** return `verdict=falsified`. The verdict is not a
function of IC choice. Detail:
- `Q_X` median across 7 new ICs = 0.227 (range 0.014 → 0.265)
- `τ̂` median = 35.9 (range 30.5 → 40.2) — **τ̂ is remarkably stable** across
  IC choice; clustering around 30-40 regardless of the IC year
- `χ²_ν` median = 6.76 (range 5.32 → 7.70) — every IC is over the 5.0 ceiling
- The 2010 IC (the primary) gave one of the strongest `Q_X` (0.272); 1995
  was the weakest (0.014).

**Implication**: the χ² failure is not noise; it's a real form-mismatch the
model has, independent of which year you initialize from.

## DLESyM member ensemble (Tier 2D, 4 × 4 = 16 atmos × ocean combinations)

- **7/16 members deferred by drift** (`drift > 0.02 K/yr` over 30 yr). All 7
  pair an `ocean_model_idx ∈ {1,2,3}` with various atmos members — i.e. the
  non-(0,0) ocean checkpoints are climatically unstable on this timescale.
  The (atmos=*, ocean=0) members are the only consistently stable family.
- Among the 9 non-deferred members: `Q_X` spans **−0.18 to +0.27**.
  Several members **lose the anti-correlation entirely** (Q_X = −0.18, −0.08,
  −0.01) and one even reverses sign. The (0,0) member happens to be on the
  **strongest** end of the model-internal Q_X distribution.
- median `Q_X` across non-deferred members = +0.051; **mean = +0.069**.

**Implication**: the primary verdict is at the *favourable* tail of DLESyM's
internal-ensemble spread. The honest verdict, averaged over DLESyM members,
is closer to "no signal" than "weak signal".

## Multi-basin (Tier 2E)

Observational anchors from `eke_quiescence_eddy_resolving.py` (GLORYS12,
2000–2023):

| basin | observed ρ(<r_loc>, EKE) | observational verdict | DLESyM Q_X | DLESyM χ²_ν | DLESyM verdict |
|---|---|---|---|---|---|
| North Atlantic | −0.250 | consistent (in [-0.30, -0.20]) | **+0.272** | 5.59 | falsified (χ²>5) |
| North Pacific | **−0.523** | UPHELD (≤ -0.30) | **−0.029** | 5.74 | falsified (no signal) |
| Southern Ocean | **−0.531** | UPHELD (≤ -0.30) | **+0.239** | **3.08** | deferred |

**The three-basin picture is the strongest "where does the model work" result of the run.**

Observations: the Quiescence law is **strong (UPHELD)** in both the North
Pacific (ρ=−0.52) and Southern Ocean (ρ=−0.53), and only **consistent** (not
strictly UPHELD) in the North Atlantic (ρ=−0.25). So nature itself prefers
the higher-EKE-variance basins for the law's expression.

DLESyM v1:
- **North Atlantic**: captures the spatial pattern (Q_X=+0.27, p<0.001) but
  the parametric form's χ² ceiling fails by 12% → **falsified**.
- **North Pacific**: completely **loses the anti-correlation** (Q_X=−0.03,
  p=0.48 — i.e. no signal). The model does not see the law where
  observations say it is strongest. **Falsified by χ² and by Q_X<floor.**
- **Southern Ocean**: captures the pattern (Q_X=+0.24, p<0.001) AND fits
  with χ²_ν=3.08 (well below 5.0 ceiling). The only failure is τ̂=29.5,
  outside the registered ×3 band of τ_obs=144. **Verdict: deferred** —
  i.e. pattern + significance + fit quality all support the law's
  **functional class**; only the specific decay constant differs by ~5×.

**Implication for the manuscript**: the model's **regime of validity** is
basin-dependent. NA and SO show the law's *direction* and (in SO) its
*shape*, but the *magnitude* of the decay (τ̂) is universally smaller than
the registered τ_obs=144 — by a factor of 3-5 across all basins where the
pattern is detected. The North Pacific is where DLESyM's atmospheric–ocean
coupling breaks down for this diagnostic; this is a real model-skill
constraint, not a falsification of the law in nature.

## Constructive follow-up: which functional form *does* fit?

`p6_flex_fit.py` re-fit DLESyM `<r_loc>` vs GLORYS12 EKE in the North Atlantic
on four parametric families:

| family | k | χ²_ν | AIC | ΔAIC vs registered |
|---|---|---|---|---|
| registered: `(1+τEKE)^-0.5` | 1 | 5.59 | -1480 | **+1916** |
| power: `(α+τEKE)^-β`        | 3 | **0.89** | -3396 | 0 (best) |
| saturating exp              | 3 | 0.89 | -3393 | +3 |
| exponential decay           | 2 | 0.93 | -3358 | +38 |

All three non-registered families dominate by ΔAIC > 1900. The data are
**very well described** by a more flexible decay; the *direction* of the
law (negative `<r_loc>`–EKE relationship) is right, but the *magnitude* of
the registered decay (τ=144) over-predicts the EKE-induced suppression of
coherence by roughly 5×.

This becomes the seed for a P7 prediction in a follow-up: "the
generalized `(α + τ·EKE)^-β` form with β ≪ 0.5 describes the deep-learning
emulator's coherence statistics; this differs from the theoretical
prediction β = 0.5 derived for fully-developed quasi-geostrophic turbulence."

## Suggested manuscript edits

1. **`03_results.tex`**: add a paragraph reporting the primary verdict and
   the IC-ensemble robustness; cite `p6_verdict_postprocess.json` and
   `summary.csv`.
2. **`04_discussion.tex`**: update P6 from "registered, execution pending"
   to "registered and falsified by χ² ceiling — see `\cref{si:deriv-regime}`."
3. **`si:deriv-regime`**: narrow the regime-of-validity to North Atlantic;
   show the Pacific contrast (DLESyM does not reproduce the law there even
   though observations do); explain that this restricts the law to basins
   where the model's atmosphere–ocean coupling and HEALPix sampling are
   sufficient.
4. **New SI subsection (suggested)**: "Functional form refinement" — show
   the flex-fit result with the generalized form fitting at χ²_ν=0.89, and
   pre-register the generalized form as a future test (P7).

## Files for the manuscript

- `figures/p6_story.pdf` — 4-panel: `<r_loc>` map, EKE map, scatter+fit, residual
- `figures/p6_ensembles.pdf` — Q_X / χ²_ν boxplots over IC + member cohorts
- `data/flex_fit.json` — all four families' fit parameters + AIC/BIC
- `data/r_loc_atlantic_2deg.nc` — the `<r_loc>(x)` field used in panel (a)
- `summary.csv` — flat table of all 26+ verdicts
- `verdicts/*.json` — every individual run's verdict
