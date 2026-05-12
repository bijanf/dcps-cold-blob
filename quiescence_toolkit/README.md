# quiescence_toolkit

**The Quiescence Signature: a theory of phase coherence in geostrophic turbulence.**

Companion code, data fetchers, and theoretical development for the
manuscript *The Quiescence Signature: A theory of phase coherence in
geostrophic turbulence* (Fallah & Rostami, in preparation).

This toolkit is **separate from** the empirical paper
*Baroclinic decoupling of the North Atlantic Cold Blob: surface
unprecedented in 12,000 years, deep unchanged* (Fallah & Rostami,
submitted, repository [`dcps-cold-blob`](https://github.com/bijanf/dcps-cold-blob)).
The two papers are companions: this one provides the mechanistic
framework, the other applies it to the modern Cold Blob.

## What's here

| Path | Contents |
|---|---|
| `src/theoretical_curve.py` | Fokker-Planck-derived `r(D)` closures (Bessel, weak-coupling, phenomenological) with curve fits to observed NA scatter. Result: simple `(1+τD)^(-1/2)` closure fits with MSE = 0.031. |
| `src/toy_model.py` | 1-D reduced-order stochastic Kuramoto with Gaussian jet, EKE-prescribed multiplicative noise, parameter sweep over `(K, U₀)`. **Honest finding:** the 1-D model with canonical parameters does not cleanly reproduce the negative correlation; 2-D dynamics are required. |
| `src/quiescence_index.py` | Operational scalar Q = −ρ(⟨r_loc⟩, ∣∇SSH∣) with spatial-block permutation p-value. Benchmark observed: Q_NA = 0.32, Q_NP = 0.35, Q_ACC = 0.46. |
| `paper/theory_section.tex` | ~2000-word LaTeX theory section, 9 subsections, 5 figures (T1–T5). |
| `paper/predictions.md` | Five falsifiable predictions (P1–P5) with decision rules. |
| `paper/cover_letter.tex` | Submission cover letter to Nature Communications. |
| `paper/roadmap_10yr.md` | Ten-year research programme on Quiescence universality across geostrophic turbulence. |
| `results/` | Pre-computed JSON tables and figures (theoretical curve fit, 1-D toy outputs, Q-index table). |

## What's planned but not yet executed

| Item | Status |
|---|---|
| ERA5 daily Z500 fetch + atmospheric Quiescence test (NH and SH 30–60°) | Code stub, fetch pending |
| CMEMS/AVISO SLA fetch + altimetry Quiescence test | Pending CDS account |
| CMIP6 HighResMIP Q-vs-resolution Mann-Whitney U test | Pangeo fetch pending |
| Temporal frequency-detuning sliding-window analysis | Pending |
| Jupyter replication notebooks | Pending |

## Key findings so far

1. **Phenomenological r(D) closure fits well.** Simple
   `r ≈ (1+τD)^(−1/2)` gives MSE = 0.031 in NA. The rigorous Bessel
   form derived from Fokker-Planck has numerical issues at small D.
2. **1-D toy fails.** Canonical 1-D Gaussian-jet model with
   nearest-neighbour Kuramoto does not reproduce ρ < 0 reliably.
   2-D dynamics on a basin grid (Round 2 toy in `dcps-cold-blob`)
   do reproduce ρ = −0.25.
3. **Q-index defined and benchmarked.** Q ∈ [−1, 1]; Q > 0 indicates
   Quiescence Signature. Observed Q = 0.32 (NA), 0.35 (NP), 0.46 (ACC).
4. **Temporal predictions falsified.** The originally pre-registered
   `ρ(R(t), −dΨ/dt) ≤ −0.4` is not supported by RAPID-era data;
   the framework is re-formulated around frequency detuning.

## Reproducing the current results

```bash
cd quiescence_toolkit
pip install xarray numpy scipy matplotlib
python src/theoretical_curve.py    # produces fig_T2_theoretical_curve.pdf
python src/toy_model.py            # produces fig_T4_toy_1d.pdf + sweep
python src/quiescence_index.py     # writes Q_index_table.json
```

All three scripts rely on cached arrays from the
`dcps-cold-blob` repository (`dcps/cache/eke_eddy_resolving/`,
`dcps/cache/multi_basin/`, `dcps/cache/spatial_perm/`).

## License

MIT. See parent repository for details.

## Citation

Until publication, please cite the repository commit hash directly.
