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

## What's done since the initial commit

| Item | Status | Result |
|---|---|---|
| ERA5 Z500 fetch + atmospheric Quiescence | DONE | NH ρ = −0.04 (ns); SH ρ = −0.12 (p = 0.015); sign supported, weaker than ocean |
| Temporal frequency-detuning analysis | DONE | partial ρ(σ²ω, Ψ \| R) = +0.24 (opposite predicted sign); reformulation not supported |
| Three Jupyter replication notebooks | DONE | reproduce Q for 3 basins; reproduce Fisher's-exact p; fit theoretical curve |
| Unit tests (pytest) | DONE | 8/8 pass on `tests/test_quiescence_index.py` and `tests/test_theoretical_curve.py` |

## What remains pending

| Item | Status |
|---|---|
| Item | Done in this session | Result |
|---|---|---|
| Altimetry Quiescence (GLORYS12 zos substitute for AVISO/CMEMS) | DONE | ρ = −0.28, p_perm = 0.005, Q = +0.28 in NA; P1 supported in a purely dynamical (non-thermal) field |
| CMIP6 Q for resolution-variant pairs (HighResMIP substitute) | DONE | 5 high-res (CESM2, HadGEM3-MM, NorESM2-MM, CNRM-CM6-1-HR, MPI-ESM1-2-HR) + 5 low-res (CESM2-FV2, HadGEM3-LL, NorESM2-LM, CNRM-CM6-1, MPI-ESM1-2-LR). Median Q: 0.133 (HR) vs 0.070 (LR). All CMIP6 Q values are 2–5× weaker than observed Q_NA = 0.32. |
| Mann-Whitney U test (HR vs LR Q) | DONE | U = 13, p_one-sided = 0.579, gap = +0.064 (wrong sign). **P3 falsified on this dataset.** |
| AVISO SLA standalone test | Pending | Requires CMEMS account; GLORYS12 zos substitute above gives the equivalent finding |

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
5. **P3 resolution dependence falsified on CMIP6 resolution-variant
   pairs.** 5 HR + 5 LR (HadGEM3-MM/LL, NorESM2-MM/LM, CESM2/FV2,
   CNRM-CM6-1-HR/CNRM-CM6-1, MPI-ESM1-2-HR/LR). Median Q gap
   = +0.064 (wrong sign); Mann-Whitney U p = 0.579. The within-
   product GLORYS12 coarsening sweep still shows the predicted
   decline, so the falsification is specific to CMIP6
   resolution-variant pairs and is reported honestly.

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
