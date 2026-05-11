# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository contains the manuscript scaffold for the DCPS-AMOC paper (`manuscript/` — modular `sections/*.tex`, `refs.bib`, `Makefile`) plus this guide and the original `roadmap.md`. No DCPS analysis code lives here yet — the data and the diagnostic pipeline are external.

## External data and pipeline

The PI maintains a separate AMOC Reanalysis Diagnostic Pipeline (ARDP) at `~/Documents/AMOC_renalysis/` (GitHub: `bijanf/ardp`, MIT). It already holds:

- **ORAS5 reanalysis 1958–2023** (~730 GB) at `data/oras5/` with monthly `sosstsst` (SST), `sossheig` (SSH/SLA), `sosaline` (SSS), `vosaline`/`vomecrty` (3D salinity/V).
- **GLORYS12V1 1993–2025** (~306 GB) at `data/glorys12/`.
- **ECCO-V4r4** (~21 GB) and **SODA 3.15.2** (~1 GB).
- **RAPID-MOCHA 26.5°N** as `data/external/rapid_moc_monthly.nc` (use this — *not* the broken `data/rapid_moc.csv` at the top of `data/`, which is an HTML error page).
- **SAMBA 34.5°S** at `data/external/samba_moc_monthly.nc` — southern-boundary counterpart.
- **Pre-computed yearly AMOC at 26.5°N** for all five products at `data/results/yearly_amoc26n_*.npz`.
- ~90 Python scripts under `scripts/`, mostly for AMOC streamfunction / F_ovS diagnostics. **None of them implement Hilbert / Kuramoto / Chimera / transfer-entropy** (verified by grep) — DCPS is genuinely a new analytical layer to add on top of this data backbone.

**Implication for Phase 1:** the roadmap's "download OISST via ERDDAP and RAPID from rapid.ac.uk" is largely obsolete. Phase 1 reduces to *subsetting* the existing ORAS5 cache to the North Atlantic basin and preprocessing (climatology removal, detrend, bandpass). OISST may still be useful as a robustness check against ORAS5, but is not the primary path.

The Methods section of the manuscript (`manuscript/sections/02_methods.tex`) reflects this — it cites ORAS5 (Zuo et al. 2019) as primary and OISST (Reynolds 2007 / Huang 2021) as one of several robustness alternatives.

## Project: the DCPS Framework for the AMOC

The user is the Principal Investigator on a manuscript proposing the **Delay-Coupled Phase-Synchronization (DCPS) Framework** for the Atlantic Meridional Overturning Circulation. The core scientific claims to keep in mind when writing code, because they constrain what every analysis must produce:

1. **AMOC volume transport (Sverdrups) ≡ Kuramoto order parameter R(t).** Phase 3 must demonstrate this empirically by correlating a network-derived R(t) against RAPID-MOCHA Sverdrup observations.
2. **The North Atlantic "Cold Blob" is a spatiotemporal Chimera state.** A *local* order parameter on a sliding spatial window should reveal high synchronization in the subtropics coexisting with desynchronization localized over the subpolar Cold Blob.
3. **Transfer entropy from subtropics → subpolar nodes drops *before* variance/critical-slowing-down rises**, evidencing R-tipping (rate-induced) rather than classical B-tipping (saddle-node).

These three hypotheses map to Figures 1, 2, 3 of the manuscript respectively. Code structure should make those three figures the load-bearing outputs.

## Execution discipline (the PI's explicit rules)

These are stated in `roadmap.md` and must be honored:

- **Do not execute the whole roadmap in one pass.** Implement one phase at a time, then stop and wait for the PI to review outputs before starting the next.
- **Phase 1 first**, scoped to: a `requirements.txt` and Python scripts that download + preprocess NOAA OISST v2 (via ERDDAP, `xarray`, no API key) and RAPID-MOCHA AMOC data. Nothing further until approval.
- **Persist preprocessed data to disk** (NetCDF or pickle) so Phase 2+ math iteration does not re-download decades of ocean data. This is non-negotiable — re-downloading on each tweak is the failure mode the PI is explicitly trying to avoid.
- **Draft LaTeX manuscript sections incrementally** as each phase's results land, not at the end.

## Phase pipeline (reference, not a license to skip ahead)

| Phase | Inputs | Key transforms | Outputs |
|---|---|---|---|
| 1. Data | OISST v2 SST/SLA, RAPID-MOCHA | coarse-grain to 2°×2° or 5°×5°; remove climatology; detrend; bandpass 1–10 yr | cached `.nc`/`.pkl` of node anomalies + Sv time series |
| 2. Math engine | Phase-1 cache | `scipy.signal.hilbert` per node → φ_i(t); compute R(t) | global R(t) time series |
| 3. Hypothesis tests | R(t), Sv, φ_i(t) | Pearson(R, Sv); local r on sliding spatial window (cartopy map); transfer entropy via `pyinform` on 10-yr sliding window | Figures 1, 2, 3 |
| 4. Manuscript | Phase-3 figures | LaTeX (Nature Physics / PNAS style) | manuscript draft |

## Known pitfalls called out by the PI

- **Hilbert edge effects:** the first/last ~6 months of `scipy.signal.hilbert` output are unreliable. If Figure 1's correlation looks bad at the time-series boundaries, trim the ends rather than chasing a math bug.
- **Domain bounds for Phase 1:** Equator → 75°N, 80°W → 0° (North Atlantic basin). Timeframe 1990 → present.

## What to do on first invocation

If the PI has not yet given a phase-specific instruction, the expected opening move is to acknowledge the roadmap, then produce `requirements.txt` and Phase 1 code only — exactly as `roadmap.md` instructs at its end. Do not proactively scaffold Phase 2+.
