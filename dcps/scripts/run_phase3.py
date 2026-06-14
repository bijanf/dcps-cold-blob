"""End-to-end Phase 3: H1, H2, H2', H3 tests and the three manuscript figures.

Outputs:
  cache/phase3_results.json     -- numerical results of all three tests
  cache/phase3_ews.nc           -- sliding-window TE / variance / AC1
  manuscript/figs/fig1_R_vs_Sv.pdf
  manuscript/figs/fig2_chimera_map.pdf
  manuscript/figs/fig3_TE_vs_CSD.pdf
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.h1 import lowpass_1yr, standardise, test_h1
from dcps.h2 import (
    COLD_BLOB_BOX,
    SUBTROPICAL_BOX,
    test_h2,
    test_h2_prime,
)
from dcps.h3 import sliding_window_ews, test_h3
from dcps.io import load_rapid_amoc

MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
PHASE3_RESULTS = CACHE_DIR / "phase3_results.json"
PHASE3_EWS = CACHE_DIR / "phase3_ews.nc"


def main():
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)

    phase1 = xr.open_dataset(CACHE_DIR / "phase1_oras5_NA_2deg.nc")
    phase2 = xr.open_dataset(CACHE_DIR / "phase2_R.nc")
    phases = xr.open_dataset(CACHE_DIR / "phase2_phases.nc")
    rapid = load_rapid_amoc()

    print("=" * 60)
    print("H1 test: Pearson rho between RAPID Sv and basin R(t)")
    print("  (Methods pre-registered three R variants: SST, SSH, pooled)")
    print("=" * 60)
    h1_variants = {}
    for label, R_var in [("R_sst", phase2.R_sst), ("R_ssh", phase2.R_ssh),
                          ("R_pooled", phase2.R_pooled)]:
        res = test_h1(R_var, rapid)
        h1_variants[label] = res
        verdict = ("strong" if res.pass_strong else
                   "suggestive" if res.pass_suggestive else "falsified")
        print(f"  {label}: rho = {res.rho:+.3f}  CI ({res.bootstrap_low:+.3f}, "
              f"{res.bootstrap_high:+.3f})  p = {res.p_value:.2e}  -> {verdict}")
    h1 = h1_variants["R_sst"]   # primary
    print(f"  primary (SST): n_months = {h1.n_months}, "
          f"window {h1.overlap_start} .. {h1.overlap_end}")

    print()
    print("=" * 60)
    print("H2 test: pre-registered area thresholds (CB incoherent + ST coherent)")
    print("=" * 60)
    h2 = test_h2(phase2.r_loc_sst)
    print(f"  fraction of years passing both thresholds: {h2.fraction_years_passing:.2f}")
    print(f"  pass_h2 = {h2.pass_h2}")
    print("  per-year areas (km^2):")
    for y in sorted(h2.a_inc_per_year):
        print(f"    {y}: A_inc(CB) = {h2.a_inc_per_year[y]:.2e}, "
              f"A_coh(ST) = {h2.a_coh_per_year[y]:.2e}")

    print()
    print("=" * 60)
    print("H2' fallback: inter-region phase-coupling concentration")
    print("=" * 60)
    h2p = test_h2_prime(phases.sst_phase)
    print(f"  R_CB mean = {h2p.R_CB_mean:.3f}")
    print(f"  R_ST mean = {h2p.R_ST_mean:.3f}")
    print(f"  C (concentration) = {h2p.C:.3f}")
    print(f"  mean |Phi_CB - Phi_ST| = {h2p.mean_phase_diff_deg:.1f} deg")
    print(f"  pass_h2p = {h2p.pass_h2p}")

    print()
    print("=" * 60)
    print("H3 test: TE collapse vs CSD rise on 10-yr sliding window")
    print("=" * 60)
    ews = sliding_window_ews(phase1.sst_anom, window_years=10)
    ews.to_netcdf(PHASE3_EWS)
    print(f"  EWS time range: {str(ews.time.min().values)[:10]} .. "
          f"{str(ews.time.max().values)[:10]}")
    print(f"  TE  (mean): {float(ews.TE_st_to_sp.mean()):.4f} bits "
          f"(min {float(ews.TE_st_to_sp.min()):.4f}, "
          f"max {float(ews.TE_st_to_sp.max()):.4f})")
    print(f"  var (mean): {float(ews.var_sp.mean()):.4f}")
    print(f"  AC1 (mean): {float(ews.ac1_sp.mean()):.3f}")
    h3 = test_h3(ews)
    print(f"  t_TE_cross    = {h3.t_te_cross}")
    print(f"  t_var_cross   = {h3.t_var_cross}")
    print(f"  t_alpha_cross = {h3.t_alpha_cross}")
    print(f"  lead (TE vs var)   = {h3.lead_years_te_vs_var:.1f} yr")
    print(f"  lead (TE vs alpha) = {h3.lead_years_te_vs_alpha:.1f} yr")
    print(f"  pass_h3 = {h3.pass_h3}    note: {h3.note}")

    # ---- Save numerical results ---------------------------------------------
    out = {
        "H1": {
            "primary_variable": "R_sst",
            "variants": {
                k: {"rho": v.rho, "p": v.p_value, "n": v.n_months,
                    "bootstrap_95ci": [v.bootstrap_low, v.bootstrap_high],
                    "pass_strong": v.pass_strong, "pass_suggestive": v.pass_suggestive,
                    "falsified": v.falsified}
                for k, v in h1_variants.items()
            },
            "overlap": [h1.overlap_start, h1.overlap_end],
        },
        "H2": {
            "fraction_years_passing": h2.fraction_years_passing,
            "n_years": h2.n_years,
            "pass": h2.pass_h2,
            "a_inc_per_year": h2.a_inc_per_year,
            "a_coh_per_year": h2.a_coh_per_year,
            "overlap": [h2.overlap_start, h2.overlap_end],
        },
        "H2_prime": {
            "R_CB_mean": h2p.R_CB_mean,
            "R_ST_mean": h2p.R_ST_mean,
            "C": h2p.C,
            "mean_phase_diff_deg": h2p.mean_phase_diff_deg,
            "n_months": h2p.n_months,
            "pass": h2p.pass_h2p,
        },
        "H3": {
            "t_TE_cross": h3.t_te_cross,
            "t_var_cross": h3.t_var_cross,
            "t_alpha_cross": h3.t_alpha_cross,
            "lead_TE_vs_var_yr": h3.lead_years_te_vs_var,
            "lead_TE_vs_alpha_yr": h3.lead_years_te_vs_alpha,
            "pass": h3.pass_h3,
            "note": h3.note,
        },
    }
    with open(PHASE3_RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {PHASE3_RESULTS}")

    # ----- Figure 1: H1 overlay --------------------------------------------------
    fig1_path = MANUSCRIPT_FIGS / "fig1_R_vs_Sv.pdf"
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    R_yymm = np.array([str(t)[:7] for t in phase2.R_sst.time.values])
    P_yymm = np.array([str(t)[:7] for t in rapid.time.values])
    common_yymm = np.intersect1d(R_yymm, P_yymm)
    R_idx = np.array([i for i, ym in enumerate(R_yymm) if ym in set(common_yymm)])
    P_idx = np.array([i for i, ym in enumerate(P_yymm) if ym in set(common_yymm)])
    R_o = phase2.R_sst.isel(time=R_idx)
    P_o = rapid.isel(time=P_idx)
    R_lp = standardise(lowpass_1yr(R_o.astype(np.float64)))
    P_lp = standardise(lowpass_1yr(P_o.astype(np.float64)))
    plot_t = R_o.time.values
    ax.plot(plot_t, R_lp, color="C0", lw=1.4, label=r"$R(t)$ (basin Kuramoto, std.)")
    ax.plot(plot_t, P_lp, color="C3", lw=1.4, label=r"$\Psi(t)$ RAPID 26.5$^\circ$N (std.)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(
        f"Figure 1.  H1: R(t) vs RAPID AMOC, "
        f"Pearson r = {h1.rho:+.2f} (95% CI {h1.bootstrap_low:+.2f}, {h1.bootstrap_high:+.2f}), "
        f"p = {h1.p_value:.1e}, n = {h1.n_months} months",
        fontsize=9,
    )
    ax.set_xlabel("year"); ax.set_ylabel("standardised")
    ax.legend(loc="best", fontsize=9)
    ax.text(0.02, 0.05, "H1 supported (strong)" if h1.pass_strong else
            ("H1 suggestive" if h1.pass_suggestive else "H1 falsified"),
            transform=ax.transAxes, fontsize=9, color="C2" if h1.pass_strong else "C1")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Wrote {fig1_path}")

    # ----- Figure 2: Chimera map (H2 + H2') -------------------------------------
    fig2_path = MANUSCRIPT_FIGS / "fig2_chimera_map.pdf"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    h1_window_slice = slice(h1.overlap_start, h1.overlap_end)
    a = ax[0]
    field500 = phase2.r_loc_sst.sel(time=h1_window_slice).mean("time")
    im = a.pcolormesh(phase2.lon, phase2.lat, field500, cmap="RdYlBu_r",
                      vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=a, label=r"$r_\mathrm{loc}$")
    a.set_title("(a) Local Kuramoto $r$, 500-km window\nmean over RAPID overlap 2004-2014")
    a.set_xlabel("lon"); a.set_ylabel("lat")
    a.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="k", lw=1.5))
    a.text(-37, 46, "Cold Blob", fontsize=8)
    a.add_patch(Rectangle((-60, 20), 50, 20, fill=False, edgecolor="k", lw=1.0, ls="--"))
    a.text(-55, 22, "Subtropics", fontsize=8)

    a = ax[1]
    # Inter-region phase-difference time series
    from dcps.h2 import regional_phase as rphase
    R_cb_t, phi_cb_t = rphase(phases.sst_phase, COLD_BLOB_BOX["lat"], COLD_BLOB_BOX["lon"])
    R_st_t, phi_st_t = rphase(phases.sst_phase, SUBTROPICAL_BOX["lat"], SUBTROPICAL_BOX["lon"])
    dphi = (phi_cb_t.values - phi_st_t.values + np.pi) % (2 * np.pi) - np.pi
    a.plot(phi_cb_t.time, dphi, lw=0.6, color="C2")
    a.axhline(0, color="grey", lw=0.4)
    a.set_ylim(-np.pi, np.pi)
    a.set_title(
        f"(b) Inter-region phase difference (Cold Blob - Subtropics)\n"
        f"H2 pass: {h2.pass_h2}  |  H2' pass: {h2p.pass_h2p}  "
        f"(C = {h2p.C:.3f}, R_CB = {h2p.R_CB_mean:.2f}, R_ST = {h2p.R_ST_mean:.2f})",
        fontsize=9,
    )
    a.set_xlabel("year"); a.set_ylabel(r"$\Delta\phi$ (rad)")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Wrote {fig2_path}")

    # ----- Figure 3: H3 TE vs CSD -----------------------------------------------
    fig3_path = MANUSCRIPT_FIGS / "fig3_TE_vs_CSD.pdf"
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True, constrained_layout=True)
    a = axes[0]
    a.plot(ews.time, ews.TE_st_to_sp, color="C0", lw=1.0)
    if h3.t_te_cross:
        a.axvline(np.datetime64(h3.t_te_cross), color="C0", ls="--", alpha=0.7,
                  label=f"$t_{{TE}}$ = {h3.t_te_cross[:7]}")
    a.set_ylabel(r"$T_{ST\to SP}$ (bits)")
    a.set_title("(a) Subtropical $\\to$ subpolar transfer entropy")
    if h3.t_te_cross:
        a.legend(loc="best", fontsize=8)

    a = axes[1]
    a.plot(ews.time, ews.var_sp, color="C1", lw=1.0)
    if h3.t_var_cross:
        a.axvline(np.datetime64(h3.t_var_cross), color="C1", ls="--", alpha=0.7,
                  label=f"$t_{{var}}$ = {h3.t_var_cross[:7]}")
        a.legend(loc="best", fontsize=8)
    a.set_ylabel(r"$\sigma^2_{SP}$ (K$^2$)")
    a.set_title("(b) Subpolar SST variance")

    a = axes[2]
    a.plot(ews.time, ews.ac1_sp, color="C3", lw=1.0)
    if h3.t_alpha_cross:
        a.axvline(np.datetime64(h3.t_alpha_cross), color="C3", ls="--", alpha=0.7,
                  label=f"$t_{{\\alpha}}$ = {h3.t_alpha_cross[:7]}")
        a.legend(loc="best", fontsize=8)
    a.set_ylabel(r"$\alpha_1$")
    a.set_xlabel("year")
    a.set_title("(c) Subpolar lag-1 autocorrelation")

    fig.suptitle(
        "Figure 3.  H3 EWS comparison on 10-yr sliding window. "
        f"pass_h3 = {h3.pass_h3}",
        fontsize=10,
    )
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"Wrote {fig3_path}")

    phase1.close(); phase2.close(); phases.close()


if __name__ == "__main__":
    main()
