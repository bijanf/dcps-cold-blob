"""DCPS 2.0 'Quiescence Hypothesis' (H1*) tests.

Pre-registered (see chat transcript and manuscript update):
  H1*-a: rho(R(t), -dPsi/dt) >= +0.4, p<0.05    (R rises as Psi falls)
  H1*-b: trend(R) > 0 AND trend(Psi) < 0, both bootstrap-significant at 0.05
  H1*-c: spatial Pearson(<r_loc>_t, <|grad SSH|>_t) <= -0.3, p<0.01

These thresholds are fixed before computation; no test result modifies them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.h1 import circular_block_bootstrap_pearson, lowpass_1yr, standardise
from dcps.io import load_rapid_amoc

MULTI = CACHE_DIR / "multi"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
RESULTS = MULTI / "h1_star_results.json"


def low_pass_quarterly(da: xr.DataArray) -> xr.DataArray:
    """3-month low-pass to suppress sub-seasonal noise without killing decadal."""
    nyq = 0.5 * 12.0
    fc = 1.0 / 0.5
    b, a = butter(4, fc / nyq, btype="low")
    out = filtfilt(b, a, da.values.astype(np.float64))
    return da.copy(data=out.astype(da.dtype))


def central_diff(da: xr.DataArray) -> xr.DataArray:
    """Centred first difference along time."""
    arr = da.values.astype(np.float64)
    der = np.empty_like(arr)
    der[1:-1] = 0.5 * (arr[2:] - arr[:-2])
    der[0] = arr[1] - arr[0]
    der[-1] = arr[-1] - arr[-2]
    return da.copy(data=der.astype(da.dtype))


def trend_with_bootstrap(da: xr.DataArray, n_boot: int = 5000, seed: int = 0,
                          block_len: int = 12) -> tuple[float, float, float, float]:
    """Linear trend (slope per unit time index) and its 95% CI + p-value."""
    y = da.values.astype(np.float64)
    n = y.size
    x = np.arange(n)
    slope, _intercept = np.polyfit(x, y, 1)
    s = float(slope)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        n_blocks = int(np.ceil(n / block_len))
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        idx = idx[:n]
        boots[k] = float(np.polyfit(x, y[idx], 1)[0])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(s), float(lo), float(hi), float(p_two)


def align_yymm(a: xr.DataArray, b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    A = np.array([str(t)[:7] for t in a.time.values])
    B = np.array([str(t)[:7] for t in b.time.values])
    common = np.intersect1d(A, B)
    ai = np.array([i for i, ym in enumerate(A) if ym in set(common)])
    bi = np.array([i for i, ym in enumerate(B) if ym in set(common)])
    return a.isel(time=ai), b.isel(time=bi)


def main():
    products = ["oras5", "glorys12", "ecco"]
    rapid = load_rapid_amoc()
    out: dict = {}

    print("=" * 70)
    print("H1*-a   rho(R(t), dPsi/dt)  expected SIGN: NEGATIVE")
    print("                            decision: support iff rho<=-0.4, p<0.05")
    print("                  (i.e. R rises -- AMOC drops <==> R(t) up)")
    print("=" * 70)
    for prod in products:
        cache = MULTI / f"phase2_R_{prod}.nc"
        if not cache.exists():
            print(f"  {prod}: NO CACHE -- skip"); continue
        ds = xr.open_dataset(cache)
        R_var_name = "R_pooled" if "R_pooled" in ds.data_vars else "R_sst"
        R = ds[R_var_name]

        R_aln, P_aln = align_yymm(R, rapid)
        if R_aln.size < 60:
            print(f"  {prod}: too short ({R_aln.size}) -- skip"); ds.close(); continue

        # Low-pass at 1 yr to focus on decadal modes; then derivative.
        R_lp = lowpass_1yr(R_aln.astype(np.float64))
        P_lp = lowpass_1yr(P_aln.astype(np.float64))
        dP = central_diff(P_lp)

        # Pearson of low-passed R against -dPsi/dt
        x = standardise(R_lp).values
        y = -standardise(dP).values
        rho, p = pearsonr(x, y)
        boots = circular_block_bootstrap_pearson(x, y, block_len=12, n_resample=5000)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        verdict = "SUPPORTED" if (rho >= 0.4 and p < 0.05) else "falsified"
        print(f"  {prod} ({R_var_name}, n={R_aln.size}): rho(R, -dPsi/dt) = "
              f"{rho:+.3f}  CI=({lo:+.3f},{hi:+.3f})  p={p:.2e}   -> {verdict}")
        out.setdefault("H1_star_a", {})[prod] = {
            "rho": float(rho), "p": float(p), "ci": [float(lo), float(hi)],
            "n": int(R_aln.size), "R_variant": R_var_name, "verdict": verdict,
        }
        ds.close()

    print()
    print("=" * 70)
    print("H1*-b   trend(R) > 0 AND trend(Psi) < 0, both bootstrap-significant")
    print("=" * 70)
    # Trend of RAPID over its full record
    trP, plo, phi_, pp = trend_with_bootstrap(rapid)
    print(f"  RAPID Psi trend (Sv/month): {trP:+.4e}  CI=({plo:+.4e},{phi_:+.4e})  p={pp:.2e}")
    psi_neg = (trP < 0) and (pp < 0.05)
    out["H1_star_b"] = {"Psi_trend_per_month": trP, "Psi_trend_p": pp, "Psi_negative": psi_neg,
                       "products": {}}
    for prod in products:
        cache = MULTI / f"phase2_R_{prod}.nc"
        if not cache.exists(): continue
        ds = xr.open_dataset(cache)
        R = ds["R_pooled"] if "R_pooled" in ds.data_vars else ds["R_sst"]
        R_aln, _ = align_yymm(R, rapid)
        trR, rlo, rhi, rp = trend_with_bootstrap(R_aln.astype(np.float64))
        verdict = ("SUPPORTED" if (trR > 0 and rp < 0.05 and psi_neg)
                   else "falsified")
        print(f"  {prod} R trend (1/month): {trR:+.4e}  CI=({rlo:+.4e},{rhi:+.4e})  "
              f"p={rp:.2e}   -> {verdict}")
        out["H1_star_b"]["products"][prod] = {"R_trend": trR, "R_p": rp,
                                              "R_pos": (trR > 0 and rp < 0.05),
                                              "verdict": verdict}
        ds.close()

    print()
    print("=" * 70)
    print("H1*-c  Pearson(<r_loc>_t, <|grad SSH|>_t) over NA cells   <= -0.3, p<0.01")
    print("=" * 70)
    for prod in products:
        p1_cache = MULTI / f"phase1_{prod}.nc"
        p2_cache = MULTI / f"phase2_R_{prod}.nc"
        if not (p1_cache.exists() and p2_cache.exists()):
            print(f"  {prod}: cache missing -- skip"); continue
        ds1 = xr.open_dataset(p1_cache)
        ds2 = xr.open_dataset(p2_cache)
        if "ssh_anom" not in ds1.data_vars:
            print(f"  {prod}: no SSH variable in cache -- skip"); ds1.close(); ds2.close(); continue

        # Need also the local r computed for this product. Use phase2 r if cached;
        # else recompute from phases.
        # Our run_multi_product wrote only R_*; local r wasn't cached. Recompute here.
        from dcps.phase import analytic_signal, edge_trim
        from dcps.order_parameter import local_r
        _, _, phi = analytic_signal(ds1.sst_anom)
        phi = edge_trim(phi)
        rl = local_r(phi, radius_km=500.0)
        rl_mean = rl.mean("time")

        # |grad SSH| from the raw SSH field (climatology + low-pass to get mean circulation)
        ssh = ds1.ssh_raw.mean("time")     # time-mean SSH
        # Compute centred-difference gradients on the regular 2-deg grid
        dssh_dy = ssh.differentiate("lat")
        dssh_dx = ssh.differentiate("lon")
        grad_mag = np.sqrt(dssh_dy ** 2 + dssh_dx ** 2)

        # Flatten and pair only valid cells
        a = rl_mean.values.ravel()
        b = grad_mag.values.ravel()
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 50:
            print(f"  {prod}: too few cells -- skip"); ds1.close(); ds2.close(); continue
        rho, p = pearsonr(a, b)
        verdict = "SUPPORTED" if (rho <= -0.3 and p < 0.01) else "falsified"
        print(f"  {prod}: spatial rho(<r_loc>, |grad SSH|) = "
              f"{rho:+.3f}  p={p:.2e}  n_cells={a.size}   -> {verdict}")
        out.setdefault("H1_star_c", {})[prod] = {"rho": float(rho), "p": float(p),
                                                  "n_cells": int(a.size),
                                                  "verdict": verdict}
        ds1.close(); ds2.close()

    # ----- Persist + print verdict -------------------------------------------
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {RESULTS}")

    # Aggregate
    print()
    print("=" * 70)
    print("OVERALL H1* VERDICT")
    print("=" * 70)
    a_passes = sum(1 for v in out.get("H1_star_a", {}).values() if v["verdict"] == "SUPPORTED")
    b_passes = sum(1 for v in out.get("H1_star_b", {}).get("products", {}).values()
                   if v["verdict"] == "SUPPORTED")
    c_passes = sum(1 for v in out.get("H1_star_c", {}).values() if v["verdict"] == "SUPPORTED")
    n_a = len(out.get("H1_star_a", {}))
    n_b = len(out.get("H1_star_b", {}).get("products", {}))
    n_c = len(out.get("H1_star_c", {}))
    print(f"  H1*-a (dynamic):  {a_passes}/{n_a} products pass")
    print(f"  H1*-b (trend):    {b_passes}/{n_b} products pass")
    print(f"  H1*-c (spatial):  {c_passes}/{n_c} products pass")


if __name__ == "__main__":
    main()
