"""Pre-Argo (1958-1999) vs post-Argo (2000-2023) Quiescence stability.

Pre-registered (commit-anchored):
  Compute rho(<r_loc>, |grad SSH|) on ORAS5 1958-1999 and 2000-2023
  separately, in each of NA, NP, ACC.  Report whether the Quiescence
  pattern is stable across the data-quality regime change.

  Pre-registered pass condition (locked):
    |rho_pre_argo - rho_post_argo| < 0.10 in every basin
      => Quiescence pattern is stable to data-quality regime;
         the pre-Argo period can be added to the analysis window.
    |Delta rho| in [0.10, 0.20] in any basin
      => partial stability; justify the post-Argo focus quantitatively.
    |Delta rho| > 0.20 in any basin
      => substantial regime change; keep post-Argo window only.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
)


OUT_DIR = CACHE_DIR / "pre_argo"


def quiescence_correlation_window(basin, start, end):
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin, start=start, end=end)
    ssh, _, _ = load_oras5_basin("sossheig", basin, start=start, end=end)
    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_mean = local_r_mean(phi, radius_km=500.0)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                        + ssh_mean.differentiate("rlon") ** 2)
    a = rl_mean.values.ravel()
    b = grad_mag.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan"), float("nan"), 0
    rho, p = pearsonr(a[m], b[m])
    return float(rho), float(p), int(m.sum())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Pre-Argo (1958-1999) vs post-Argo (2000-2023) Quiescence stability")
    print("=" * 70)

    windows = {
        "pre_argo": ("1958-01-01", "1999-12-31"),
        "post_argo": ("2000-01-01", "2023-12-31"),
    }
    results = {}
    for basin in BASINS:
        results[basin] = {}
        for wkey, (s, e) in windows.items():
            t0 = time.time()
            try:
                rho, p, n = quiescence_correlation_window(basin, s, e)
                results[basin][wkey] = {"rho": rho, "p": p, "n_cells": n}
                print(f"  {basin:<10}  {wkey:<10}  rho = {rho:+.3f}  p = {p:.2e}  n = {n}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                results[basin][wkey] = {"rho": float("nan"), "error": str(e)}
                print(f"  {basin:<10}  {wkey:<10}  FAILED: {e}")
        if "pre_argo" in results[basin] and "post_argo" in results[basin]:
            rho_pre = results[basin]["pre_argo"].get("rho", float("nan"))
            rho_post = results[basin]["post_argo"].get("rho", float("nan"))
            d = abs(rho_pre - rho_post)
            results[basin]["delta_rho"] = float(d)
            if d < 0.10:
                results[basin]["verdict"] = "stable (Delta < 0.10)"
            elif d < 0.20:
                results[basin]["verdict"] = "partial stability"
            else:
                results[basin]["verdict"] = "regime change > 0.20"
            print(f"    {basin:<10}  |Delta rho| = {d:.3f}  -- {results[basin]['verdict']}")

    with open(OUT_DIR / "pre_argo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'pre_argo_results.json'}")


if __name__ == "__main__":
    main()
