"""Story figure for P6: <r_loc> map + EKE map + scatter w/ fit + residual map.

4-panel Nature-style PDF for the SI:
  (a) <r_loc>(x) on Atlantic 2-deg from primary 30-yr DLESyM rollout
  (b) GLORYS12 EKE climatology on the same grid
  (c) Cell-wise scatter of <r_loc> vs EKE with registered tau=144 and the
      best-AIC alternative parametric form overlaid
  (d) Residual map of <r_loc> - (1+tau_fit*EKE)^-0.5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/fallah/NEW_Theory/dcps/scripts")
from multi_basin_quiescence import (  # noqa: E402
    BASINS, basin_target_grid,
    instantaneous_phase, local_r_mean, preprocess_anomaly,
)
from quiescence_dlesym import healpix_to_basin_grid, TAU_OBS_NA  # noqa: E402

SST_PATH = Path("/p/projects/poem/fallah/dlesym_p6_out/p6_30yr_sst.nc")
EKE_PATH = Path("/p/projects/poem/fallah/cache/glorys12_eke_clim_atlantic_2deg.nc")
FLEX_JSON = Path("/p/projects/poem/fallah/p6_bundle/data/flex_fit.json")
OUT_PDF = Path("/p/projects/poem/fallah/p6_bundle/figures/p6_story.pdf")
RLOC_NC = Path("/p/projects/poem/fallah/p6_bundle/data/r_loc_atlantic_2deg.nc")


def _eval_form(name: str, params, eke):
    if name == "registered":
        return (1.0 + params[0] * eke) ** -0.5
    if name == "power":
        a, t, b = params
        return (a + t * eke) ** -b
    if name == "exp":
        r0, t = params
        return r0 * np.exp(-t * eke)
    if name == "saturating":
        r_inf, r0, t = params
        return r_inf + (r0 - r_inf) * np.exp(-t * eke)
    raise ValueError(name)


def main() -> int:
    print(f"loading {SST_PATH}")
    sst_basin = healpix_to_basin_grid(SST_PATH, basin="atlantic")
    bp = preprocess_anomaly(sst_basin)
    phase = instantaneous_phase(bp)
    rloc = local_r_mean(phase, radius_km=500.0)

    eke = xr.open_dataarray(EKE_PATH)

    rloc.to_netcdf(RLOC_NC.parent.mkdir(parents=True, exist_ok=True) or RLOC_NC)
    print(f"wrote {RLOC_NC}")

    lat_c, rlon_c, _, _ = basin_target_grid("atlantic")
    lon_display = np.asarray(rlon_c) + BASINS["atlantic"]["lon_offset"]

    rv = rloc.values
    ev = eke.values
    mask = np.isfinite(rv) & np.isfinite(ev)

    # Load flex fit if available
    flex = json.loads(FLEX_JSON.read_text()) if FLEX_JSON.exists() else None
    reg_params = None
    alt_params = None
    alt_name = None
    if flex:
        for f in flex.get("fits", []):
            if f.get("family") == "registered" and "params" in f:
                reg_params = f["params"]
            if alt_params is None and f.get("family") not in (None, "registered") \
                    and "params" in f and f.get("family") == flex.get("best_family"):
                alt_params = f["params"]
                alt_name = f["family"]
        # Fallback: pick best non-registered AIC if not yet
        if alt_params is None:
            cands = [f for f in flex.get("fits", [])
                     if f.get("family") not in (None, "registered") and "params" in f]
            cands.sort(key=lambda f: f.get("AIC", 1e18))
            if cands:
                alt_params = cands[0]["params"]
                alt_name = cands[0]["family"]

    # ------ Figure layout ----------------------------------------------------
    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) <r_loc> map
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.pcolormesh(lon_display, lat_c, rv, cmap="viridis", shading="auto")
    plt.colorbar(im_a, ax=ax_a, label=r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax_a.set_title("(a) DLESyM 30-yr $\\langle r_{loc}\\rangle$")
    ax_a.set_xlabel("Longitude")
    ax_a.set_ylabel("Latitude")

    # (b) EKE map
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.pcolormesh(
        lon_display, lat_c, ev, cmap="magma", shading="auto",
        vmax=np.nanpercentile(ev[mask], 97) if mask.any() else None,
    )
    plt.colorbar(im_b, ax=ax_b, label=r"EKE [m$^2$/s$^2$]")
    ax_b.set_title("(b) GLORYS12 EKE clim")
    ax_b.set_xlabel("Longitude")

    # (c) scatter + curves
    ax_c = fig.add_subplot(gs[1, 0])
    x = rv[mask]
    y = ev[mask]
    ax_c.scatter(y, x, s=6, alpha=0.4, color="C0", label=f"cells (n={mask.sum()})")
    e_grid = np.linspace(0, np.nanmax(y), 200)
    # Registered tau=144
    ax_c.plot(e_grid, _eval_form("registered", [TAU_OBS_NA], e_grid),
              "k--", lw=1.2, label=f"reg tau={TAU_OBS_NA:.0f}")
    if reg_params:
        ax_c.plot(e_grid, _eval_form("registered", reg_params, e_grid),
                  "C3-", lw=1.2,
                  label=f"reg-fit tau={reg_params[0]:.1f}")
    if alt_params is not None and alt_name:
        ax_c.plot(e_grid, _eval_form(alt_name, alt_params, e_grid),
                  "C2-", lw=1.2, label=f"{alt_name} fit")
    ax_c.set_xlabel("EKE [m$^2$/s$^2$]")
    ax_c.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax_c.set_title("(c) cell-wise law")
    ax_c.legend(fontsize=7, loc="best")

    # (d) residual map (uses registered-fit tau if available, else tau_obs)
    tau_use = reg_params[0] if reg_params else TAU_OBS_NA
    resid = rv - (1.0 + tau_use * ev) ** -0.5
    ax_d = fig.add_subplot(gs[1, 1])
    vmax = np.nanmax(np.abs(resid[mask])) if mask.any() else 0.1
    im_d = ax_d.pcolormesh(lon_display, lat_c, resid, cmap="RdBu_r",
                            shading="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im_d, ax=ax_d, label="r_loc - law")
    ax_d.set_title(f"(d) residual (tau={tau_use:.1f})")
    ax_d.set_xlabel("Longitude")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
