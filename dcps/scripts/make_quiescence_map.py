"""Quiescence pattern map for the merged paper, using the canonical
cached fields from the published pipeline:
  (a) Time-mean local Kuramoto coherence <r_loc(x)>  (2 deg, GLORYS12)
  (b) Geostrophic EKE (2 deg, box-averaged from 1/12 deg native)
  (c) Cell-wise scatter; reports Pearson rho via JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

REPO = Path("/home/bijanf/Documents/NEW_Theory")
EKE = REPO / "dcps/cache/eke_eddy_resolving"
OUT = REPO / "manuscript/figs/fig_quiescence_map.pdf"


def main():
    rl_ds = xr.open_dataset(EKE / "rl_mean_2deg_glorys12.nc")
    rl = rl_ds[list(rl_ds.data_vars)[0]]
    eke_2deg = xr.open_dataset(EKE / "eke_2deg_from_native.nc")["EKE_box_avg_2deg"]
    info = json.loads((EKE / "eke_eddy_test.json").read_text())
    rho = float(info["rho_eke_eddy_resolving"])
    n_cells = int(info["n_cells"])
    Q = -rho
    print(f"  canonical rho = {rho:+.3f}  (n = {n_cells})  Q = {Q:+.3f}")

    lat = rl["lat"].values
    rlon = rl["rlon"].values
    lon = rlon - 80.0  # rlon range 1..79 -> -79..-1

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(13.5, 4.5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        ax0 = fig.add_subplot(gs[0], projection=proj)
        ax1 = fig.add_subplot(gs[1], projection=proj)
        ax2 = fig.add_subplot(gs[2])
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 4.5),
                                             constrained_layout=True)

    LON_W, LON_E, LAT_S, LAT_N = -80, 0, 0, 75

    # === Panel (a): r_loc ===
    R = rl.values
    vmin = np.nanpercentile(R, 3); vmax = np.nanpercentile(R, 97)
    if HAS_CARTOPY:
        im0 = ax0.pcolormesh(lon, lat, R, cmap="viridis",
                              vmin=vmin, vmax=vmax, shading="auto",
                              transform=proj)
        ax0.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax0.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax0.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
        gl = ax0.gridlines(draw_labels=True, linewidth=0.2, color="0.5",
                            linestyle=":")
        gl.top_labels = False; gl.right_labels = False
    else:
        im0 = ax0.pcolormesh(lon, lat, R, cmap="viridis",
                              vmin=vmin, vmax=vmax, shading="auto")
        ax0.set_xlim(LON_W, LON_E); ax0.set_ylim(LAT_S, LAT_N)
        ax0.set_xlabel("Longitude"); ax0.set_ylabel("Latitude")
    ax0.set_title(r"(a)  $\langle r_{\mathrm{loc}}\rangle_{t}$  "
                  r"local phase coherence", fontsize=10, loc="left")
    cb0 = fig.colorbar(im0, ax=ax0, orientation="horizontal",
                       pad=0.08, shrink=0.85)
    cb0.set_label("phase coherence")
    cb0.ax.tick_params(labelsize=8)
    cb_box_b = Rectangle((-50, 45), width=35, height=15, linewidth=2.0,
                          edgecolor="red", facecolor="none", linestyle="--",
                          zorder=10,
                          transform=proj if HAS_CARTOPY else None)
    if HAS_CARTOPY:
        ax0.add_patch(cb_box_b)
    else:
        ax0.add_patch(cb_box_b)

    # === Panel (b): EKE ===
    E = eke_2deg.values
    e_lo = max(np.nanpercentile(E, 5), 1e-4)
    e_hi = np.nanpercentile(E, 99)
    if HAS_CARTOPY:
        im1 = ax1.pcolormesh(lon, lat, E, cmap="inferno",
                              shading="auto", transform=proj,
                              norm=LogNorm(vmin=e_lo, vmax=e_hi))
        ax1.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax1.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
        gl = ax1.gridlines(draw_labels=True, linewidth=0.2, color="0.5",
                            linestyle=":")
        gl.top_labels = False; gl.right_labels = False
    else:
        im1 = ax1.pcolormesh(lon, lat, E, cmap="inferno",
                              shading="auto",
                              norm=LogNorm(vmin=e_lo, vmax=e_hi))
        ax1.set_xlim(LON_W, LON_E); ax1.set_ylim(LAT_S, LAT_N)
        ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.set_title(r"(b)  Geostrophic EKE  (1/12$^\circ$ native,"
                  r" 2$^\circ$ box average)", fontsize=10, loc="left")
    cb1 = fig.colorbar(im1, ax=ax1, orientation="horizontal",
                       pad=0.08, shrink=0.85)
    cb1.set_label(r"EKE  (m$^{2}$/s$^{2}$)")
    cb1.ax.tick_params(labelsize=8)

    # === Panel (c): scatter ===
    R_flat = R.flatten()
    E_flat = E.flatten()
    m = np.isfinite(R_flat) & np.isfinite(E_flat)
    ax2.scatter(E_flat[m], R_flat[m], s=12, c="#1f77b4",
                 alpha=0.6, edgecolors="none")
    ax2.set_xlabel(r"Geostrophic EKE  (m$^{2}$/s$^{2}$)")
    ax2.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_{t}$")
    ax2.set_xscale("log")
    ax2.set_title(r"(c)  Quiescence Signature scatter",
                   fontsize=10, loc="left")
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
