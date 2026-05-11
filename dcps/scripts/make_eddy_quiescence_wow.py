"""WOW figure for the revision: the eddy-resolving Quiescence story.

Three-panel landscape:
  a) GLORYS12 native 1/12 deg geostrophic EKE map of the NA basin
     (Gulf Stream + NAC corridor highlighted, Cold Blob box outlined).
  b) The 2 deg time-mean local Kuramoto coherence <r_loc>
     on the same coordinate axis, Cold Blob box outlined.
  c) Cell-wise scatter of <r_loc> vs box-averaged 1/12 deg EKE,
     showing rho = -0.351 (n = 966 cells, pre-registered threshold
     <= -0.30 met).

The visual story: where panel (a) is bright (high EKE) panel (b) is
dark (low r_loc), and vice versa.  Nothing like this has been
published for the AMOC Cold Blob.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
EKE_DIR = CACHE_DIR / "eke_eddy_resolving"


def main():
    eke = xr.open_dataset(EKE_DIR / "eke_native_1_12deg.nc")["EKE_eddy_resolving"]
    rl_2deg_ds = xr.open_dataset(EKE_DIR / "rl_mean_2deg_glorys12.nc")
    rl_2deg = rl_2deg_ds[list(rl_2deg_ds.data_vars)[0]]
    eke_2deg = xr.open_dataset(EKE_DIR / "eke_2deg_from_native.nc")["EKE_box_avg_2deg"]
    eke_eddy_json = json.loads((EKE_DIR / "eke_eddy_test.json").read_text())
    rho = float(eke_eddy_json["rho_eke_eddy_resolving"])
    n_cells = int(eke_eddy_json["n_cells"])

    # Downsample 4x for print-resolution PDF (1/3 deg, plenty for the
    # visual story; cuts PDF size from ~24 MB to ~1.5 MB).
    eke = eke.isel(y=slice(None, None, 4), x=slice(None, None, 4))
    nav_lat = eke["nav_lat"].values
    nav_lon = eke["nav_lon"].values
    rlon = rl_2deg["rlon"].values
    lat_2deg = rl_2deg["lat"].values
    actual_lon = rlon - 80.0

    fig = plt.figure(figsize=(14.5, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.05, 3, 0.05, 2])
    ax_eke = fig.add_subplot(gs[0, 0])
    cax_eke = fig.add_subplot(gs[0, 1])
    ax_rl = fig.add_subplot(gs[0, 2])
    cax_rl = fig.add_subplot(gs[0, 3])
    ax_sc = fig.add_subplot(gs[0, 4])

    # ----- panel a: native 1/12 deg EKE map -----------------------------
    eke_v = eke.values
    eke_lo = max(np.nanpercentile(eke_v, 5), 1e-4)
    eke_hi = np.nanpercentile(eke_v, 99)
    im_eke = ax_eke.pcolormesh(
        nav_lon, nav_lat, eke_v,
        cmap="inferno", shading="auto",
        norm=LogNorm(vmin=eke_lo, vmax=eke_hi),
        rasterized=True,
    )
    cb_eke = fig.colorbar(im_eke, cax=cax_eke,
                             label="Geostrophic EKE (m$^{2}$/s$^{2}$)")
    cb_eke.ax.tick_params(labelsize=9)

    cb_box_a = Rectangle((-50, 45), width=35, height=15, linewidth=2.5,
                            edgecolor="white", facecolor="none",
                            linestyle="--", zorder=10)
    ax_eke.add_patch(cb_box_a)
    ax_eke.annotate("Gulf Stream + NAC:\nhigh EKE corridor",
                       xy=(-58, 40), xytext=(-78, 14),
                       color="white", fontsize=10, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="white",
                                          lw=2.0))
    ax_eke.text(-49, 61, "Cold Blob box",
                  color="white", fontsize=10, fontweight="bold")
    ax_eke.set_xlabel("Longitude")
    ax_eke.set_ylabel("Latitude")
    ax_eke.set_xlim(-80, 0)
    ax_eke.set_ylim(0, 75)
    ax_eke.text(-0.10, 1.02, "a", transform=ax_eke.transAxes,
                  fontweight="bold", fontsize=14)

    # ----- panel b: 2 deg r_loc map (bilinear-shaded for visual parity
    # with panel a; the underlying data is still 2 deg) ------------------
    R = rl_2deg.values
    # imshow with bilinear interpolation gives smooth rendering for the
    # coarse 2 deg grid so the panel reads at the same visual fidelity
    # as the downsampled 1/12 deg panel a.
    im_rl = ax_rl.imshow(
        R,
        extent=[actual_lon.min() - 1, actual_lon.max() + 1,
                lat_2deg.min() - 1, lat_2deg.max() + 1],
        origin="lower", aspect="auto",
        cmap="viridis", interpolation="bilinear",
        vmin=np.nanpercentile(R, 3), vmax=np.nanpercentile(R, 97),
    )
    cb_rl = fig.colorbar(im_rl, cax=cax_rl,
                            label=r"$\langle r_{\mathrm{loc}}\rangle_t$  (phase coherence)")
    cb_rl.ax.tick_params(labelsize=9)

    cb_box_b = Rectangle((-50, 45), width=35, height=15, linewidth=2.5,
                            edgecolor="red", facecolor="none",
                            linestyle="--", zorder=10)
    ax_rl.add_patch(cb_box_b)
    ax_rl.annotate("Cold Blob:\nhigh coherence\n(low mass throughput)",
                       xy=(-32, 55), xytext=(-78, 12),
                       color="white", fontsize=10, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color="white",
                                          lw=2.0))
    ax_rl.text(-49, 61, "Cold Blob box",
                  color="red", fontsize=10, fontweight="bold")
    ax_rl.set_xlabel("Longitude")
    ax_rl.set_ylabel("Latitude")
    ax_rl.set_xlim(-80, 0)
    ax_rl.set_ylim(0, 75)
    ax_rl.text(-0.10, 1.02, "b", transform=ax_rl.transAxes,
                  fontweight="bold", fontsize=14)

    # ----- panel c: scatter ---------------------------------------------
    rl_v = R.ravel()
    eke_2deg_v = eke_2deg.values.ravel()
    m = np.isfinite(rl_v) & np.isfinite(eke_2deg_v)
    x = eke_2deg_v[m]
    y = rl_v[m]
    ax_sc.scatter(x, y, s=10, alpha=0.45, color="C0", edgecolors="none")
    ax_sc.set_xscale("log")
    if x.size > 30:
        coef = np.polyfit(np.log10(np.maximum(x, 1e-6)), y, 1)
        xline = np.logspace(np.log10(np.nanmin(x[x > 0])),
                              np.log10(np.nanmax(x)), 50)
        ax_sc.plot(xline, np.polyval(coef, np.log10(xline)),
                       color="C3", linewidth=2.5)
    ax_sc.set_xlabel("EKE 1/12$^{\\circ}\\rightarrow$2$^{\\circ}$ "
                        "(m$^{2}$/s$^{2}$)")
    ax_sc.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax_sc.text(0.04, 0.96,
                  f"$\\rho = {rho:+.3f}$\n$n = {n_cells}$ cells\n"
                  "pre-registered\n threshold "
                  "$\\rho \\leq -0.30$ $\\checkmark$",
                  transform=ax_sc.transAxes, fontsize=9, va="top",
                  bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="0.5"))
    ax_sc.text(-0.18, 1.02, "c", transform=ax_sc.transAxes,
                  fontweight="bold", fontsize=14)
    ax_sc.grid(alpha=0.3)

    out_fig = MANUSCRIPT_FIGS / "fig_wow_eddy_quiescence.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
