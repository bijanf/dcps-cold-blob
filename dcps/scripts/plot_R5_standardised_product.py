"""Plot R5 (reviewer-driven): map of the standardised product c(x).

c(x) is the cell-wise product of the standardised <r_loc> anomaly and
the standardised EKE anomaly:
    c(x) = ((<r_loc>(x) - rbar) / sigma_r) * ((EKE(x) - Ebar) / sigma_E)

By construction mean(c) = rho exactly (Pearson rho is the average of
the standardised products).  The map therefore visualises the spatial
distribution of each cell's contribution to the basin-wide
anti-correlation.  Blue (negative) cells SUPPORT the anti-correlation;
red (positive) cells OPPOSE it.

Linear units only.  A single dashed contour at the 30th-percentile EKE
demarcates the quiescent region.  Dashed Cold Blob rectangle.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, SINGLE_COL_IN
apply_nature_style()

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


EKE_DIR = CACHE_DIR / "eke_eddy_resolving"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

EQ_BAND_LAT = 15.0
COLDBLOB_LON_W, COLDBLOB_LON_E = -50.0, -15.0
COLDBLOB_LAT_S, COLDBLOB_LAT_N = 45.0, 60.0


def main():
    rl_ds = xr.open_dataset(EKE_DIR / "rl_mean_2deg_glorys12.nc")
    rl = rl_ds[list(rl_ds.data_vars)[0]]
    eke = xr.open_dataset(EKE_DIR / "eke_2deg_from_native.nc")["EKE_box_avg_2deg"]
    lat = rl["lat"].values
    rlon = rl["rlon"].values
    lon = rlon - 80.0
    R = rl.values.astype(float)
    E = eke.values.astype(float)
    eq = (np.abs(lat) < EQ_BAND_LAT)[:, None]
    R[np.broadcast_to(eq, R.shape)] = np.nan
    E[np.broadcast_to(eq, E.shape)] = np.nan

    m = np.isfinite(R) & np.isfinite(E)
    rbar = float(np.nanmean(R[m])); sigma_r = float(np.nanstd(R[m]))
    Ebar = float(np.nanmean(E[m])); sigma_E = float(np.nanstd(E[m]))
    C = ((R - rbar) / sigma_r) * ((E - Ebar) / sigma_E)
    C[~m] = np.nan
    rho_check = float(np.nanmean(C))
    assert np.isfinite(rho_check), "non-finite rho check"
    Q = -rho_check
    print(f"Plot R5: mean(c) = rho = {rho_check:.6f}, Q = {Q:.6f}, "
          f"n = {int(m.sum())}")

    eke_p30 = float(np.nanpercentile(E[m], 30))
    print(f"  30th-pctile EKE = {eke_p30:.5f} m^2/s^2  "
          f"= {eke_p30*1e4:.1f} cm^2/s^2")

    fig = plt.figure(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.95))
    if HAVE_CARTOPY:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_extent([-80, 0, 15, 70], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2,
                       edgecolor="0.55", linewidth=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=3,
                       edgecolor="0.35")
        ax.set_xticks(np.arange(-80, 1, 20), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(20, 71, 15), crs=ccrs.PlateCarree())
        trans_kw = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor("0.85")
        ax.set_xlim(-80, 0)
        ax.set_ylim(15, 70)
        ax.set_xticks(np.arange(-80, 1, 20))
        ax.set_yticks(np.arange(20, 71, 15))
        trans_kw = {}

    vmax = float(np.nanpercentile(np.abs(C), 98))
    im = ax.pcolormesh(lon, lat, C, cmap="RdBu_r", shading="auto",
                        vmin=-vmax, vmax=+vmax, rasterized=True,
                        **trans_kw)

    # 30th-percentile EKE contour (linear EKE).
    cs = ax.contour(lon, lat, E, levels=[eke_p30], colors="black",
                     linestyles="--", linewidths=0.7, **trans_kw)
    ax.clabel(cs, fmt={eke_p30: r"EKE$_{p30}$"}, fontsize=5)

    cb_rect = Rectangle(
        (COLDBLOB_LON_W, COLDBLOB_LAT_S),
        COLDBLOB_LON_E - COLDBLOB_LON_W,
        COLDBLOB_LAT_N - COLDBLOB_LAT_S,
        linewidth=1.0, edgecolor="black", facecolor="none",
        linestyle="--", zorder=5, **trans_kw,
    )
    ax.add_patch(cb_rect)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04, fraction=0.06)
    cb.set_label(r"$c(\mathbf{x})$  (standardised product)")
    cb.ax.tick_params(labelsize=8)

    out = MANUSCRIPT_FIGS / "fig_R5_standardised_product.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
