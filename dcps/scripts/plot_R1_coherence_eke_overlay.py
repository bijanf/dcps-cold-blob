"""Plot R1 (reviewer-driven): spatial map of <r_loc> with EKE contours.

Single-panel North Atlantic map: time-mean local Kuramoto coherence
<r_loc>_t shown as a filled pcolormesh; eddy kinetic energy (EKE)
climatology overlaid as black dashed contour lines at LINEAR levels
(50, 200, 500, 1000, 2000 cm^2 s^-2).  Dashed Cold Blob rectangle.

Linear units only -- no log scaling on the colour bar or the contour
levels.  Equatorial geostrophic mask |lat| < 15 deg applied.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

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

# Cold Blob rectangle, matching the convention in make_eddy_quiescence_wow.py
COLDBLOB_LON_W, COLDBLOB_LON_E = -50.0, -15.0
COLDBLOB_LAT_S, COLDBLOB_LAT_N = 45.0, 60.0

EQ_BAND_LAT = 15.0  # |lat| < 15 -> geostrophic mask
EKE_CONTOURS_CM2S2 = (50, 200, 500, 1000, 2000)


def _load_basin_fields():
    rl_ds = xr.open_dataset(EKE_DIR / "rl_mean_2deg_glorys12.nc")
    rl = rl_ds[list(rl_ds.data_vars)[0]]
    eke_ds = xr.open_dataset(EKE_DIR / "eke_2deg_from_native.nc")
    eke = eke_ds["EKE_box_avg_2deg"]

    rlon = rl["rlon"].values
    lat = rl["lat"].values
    lon = rlon - 80.0  # rotated -> actual

    eq_mask = np.broadcast_to(
        (np.abs(lat) < EQ_BAND_LAT)[:, None], (lat.size, lon.size),
    )
    R = rl.values.astype(float)
    E = eke.values.astype(float)
    R[eq_mask] = np.nan
    E[eq_mask] = np.nan
    return lat, lon, R, E


def main():
    lat, lon, R, E = _load_basin_fields()

    # Convert EKE to cm^2 s^-2 for the contour annotation (linear, only
    # the units change; equivalent multiplication by 1e4 of m^2 s^-2).
    E_cm2s2 = E * 1e4

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

    im = ax.pcolormesh(lon, lat, R, cmap="viridis", shading="auto",
                       vmin=np.nanpercentile(R, 2),
                       vmax=np.nanpercentile(R, 98),
                       rasterized=True, **trans_kw)
    cs = ax.contour(lon, lat, E_cm2s2, levels=list(EKE_CONTOURS_CM2S2),
                    colors="black", linestyles="--", linewidths=0.6,
                    **trans_kw)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%d")

    cb_rect = Rectangle(
        (COLDBLOB_LON_W, COLDBLOB_LAT_S),
        COLDBLOB_LON_E - COLDBLOB_LON_W,
        COLDBLOB_LAT_N - COLDBLOB_LAT_S,
        linewidth=1.0, edgecolor="red", facecolor="none",
        linestyle="--", zorder=5, **trans_kw,
    )
    ax.add_patch(cb_rect)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04, fraction=0.06)
    cb.set_label(r"$\langle r_{\mathrm{loc}}\rangle_t$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Compute Q from the same arrays so the caption value matches the
    # cell-wise correlation underlying this map.
    flat_r = R.ravel(); flat_e = E.ravel()
    m = np.isfinite(flat_r) & np.isfinite(flat_e)
    rho, _ = pearsonr(flat_r[m], flat_e[m])
    Q = -float(rho)
    print(f"Plot R1: cell-wise rho = {rho:.3f}, Q = {Q:.3f}, n_cells = {int(m.sum())}")

    out = MANUSCRIPT_FIGS / "fig_R1_coherence_eke_overlay.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
