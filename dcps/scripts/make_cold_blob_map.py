"""Two-panel North Atlantic figure for the merged paper:
  (a) HadISST 1870-2023 linear SST trend per cell -- the iconic
      Cold Blob trend map.
  (b) Same NA domain with the five testable Caesar 2021 proxy
      sites overlaid, shape = depth class (circle = surface or
      thermocline <=400 m; square = deep >=2000 m), color =
      pre-registered verdict (red = unprecedented |z| > 3,
      gray = within envelope).

Approximate site coordinates are from the Caesar 2021
compilation and the underlying source publications (literature
values; for visualization only -- the cell-wise paleo statistic
is independent of plot position).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

REPO = Path("/home/bijanf/Documents/NEW_Theory")
HAD = REPO / "data/external/hadisst/HadISST_sst.nc"
OUT = REPO / "manuscript/figs/fig_cold_blob_map.pdf"

# NA basin window (matches the paper's domain)
LAT_S, LAT_N = 0.0, 75.0
LON_W, LON_E = -80.0, 0.0

# Proxy sites: (label, lat, lon, depth_class, verdict, z_score)
# Coordinates are approximate, drawn from Caesar 2021 source publications.
# Thornalley 2018 T_sub and sortable silt are from the SAME parent core
# RAPID-12-1K but at different depth horizons; we offset the markers
# visually so both are distinguishable.  Osmann 2019 MAS is from western
# North Atlantic abyssal sediment cores (Bermuda Rise region).
PROXIES = [
    ("Thornalley T_sub",      58.5, -27.0, "surface", "unprec",  19.7),
    ("Spooner T. quinqueloba",58.5, -22.5, "surface", "unprec",   4.8),
    ("Rahmstorf AMOC index",  53.5, -33.0, "surface", "unprec",  22.1),
    ("Thornalley silt",       57.0, -29.5, "deep",    "envelope", 0.8),
    ("Osmann MAS",            33.5, -62.0, "deep",    "envelope", 0.0),
]


def trend_per_cell(da: xr.DataArray) -> xr.DataArray:
    """Linear trend in deg C / century from a (time, lat, lon) SST DataArray.
    Masks values where >25% of months are missing.
    """
    # Sentinel HadISST: -1000 for missing
    sst = da.where(da > -100)
    years = (sst.time.dt.year + (sst.time.dt.month - 0.5) / 12.0).astype(float)
    x = years.values
    y = sst.values
    n_t, n_y, n_x = y.shape
    slope = np.full((n_y, n_x), np.nan)
    flat = y.reshape(n_t, -1)
    valid = np.isfinite(flat).sum(axis=0) > 0.75 * n_t
    mean_x = np.nanmean(x)
    dx = x - mean_x
    denom = np.nansum(dx ** 2)
    for j in np.where(valid)[0]:
        yj = flat[:, j]
        m = np.isfinite(yj)
        if m.sum() < 0.75 * n_t:
            continue
        b = np.nansum(dx[m] * (yj[m] - np.nanmean(yj[m]))) / np.nansum(dx[m] ** 2)
        slope.flat[j] = b * 100.0  # per century
    return xr.DataArray(slope, dims=("latitude", "longitude"),
                        coords={"latitude": da.latitude,
                                "longitude": da.longitude})


def main():
    ds = xr.open_dataset(HAD)
    sst = ds["sst"]
    sst = sst.sel(time=slice("1870-01-01", "2023-12-31"))
    # NA basin subset
    sst_na = sst.sel(latitude=slice(LAT_N, LAT_S),
                     longitude=slice(LON_W, LON_E))
    print(f"  NA subset shape: {sst_na.shape}")
    print("  Computing per-cell trends ...")
    trend = trend_per_cell(sst_na)
    print(f"  Trend range: {np.nanmin(trend.values):+.2f} to "
          f"{np.nanmax(trend.values):+.2f} K/century")

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                 subplot_kw={"projection": proj},
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                 constrained_layout=True)

    # === Panel (a): HadISST trend map ===
    ax = axes[0]
    vmax = 2.5  # K/century, symmetric
    if HAS_CARTOPY:
        im = ax.pcolormesh(trend.longitude, trend.latitude, trend.values,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto", transform=proj)
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.5",
                          linestyle=":")
        gl.top_labels = False; gl.right_labels = False
    else:
        im = ax.pcolormesh(trend.longitude, trend.latitude, trend.values,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.set_xlim(LON_W, LON_E); ax.set_ylim(LAT_S, LAT_N)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("(a)  HadISST 1870--2023 SST trend",
                 fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.08, shrink=0.85, aspect=30)
    cb.set_label(r"SST trend  ($^\circ$C per century)")
    cb.ax.tick_params(labelsize=8)

    # === Panel (b): Proxy sites on the same domain ===
    ax = axes[1]
    if HAS_CARTOPY:
        im2 = ax.pcolormesh(trend.longitude, trend.latitude, trend.values,
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            shading="auto", alpha=0.4, transform=proj)
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.5",
                          linestyle=":")
        gl.top_labels = False; gl.right_labels = False
    else:
        ax.pcolormesh(trend.longitude, trend.latitude, trend.values,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      shading="auto", alpha=0.4)
        ax.set_xlim(LON_W, LON_E); ax.set_ylim(LAT_S, LAT_N)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # Plot proxies: marker by depth, color by verdict.
    for (lab, lat, lon, depth, verdict, z) in PROXIES:
        marker = "o" if depth == "surface" else "s"
        face = "#d62728" if verdict == "unprec" else "0.55"
        edge = "black"
        kw = dict(marker=marker, s=180, c=face, edgecolors=edge,
                  linewidths=1.5, zorder=5)
        if HAS_CARTOPY:
            ax.scatter(lon, lat, transform=proj, **kw)
        else:
            ax.scatter(lon, lat, **kw)
    ax.set_title("(b)  Five testable Caesar 2021 proxy sites",
                 fontsize=11, loc="left")

    # Legend
    surface_h = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor="#d62728", markeredgecolor="k",
                            markersize=10, label=r"Surface or thermocline, $|z|>3$")
    deep_h = plt.Line2D([0], [0], marker="s", color="w",
                         markerfacecolor="0.55", markeredgecolor="k",
                         markersize=10, label=r"Deep, within envelope")
    leg = ax.legend(handles=[surface_h, deep_h], loc="lower right",
                    fontsize=8, framealpha=0.92)

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
