"""Story figure for P6: <r_loc> map + EKE map + scatter w/ fit + residual.

4-panel SI figure for the P6 cross-prediction (registered branch
falsified by chi^2 ceiling):

  (a) DLESyM 30-yr <r_loc>(x) on Atlantic 2-degree basin grid,
      masked to the analysis domain (cells where both <r_loc> and
      the eddy-resolving EKE climatology are finite).
  (b) GLORYS12 climatological EKE on the same grid.  The white band
      south of ~15 N is the equatorial geostrophic mask
      |lat| < 15 deg applied in
      ``dcps/scripts/eke_quiescence_eddy_resolving.py`` to avoid the
      1/f singularity in (g/f) * grad(eta).
  (c) Cell-wise scatter of <r_loc> vs EKE with the registered
      tau = 144 curve and the best-AIC alternative from the flex-fit
      ensemble overlaid.
  (d) Residual map of <r_loc> minus the best-fit registered law.

The shared analysis mask is applied to all four panels so the
figure shows exactly the n cells used for Q_X and the flex-fit.

Inputs (override on CLI):
  --rloc PATH    NetCDF with <r_loc> on the basin grid
  --eke  PATH    NetCDF with EKE climatology on the basin grid
  --flex PATH    flex_fit.json from ``p6_flex_fit.py``
  --out  PATH    output PDF
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dcps" / "scripts"))
from multi_basin_quiescence import BASINS, basin_target_grid  # noqa: E402

try:
    from dcps.nature_style import apply_nature_style
    apply_nature_style()
except Exception:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

TAU_OBS_NA = 144.0


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


def _open_da(path: Path) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except ValueError:
        ds = xr.open_dataset(path)
        return ds[list(ds.data_vars)[0]]


# Maps start at lat = 14 deg N: the GLORYS12 geostrophic EKE field is
# masked for |lat| < 15 deg (1/f singularity, see
# dcps/scripts/eke_quiescence_eddy_resolving.py), so the equatorial band
# is white in panels (b) and (d).  Cropping at 14 deg removes the
# unhelpful blank strip while keeping a 1-deg margin under the data.
LAT_MIN_DISPLAY = 14.0
LAT_MAX_DISPLAY = 75.0


def _make_map_ax(fig, gs_cell, lon_extent):
    if HAVE_CARTOPY:
        ax = fig.add_subplot(gs_cell, projection=ccrs.PlateCarree())
        ax.set_extent([lon_extent[0], lon_extent[1],
                       LAT_MIN_DISPLAY, LAT_MAX_DISPLAY],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2,
                       edgecolor="0.55", linewidth=0.4)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3,
                       edgecolor="0.35")
        ax.set_xticks(np.arange(lon_extent[0], lon_extent[1] + 1, 20),
                      crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(15, 76, 15), crs=ccrs.PlateCarree())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(direction="in", length=2.5, pad=2)
        return ax, dict(transform=ccrs.PlateCarree())
    ax = fig.add_subplot(gs_cell)
    ax.set_xlim(lon_extent[0], lon_extent[1])
    ax.set_ylim(LAT_MIN_DISPLAY, LAT_MAX_DISPLAY)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    return ax, {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rloc", type=Path, required=True)
    p.add_argument("--eke",  type=Path, required=True)
    p.add_argument("--flex", type=Path, default=None)
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--basin", default="atlantic")
    args = p.parse_args(argv)

    rloc = _open_da(args.rloc)
    eke = _open_da(args.eke)
    rv = rloc.values
    ev = eke.values
    mask = np.isfinite(rv) & np.isfinite(ev)

    # Apply the shared analysis mask to <r_loc> so panel (a) shows
    # the same physical domain as (b), (c), (d).
    rv_masked = np.where(mask, rv, np.nan)
    ev_masked = np.where(mask, ev, np.nan)

    lat_c, rlon_c, _, _ = basin_target_grid(args.basin)
    lon_display = np.asarray(rlon_c) + BASINS[args.basin]["lon_offset"]
    lon_extent = BASINS[args.basin]["lon_display"]

    flex = json.loads(args.flex.read_text()) if (args.flex and args.flex.exists()) else None
    reg_params = alt_params = None
    alt_name = None
    if flex:
        for f in flex.get("fits", []):
            if f.get("family") == "registered":
                reg_params = f["params"]
            if (alt_name is None and f.get("family") not in (None, "registered")
                    and f.get("family") == flex.get("best_family")):
                alt_params = f["params"]
                alt_name = f["family"]

    fig = plt.figure(figsize=(7.09, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    def _panel_label(ax, label, dx=-0.10):
        # Panel letter OUTSIDE the axes (top-left), no bbox.
        ax.text(dx, 1.02, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")

    # (a) <r_loc>
    ax_a, kw_a = _make_map_ax(fig, gs[0, 0], lon_extent)
    im_a = ax_a.pcolormesh(lon_display, lat_c, rv_masked, cmap="viridis",
                           shading="auto",
                           vmin=np.nanpercentile(rv_masked, 2),
                           vmax=np.nanpercentile(rv_masked, 98), **kw_a)
    plt.colorbar(im_a, ax=ax_a, label=r"$\langle r_{\mathrm{loc}}\rangle_t$",
                 shrink=0.85, pad=0.02)
    _panel_label(ax_a, "a")

    # (b) EKE
    ax_b, kw_b = _make_map_ax(fig, gs[0, 1], lon_extent)
    im_b = ax_b.pcolormesh(lon_display, lat_c, ev_masked, cmap="magma",
                           shading="auto",
                           vmax=np.nanpercentile(ev_masked, 97), **kw_b)
    plt.colorbar(im_b, ax=ax_b, label=r"EKE [m$^{2}$ s$^{-2}$]",
                 shrink=0.85, pad=0.02)
    _panel_label(ax_b, "b")

    # (c) scatter + curves
    ax_c = fig.add_subplot(gs[1, 0])
    x = rv[mask]; y = ev[mask]
    ax_c.scatter(y, x, s=4, alpha=0.35, color="0.45",
                 label=f"cells (n={int(mask.sum())})", linewidths=0)
    e_grid = np.linspace(0, float(np.nanmax(y)), 200)
    ax_c.plot(e_grid, _eval_form("registered", [TAU_OBS_NA], e_grid),
              "k--", lw=1.1,
              label=rf"registered $\tau\!=\!{TAU_OBS_NA:.0f}$")
    if reg_params:
        ax_c.plot(e_grid, _eval_form("registered", reg_params, e_grid),
                  color="C3", lw=1.1,
                  label=rf"reg-fit $\hat\tau\!=\!{reg_params[0]:.1f}$")
    if alt_params is not None and alt_name:
        ax_c.plot(e_grid, _eval_form(alt_name, alt_params, e_grid),
                  color="C2", lw=1.1,
                  label=f"{alt_name} fit")
    ax_c.set_xlabel(r"EKE [m$^{2}$ s$^{-2}$]")
    ax_c.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
    # Legend pushed below the axes so it never overlaps the scatter.
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, frameon=False, fontsize=7, handlelength=2.0,
                columnspacing=1.4)
    ax_c.tick_params(direction="in", length=2.5)
    _panel_label(ax_c, "c")

    # (d) residual (registered-fit tau if available, else tau_obs)
    tau_use = reg_params[0] if reg_params else TAU_OBS_NA
    resid = np.where(mask, rv - (1.0 + tau_use * np.where(mask, ev, 0.0)) ** -0.5,
                     np.nan)
    vmax = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 0.1
    ax_d, kw_d = _make_map_ax(fig, gs[1, 1], lon_extent)
    im_d = ax_d.pcolormesh(lon_display, lat_c, resid, cmap="RdBu_r",
                           shading="auto", vmin=-vmax, vmax=vmax, **kw_d)
    plt.colorbar(im_d, ax=ax_d,
                 label=r"$\langle r_{\mathrm{loc}}\rangle - $ fitted law",
                 shrink=0.85, pad=0.02)
    _panel_label(ax_d, "d")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
