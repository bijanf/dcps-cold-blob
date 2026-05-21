"""Multi-model EKE difference maps with sign-agreement stippling.

For every per-model NetCDF in dcps/cache/eke_epoch_diff/ (produced by
compute_eke_epoch_diff.py), compute mid-baseline and far-baseline EKE
differences, then render the multi-model mean of each difference.

Significance overlay (IPCC AR6 style): stipple grid cells where
>= 80 % of contributing models agree on the sign of the change.

Output: manuscript/figs/fig_eke_epoch_diff_<basin>.pdf

Three panels:
  (a) far minus baseline  (2070-2099 minus 1850-1900)
  (b) mid minus baseline  (2030-2060 minus 1850-1900)
  (c) far minus mid       (incremental warming from 2030-2060 to 2070-2099)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
import sys
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
apply_nature_style()

DIFF_DIR = CACHE_DIR / "eke_epoch_diff"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

SIGN_AGREE_FRAC = 0.8
COLD_BLOB = dict(lon=(-50, -15), lat=(45, 60))


def _stack(basin: str) -> tuple[xr.Dataset, list[str]]:
    files = sorted(DIFF_DIR.glob(f"*_{basin}_epoch_diff.nc"))
    if not files:
        raise FileNotFoundError(f"no diff NetCDFs in {DIFF_DIR}")
    dss, models = [], []
    base = None
    for f in files:
        try:
            ds = xr.open_dataset(f)
            if base is None: base = ds
            else: ds = ds.reindex_like(base, method=None)
            dss.append(ds)
            models.append(ds.attrs.get("model", f.stem.split("_")[0]))
        except Exception as e:
            print(f"  skip {f.name}: {type(e).__name__}")
    stack = xr.concat(dss, dim="model").assign_coords(model=models)
    return stack, models


def _stipple(ax, lon, lat, mask):
    """Overlay sparse dots where mask is True."""
    skip = 2
    Lo, La = np.meshgrid(lon, lat)
    sel = mask & (np.arange(mask.shape[0])[:, None] % skip == 0) \
              & (np.arange(mask.shape[1])[None, :] % skip == 0)
    ax.scatter(Lo[sel], La[sel], s=0.6, color="0.15",
                marker=".", linewidths=0, zorder=4)


def _draw_cold_blob(ax, lon_offset):
    """Draw the Cold Blob rectangle in the axis's coordinate system.

    The plotted axis uses absolute longitude after we shift rlon by
    lon_offset, so we can pass the literal Caesar 2018 box (-50, 45)
    to (-15, 60) directly.
    """
    from matplotlib.patches import Rectangle
    lon0, lon1 = COLD_BLOB["lon"]
    lat0, lat1 = COLD_BLOB["lat"]
    ax.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                             facecolor="none", edgecolor="0.15",
                             linestyle="--", linewidth=0.7, zorder=3))


def _panel(ax, mean_diff, sign_frac, vmax, letter, title, lon, lat,
            lon_offset, show_ylabel=True):
    pm = ax.pcolormesh(lon, lat, mean_diff.values,
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         shading="auto")
    ax.set_facecolor("0.85")
    mask = sign_frac >= SIGN_AGREE_FRAC
    _stipple(ax, lon, lat, mask.values)
    _draw_cold_blob(ax, lon_offset)
    ax.set_xlabel("longitude (deg E)", fontsize=6, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("latitude (deg N)", fontsize=6, labelpad=2)
    else:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5)
    ax.set_title(title, fontsize=6, pad=2)
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
              ha="left", va="top", fontsize=8, fontweight="bold",
              color="white",
              bbox=dict(facecolor="0.15", edgecolor="none",
                        boxstyle="round,pad=0.15"))
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    stack, models = _stack(args.basin)
    N = len(models)
    print(f"basin={args.basin}  N_models_in_cache={N}")
    print(f"  models: {', '.join(models)}")

    per_diff = dict(
        far_minus_base=stack["far"] - stack["baseline"],
        mid_minus_base=stack["mid"] - stack["baseline"],
        far_minus_mid=stack["far"]  - stack["mid"],
    )

    def _stats(per_model_diff):
        mean = per_model_diff.mean("model", skipna=True)
        sign_pos = (per_model_diff > 0).sum("model")
        sign_neg = (per_model_diff < 0).sum("model")
        max_sign = xr.ufuncs.maximum(sign_pos, sign_neg) \
            if hasattr(xr, "ufuncs") else np.maximum(sign_pos, sign_neg)
        # xr.ufuncs is deprecated; use numpy.maximum on the underlying values
        from numpy import maximum as np_max
        max_sign = sign_pos.where(sign_pos >= sign_neg, sign_neg)
        n_valid = per_model_diff.notnull().sum("model")
        sign_frac = max_sign / n_valid.where(n_valid > 0, np.nan)
        return mean, sign_frac

    far_mean, far_sf = _stats(per_diff["far_minus_base"])
    mid_mean, mid_sf = _stats(per_diff["mid_minus_base"])
    inc_mean, inc_sf = _stats(per_diff["far_minus_mid"])

    div_vmax = float(np.nanpercentile(
        np.abs(np.concatenate([
            far_mean.values.ravel(),
            mid_mean.values.ravel(),
            inc_mean.values.ravel(),
        ])), 98))
    div_vmax = max(div_vmax, 1e-7)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.34),
                       constrained_layout=False)
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06],
                    wspace=0.14, left=0.06, right=0.94,
                    bottom=0.13, top=0.92, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    cax = fig.add_subplot(gs[0, 3])
    lat = stack["lat"].values
    lon_offset = float(BASINS[args.basin]["lon_offset"])
    lon = stack["rlon"].values + lon_offset
    _panel(ax0, far_mean, far_sf, div_vmax, "(a)",
            r"far $-$ baseline", lon, lat, lon_offset, show_ylabel=True)
    _panel(ax1, mid_mean, mid_sf, div_vmax, "(b)",
            r"mid $-$ baseline", lon, lat, lon_offset, show_ylabel=False)
    pm = _panel(ax2, inc_mean, inc_sf, div_vmax, "(c)",
                  r"far $-$ mid", lon, lat, lon_offset, show_ylabel=False)
    # single shared colorbar at the far right
    cb = plt.colorbar(pm, cax=cax)
    cb.set_label(r"$\Delta|\nabla\zeta|^2$  (m$^2$ s$^{-2}$)",
                   fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5)
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = MANUSCRIPT_FIGS / f"fig_eke_epoch_diff_{args.basin}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
