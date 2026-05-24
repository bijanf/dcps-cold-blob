"""Multi-model EKE difference maps with sign-agreement stippling.

For every per-model NetCDF in dcps/cache/eke_epoch_diff/ (produced by
compute_eke_epoch_diff.py), compute scenario-resolved mid-baseline and
far-baseline EKE differences, then render the multi-model mean of each.

Significance overlay (IPCC AR6 style): stipple grid cells where
>= 80 % of contributing models agree on the sign of the change.

Output: manuscript/figs/fig_eke_epoch_diff_<basin>.pdf

Four panels (2 rows x 2 cols), shared colour scale on the right:
  Row 1 (SSP5-8.5): (a) mid - baseline   (b) far - baseline
  Row 2 (SSP2-4.5): (c) mid - baseline   (d) far - baseline

Baseline 1850-1900 is shared across scenarios (historical period);
each row therefore uses the same baseline data, drawn from the ssp585
cache files (which is where the baseline lives).  The mid/far data
for SSP2-4.5 come from <model>_<basin>_epoch_diff_ssp245.nc files.
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


def _stack(basin: str, experiment: str = "ssp585"
            ) -> tuple[xr.Dataset, list[str]]:
    """Stack per-model epoch-diff NetCDFs for one scenario.

    For ssp585 the cache file is <model>_<basin>_epoch_diff.nc with
    baseline+mid+far vars.  For ssp245 the cache file is
    <model>_<basin>_epoch_diff_ssp245.nc with mid+far only; the shared
    baseline is loaded from the corresponding ssp585 file.
    """
    suffix = "" if experiment == "ssp585" else f"_{experiment}"
    files = sorted(DIFF_DIR.glob(f"*_{basin}_epoch_diff{suffix}.nc"))
    if not files:
        raise FileNotFoundError(
            f"no diff NetCDFs for {experiment} in {DIFF_DIR}")
    dss, models = [], []
    base = None
    for f in files:
        try:
            ds = xr.open_dataset(f)
            model = ds.attrs.get("model", f.stem.split("_")[0])
            # For ssp245 files, attach the shared baseline from ssp585
            if experiment != "ssp585" and "baseline" not in ds.data_vars:
                base_path = DIFF_DIR / f"{model}_{basin}_epoch_diff.nc"
                if not base_path.exists():
                    print(f"  skip {model}: no ssp585 baseline cache")
                    continue
                base_ds = xr.open_dataset(base_path)
                ds = ds.assign(baseline=base_ds["baseline"])
            if base is None: base = ds
            else: ds = ds.reindex_like(base, method=None)
            dss.append(ds)
            models.append(model)
        except Exception as e:
            print(f"  skip {f.name}: {type(e).__name__}: {e}")
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


def _stats(per_model_diff):
    """Multi-model mean + sign-agreement fraction (max(pos, neg) / N_valid)."""
    mean = per_model_diff.mean("model", skipna=True)
    sign_pos = (per_model_diff > 0).sum("model")
    sign_neg = (per_model_diff < 0).sum("model")
    max_sign = sign_pos.where(sign_pos >= sign_neg, sign_neg)
    n_valid = per_model_diff.notnull().sum("model")
    sign_frac = max_sign / n_valid.where(n_valid > 0, np.nan)
    return mean, sign_frac


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    stack_585, mods_585 = _stack(args.basin, experiment="ssp585")
    try:
        stack_245, mods_245 = _stack(args.basin, experiment="ssp245")
    except FileNotFoundError as e:
        print(f"  SSP2-4.5 cache missing ({e}); rendering SSP5-8.5 only")
        stack_245, mods_245 = None, []
    print(f"basin={args.basin}  N_ssp585={len(mods_585)}  "
          f"N_ssp245={len(mods_245)}")

    mid585_mean, mid585_sf = _stats(stack_585["mid"] - stack_585["baseline"])
    far585_mean, far585_sf = _stats(stack_585["far"] - stack_585["baseline"])
    if stack_245 is not None:
        mid245_mean, mid245_sf = _stats(stack_245["mid"] - stack_245["baseline"])
        far245_mean, far245_sf = _stats(stack_245["far"] - stack_245["baseline"])
    else:
        mid245_mean = far245_mean = None
        mid245_sf = far245_sf = None

    # Symmetric vmax = 98th percentile across all four mean maps
    all_vals = [mid585_mean.values.ravel(), far585_mean.values.ravel()]
    if mid245_mean is not None:
        all_vals += [mid245_mean.values.ravel(), far245_mean.values.ravel()]
    div_vmax = float(np.nanpercentile(np.abs(np.concatenate(all_vals)), 98))
    div_vmax = max(div_vmax, 1e-7)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.62),
                       constrained_layout=False)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.06],
                    height_ratios=[1.0, 1.0],
                    wspace=0.14, hspace=0.30,
                    left=0.06, right=0.94,
                    bottom=0.08, top=0.94, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a, sharex=ax_a)
    ax_c = fig.add_subplot(gs[1, 0], sharey=ax_a, sharex=ax_a)
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_a, sharex=ax_a)
    cax = fig.add_subplot(gs[:, 2])

    lat = stack_585["lat"].values
    lon_offset = float(BASINS[args.basin]["lon_offset"])
    lon = stack_585["rlon"].values + lon_offset

    _panel(ax_a, mid585_mean, mid585_sf, div_vmax, "(a)",
            r"SSP5-8.5: mid (2030-2060) $-$ baseline",
            lon, lat, lon_offset, show_ylabel=True)
    _panel(ax_b, far585_mean, far585_sf, div_vmax, "(b)",
            r"SSP5-8.5: far (2070-2099) $-$ baseline",
            lon, lat, lon_offset, show_ylabel=False)
    if mid245_mean is not None:
        _panel(ax_c, mid245_mean, mid245_sf, div_vmax, "(c)",
                r"SSP2-4.5: mid (2030-2060) $-$ baseline",
                lon, lat, lon_offset, show_ylabel=True)
        pm = _panel(ax_d, far245_mean, far245_sf, div_vmax, "(d)",
                      r"SSP2-4.5: far (2070-2099) $-$ baseline",
                      lon, lat, lon_offset, show_ylabel=False)
    else:
        ax_c.set_visible(False); ax_d.set_visible(False)
        pm = ax_b.collections[0]

    # remove x labels on row 1 to avoid overlap with row 2 titles
    for ax in (ax_a, ax_b):
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")

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
