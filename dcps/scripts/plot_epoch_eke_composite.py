"""Multi-model composite EKE maps across piControl, historical, ssp585.

Reads every NetCDF in ``dcps/cache/eke_maps/<model>_<basin>_epoch_eke.nc``
and builds a 2x3 panel:

  (a) piControl multi-model-mean |grad SSH|^2 (Holocene-equivalent ceiling)
  (b) historical mean (1850-2014)
  (c) ssp585 mean (2015-2099)
  (d) historical minus piControl  (industrial-era anomaly)
  (e) ssp585 minus piControl       (full warming signature)
  (f) ssp585 minus historical     (projection-era acceleration)

Resilient to incomplete caches: renders with however many models are
currently available; N appears in the caption text.

Output: manuscript/figs/fig_epoch_eke_composite_<basin>.pdf
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

EKE_DIR = CACHE_DIR / "eke_maps"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def _common_grid_stack(basin: str) -> tuple[xr.Dataset, list[str]]:
    """Stack every model's NetCDF along a 'model' dim.  Models that fail
    to align to the common (lat, rlon) basin grid are skipped."""
    files = sorted(EKE_DIR.glob(f"*_{basin}_epoch_eke.nc"))
    if not files:
        raise FileNotFoundError(f"no EKE NetCDFs in {EKE_DIR}")
    datasets, models = [], []
    base = None
    for f in files:
        try:
            ds = xr.open_dataset(f)
            if base is None:
                base = ds
            else:
                ds = ds.reindex_like(base, method=None)
            datasets.append(ds)
            models.append(ds.attrs.get("model", f.stem.split("_")[0]))
        except Exception as e:
            print(f"  skip {f.name}: {type(e).__name__}")
    stack = xr.concat(datasets, dim="model").assign_coords(model=models)
    return stack, models


def _panel(ax, field, title, cmap, vmin, vmax, lon_offset,
            divergent=False):
    lon = field["rlon"].values + lon_offset
    lat = field["lat"].values
    pm = ax.pcolormesh(lon, lat, field.values, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto")
    ax.set_facecolor("0.85")
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3.5%", pad=0.05)
    cb = plt.colorbar(pm, cax=cax)
    cb.set_label(title, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    args = ap.parse_args()

    stack, models = _common_grid_stack(args.basin)
    N = len(models)
    print(f"basin={args.basin}  N_models_in_cache={N}")
    print(f"  models: {', '.join(models)}")

    mean_pi = stack["pi"].mean("model", skipna=True)
    mean_hi = stack["historical"].mean("model", skipna=True)
    mean_sp = (stack["ssp585"].mean("model", skipna=True)
                if "ssp585" in stack else None)

    raw_vmax = float(np.nanpercentile(
        np.concatenate([mean_pi.values.ravel(),
                          mean_hi.values.ravel(),
                          mean_sp.values.ravel() if mean_sp is not None
                          else np.array([np.nan])]), 98))
    raw_vmax = max(raw_vmax, 1e-6)

    diff_hi = mean_hi - mean_pi
    diff_sp = (mean_sp - mean_pi) if mean_sp is not None else None
    diff_sphi = (mean_sp - mean_hi) if mean_sp is not None else None
    div_vmax = float(np.nanpercentile(
        np.abs(np.concatenate([
            diff_hi.values.ravel(),
            diff_sp.values.ravel() if diff_sp is not None
            else np.array([np.nan]),
            diff_sphi.values.ravel() if diff_sphi is not None
            else np.array([np.nan]),
        ])), 98))
    div_vmax = max(div_vmax, 1e-7)

    fig, axes = plt.subplots(
        2, 3, figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.55),
        gridspec_kw=dict(wspace=0.55, hspace=0.45),
    )
    lon_offset = float(BASINS[args.basin]["lon_offset"])
    panels = [
        (axes[0, 0], mean_pi, r"piControl  $\langle|\nabla\zeta|^2\rangle$",
         "viridis", 0, raw_vmax, False, "(a)"),
        (axes[0, 1], mean_hi, r"historical  $\langle|\nabla\zeta|^2\rangle$",
         "viridis", 0, raw_vmax, False, "(b)"),
        (axes[0, 2],
         mean_sp if mean_sp is not None else mean_hi * np.nan,
         r"ssp585  $\langle|\nabla\zeta|^2\rangle$",
         "viridis", 0, raw_vmax, False, "(c)"),
        (axes[1, 0], diff_hi, r"historical $-$ piControl",
         "RdBu_r", -div_vmax, div_vmax, True, "(d)"),
        (axes[1, 1],
         diff_sp if diff_sp is not None else diff_hi * np.nan,
         r"ssp585 $-$ piControl",
         "RdBu_r", -div_vmax, div_vmax, True, "(e)"),
        (axes[1, 2],
         diff_sphi if diff_sphi is not None else diff_hi * np.nan,
         r"ssp585 $-$ historical",
         "RdBu_r", -div_vmax, div_vmax, True, "(f)"),
    ]
    for ax, field, title, cmap, vmin, vmax, div, letter in panels:
        _panel(ax, field, title, cmap, vmin, vmax, lon_offset, div)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes,
                  ha="left", va="top", fontsize=8, fontweight="bold")
    fig.text(0.5, -0.02,
              f"N = {N} CMIP6 models (Atlantic 2 deg basin grid). "
              f"Anomalies are multi-model means of |grad zeta|^2 in m^2. "
              f"Red = stronger EKE; blue = weaker EKE.",
              ha="center", va="top", fontsize=6, color="0.3")
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    out_pdf = (MANUSCRIPT_FIGS
                / f"fig_epoch_eke_composite_{args.basin}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
