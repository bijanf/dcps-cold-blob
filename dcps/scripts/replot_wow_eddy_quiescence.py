"""Replot fig_wow_eddy_quiescence.pdf from cached 1/2-degree
EKE and <r_loc> NetCDFs without re-running the GLORYS pipeline.

Follows the project figure-quality standard:
  - no in-plot titles
  - bold corner panel letters as the only in-axis identifier
  - no in-plot statistics text box (statistics live in the caption)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr


TARGET_DEG = 0.5


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eke", type=Path, required=True,
                   help="Cached EKE NetCDF on 1/2 deg basin grid.")
    p.add_argument("--rloc", type=Path, required=True,
                   help="Cached <r_loc> NetCDF on 1/2 deg basin grid.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PDF path.")
    args = p.parse_args(argv)

    eke_target = xr.open_dataarray(args.eke) if args.eke.suffix == ".nc" else None
    if eke_target is None:
        with xr.open_dataset(args.eke) as ds:
            eke_target = ds[list(ds.data_vars)[0]]

    rl_mean = xr.open_dataarray(args.rloc) if args.rloc.suffix == ".nc" else None
    if rl_mean is None:
        with xr.open_dataset(args.rloc) as ds:
            rl_mean = ds[list(ds.data_vars)[0]]

    LAT = eke_target["lat"].values
    LON = eke_target["lon"].values

    eke_v = eke_target.values.copy()
    R = rl_mean.values.copy()
    common_valid = np.isfinite(eke_v) & np.isfinite(R)
    eke_v[~common_valid] = np.nan
    R[~common_valid] = np.nan

    a_flat = R.ravel()
    b_flat = eke_v.ravel()
    m = np.isfinite(a_flat) & np.isfinite(b_flat)
    rho = float(np.corrcoef(a_flat[m], b_flat[m])[0, 1])
    n = int(m.sum())
    print(f"rho = {rho:+.3f}, n = {n}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(14.5, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.05, 3, 0.05, 2])
    ax_eke = fig.add_subplot(gs[0, 0])
    cax_eke = fig.add_subplot(gs[0, 1])
    ax_rl = fig.add_subplot(gs[0, 2])
    cax_rl = fig.add_subplot(gs[0, 3])
    ax_sc = fig.add_subplot(gs[0, 4])

    extent = [LON.min() - TARGET_DEG / 2, LON.max() + TARGET_DEG / 2,
              LAT.min() - TARGET_DEG / 2, LAT.max() + TARGET_DEG / 2]

    # Panel (a): 1/2-degree EKE on a log scale.
    eke_lo = max(np.nanpercentile(eke_v, 5), 1e-4)
    eke_hi = np.nanpercentile(eke_v, 99)
    im_eke = ax_eke.imshow(
        eke_v, extent=extent, origin="lower", aspect="auto",
        cmap="inferno", norm=LogNorm(vmin=eke_lo, vmax=eke_hi),
        interpolation="bilinear", rasterized=True,
    )
    fig.colorbar(im_eke, cax=cax_eke,
                  label=r"Geostrophic EKE (m$^{2}$/s$^{2}$)")
    cb_box_a = Rectangle((-50, 45), width=35, height=15, linewidth=2.0,
                          edgecolor="white", facecolor="none",
                          linestyle="--", zorder=10)
    ax_eke.add_patch(cb_box_a)
    ax_eke.set_xlabel("Longitude")
    ax_eke.set_ylabel("Latitude")
    ax_eke.set_xlim(-80, 0)
    ax_eke.set_ylim(0, 75)
    ax_eke.text(0.02, 0.97, "a", transform=ax_eke.transAxes,
                 fontweight="bold", fontsize=14, va="top",
                 bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.85, pad=2.0))

    # Panel (b): 1/2-degree <r_loc>.
    im_rl = ax_rl.imshow(
        R, extent=extent, origin="lower", aspect="auto",
        cmap="viridis", interpolation="bilinear", rasterized=True,
        vmin=np.nanpercentile(R, 3), vmax=np.nanpercentile(R, 97),
    )
    fig.colorbar(im_rl, cax=cax_rl,
                  label=r"$\langle r_{\mathrm{loc}}\rangle_t$  (phase coherence)")
    cb_box_b = Rectangle((-50, 45), width=35, height=15, linewidth=2.0,
                          edgecolor="white", facecolor="none",
                          linestyle="--", zorder=10)
    ax_rl.add_patch(cb_box_b)
    ax_rl.set_xlabel("Longitude")
    ax_rl.set_ylabel("Latitude")
    ax_rl.set_xlim(-80, 0)
    ax_rl.set_ylim(0, 75)
    ax_rl.text(0.02, 0.97, "b", transform=ax_rl.transAxes,
                fontweight="bold", fontsize=14, va="top",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=2.0))

    # Panel (c): scatter — NO in-plot statistics text box (rho and n
    # in the caption per the project figure-quality standard).
    ax_sc.scatter(b_flat[m], a_flat[m], s=8, alpha=0.35, color="C0",
                   edgecolors="none")
    ax_sc.set_xscale("log")
    if m.sum() > 30:
        x_pos = b_flat[m][b_flat[m] > 0]
        y_pos = a_flat[m][b_flat[m] > 0]
        coef = np.polyfit(np.log10(x_pos), y_pos, 1)
        xline = np.logspace(np.log10(x_pos.min()),
                              np.log10(x_pos.max()), 50)
        ax_sc.plot(xline, np.polyval(coef, np.log10(xline)),
                     color="C3", linewidth=2.5)
    ax_sc.set_xlabel(r"EKE 1/12$^{\circ}\!\to\!1/2^{\circ}$ (m$^{2}$/s$^{2}$)")
    ax_sc.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$  (1/2$^{\circ}$)")
    ax_sc.text(0.02, 0.97, "c", transform=ax_sc.transAxes,
                fontweight="bold", fontsize=14, va="top",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=2.0))
    ax_sc.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
