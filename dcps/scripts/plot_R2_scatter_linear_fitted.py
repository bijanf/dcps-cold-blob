"""Plot R2 (reviewer-driven): scatter <r_loc> vs EKE on a LINEAR x-axis.

Hexbin density plus the fitted one-parameter curve
    <r_loc> = 1 / sqrt(1 + tau * EKE)
on a LINEAR EKE axis.  No log scaling anywhere.  Decile-mean red circles
guide the eye.  rho, Q and the fitted tau-hat are annotated in a corner
text block; the same statistics also appear in the caption.
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, SINGLE_COL_IN
apply_nature_style()


EKE_DIR = CACHE_DIR / "eke_eddy_resolving"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

EQ_BAND_LAT = 15.0
TAU_INIT = 144.0  # pre-registered NA observational value


def _quiescence(x, tau):
    """One-parameter quiescence curve y = 1 / sqrt(1 + tau * x)."""
    return 1.0 / np.sqrt(1.0 + tau * x)


def _load_cells():
    rl_ds = xr.open_dataset(EKE_DIR / "rl_mean_2deg_glorys12.nc")
    rl = rl_ds[list(rl_ds.data_vars)[0]]
    eke = xr.open_dataset(EKE_DIR / "eke_2deg_from_native.nc")["EKE_box_avg_2deg"]
    lat = rl["lat"].values
    rlon = rl["rlon"].values
    eq = (np.abs(lat) < EQ_BAND_LAT)[:, None]
    R = rl.values.astype(float)
    E = eke.values.astype(float)
    R[np.broadcast_to(eq, R.shape)] = np.nan
    E[np.broadcast_to(eq, E.shape)] = np.nan
    r_flat = R.ravel()
    e_flat = E.ravel()
    m = np.isfinite(r_flat) & np.isfinite(e_flat) & (e_flat >= 0)
    return r_flat[m], e_flat[m]


def main():
    y, x_m2s2 = _load_cells()         # y = <r_loc>, x_m2s2 = EKE [m^2/s^2]
    x = x_m2s2 * 1e4                   # convert to cm^2/s^2 for O(1) axis
    rho, _ = pearsonr(x, y)
    Q = -float(rho)

    # Fit tau on the LINEAR data.  curve_fit on cm^2/s^2: tau will be in
    # units of (cm^2/s^2)^-1; we report it back in m^2/s^2 units so it
    # matches the pre-registered value tau = 144 (m^2/s^2)^-1.
    try:
        popt, _pcov = curve_fit(
            _quiescence, x_m2s2, y, p0=[TAU_INIT],
            bounds=(1e-3, 1e6), maxfev=5000,
        )
        tau_hat = float(popt[0])
    except Exception:
        tau_hat = float("nan")
    print(f"Plot R2: rho = {rho:.3f}, Q = {Q:.3f}, "
          f"fitted tau = {tau_hat:.2f} (m^2/s^2)^-1, n = {len(y)}")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.85))

    # ---- hexbin density (LINEAR x) -----------------------------------
    hb = ax.hexbin(x, y, gridsize=28, mincnt=1,
                    cmap="Blues", linewidths=0.0)

    # ---- fitted curve -------------------------------------------------
    x_grid_m2s2 = np.linspace(0.0, float(np.nanmax(x_m2s2)) * 1.02, 400)
    y_th = _quiescence(x_grid_m2s2, tau_hat)
    ax.plot(x_grid_m2s2 * 1e4, y_th, color="C3", lw=1.4,
             label=fr"$y=(1+\hat\tau\,\mathrm{{EKE}})^{{-1/2}}$, "
                   fr"$\hat\tau={tau_hat:.0f}$ (m$^2$/s$^2$)$^{{-1}}$")

    # ---- decile means (LINEAR EKE bins) -------------------------------
    n_bins = 12
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if sel.sum() >= 4:
            means[i] = y[sel].mean()
    ax.plot(centres, means, "o", color="C3", markersize=4,
             markeredgewidth=0.0, label="decile mean")

    ax.set_xlabel(r"EKE  (cm$^2$/s$^2$)")
    ax.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax.set_xlim(left=0)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5,
               handlelength=1.6)

    # density colour bar (small, separate)
    cb = fig.colorbar(hb, ax=ax, shrink=0.6, pad=0.02, fraction=0.05)
    cb.set_label("cell count", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    out = MANUSCRIPT_FIGS / "fig_R2_scatter_linear_fitted.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
