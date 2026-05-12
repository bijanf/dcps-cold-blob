"""Quiescence Signature: Fokker-Planck-derived analytical r(D) curve.

Step 1 of the theoretical-paper development.

Derivation (1-D weak-coupling limit):
  For a chain of nearest-neighbour-coupled Kuramoto oscillators with
  identical natural frequencies and per-site Gaussian white-noise
  amplitude D(x), the stationary phase-difference distribution
  satisfies the steady-state Fokker-Planck equation

      d_x[ D(x) d_x P ] + d_psi[ sin(psi) P ] = 0,

  where psi = phi_{i+1} - phi_i is the local phase difference.

  In the weak-coupling (small-K) limit, the steady-state pdf is
  approximately Boltzmann-like:

      P(psi; x) ~ exp[ -V(psi) / (D(x)/K) ],
      V(psi)   = 1 - cos(psi).

  The local order parameter is

      r(x) = <cos(psi)> = I_1(K/D) / I_0(K/D),

  where I_0, I_1 are modified Bessel functions.  This is the
  classical Kuramoto-on-a-circle order parameter at effective
  temperature D/K.

  For the small-K / large-D regime relevant to high-EKE ocean
  regions, the Bessel ratio simplifies to

      r(x) ~ K / (2 D(x))                       (1)

  which is a steeper relationship than the simple
  (1 + tau D)^{-1/2} form often cited.  We expose three closures
  in this module so the user can compare:

    closure 'bessel':  r = I_1(K/D) / I_0(K/D)   (exact)
    closure 'weak':    r = K / (2D)              (small K/D limit)
    closure 'simple':  r = 1 / sqrt(1 + tau D)   (one-parameter
                       phenomenological)

Validation:
  Compare each closure against observed (<r_loc>, EKE) scatter in
  NA, NP, ACC and report residual mean-squared error.

"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import i0, i1


def r_bessel(D, K):
    """Exact 1-D weak-coupling Kuramoto-on-a-circle steady-state r(D).

    r = I_1(K/D) / I_0(K/D); reduces to K/(2D) for large D/K.
    """
    D = np.asarray(D, dtype=float)
    ratio = np.where(D > 0, K / np.maximum(D, 1e-12), np.inf)
    return i1(ratio) / np.maximum(i0(ratio), 1e-30)


def r_weak(D, K):
    """Small-K asymptote: r ~ K / (2D)."""
    D = np.asarray(D, dtype=float)
    return np.where(D > 0, K / (2.0 * np.maximum(D, 1e-12)), 1.0)


def r_simple(D, tau):
    """One-parameter phenomenological closure: r = 1 / sqrt(1 + tau D)."""
    D = np.asarray(D, dtype=float)
    return 1.0 / np.sqrt(1.0 + tau * np.maximum(D, 0.0))


def fit_and_score(D, r_obs, closure="bessel"):
    """Fit the chosen closure to (D, r_obs) by least-squares and
    return (best_param, fitted_curve, residual_MSE).
    """
    valid = np.isfinite(D) & np.isfinite(r_obs)
    if valid.sum() < 5:
        return None, None, float("nan")
    D_v = D[valid]
    r_v = r_obs[valid]
    try:
        if closure == "bessel":
            popt, _ = curve_fit(r_bessel, D_v, r_v, p0=[1.0], maxfev=5000)
            r_fit = r_bessel(D_v, *popt)
        elif closure == "weak":
            popt, _ = curve_fit(r_weak, D_v, r_v, p0=[1.0], maxfev=5000)
            r_fit = r_weak(D_v, *popt)
        elif closure == "simple":
            popt, _ = curve_fit(r_simple, D_v, r_v, p0=[1.0], maxfev=5000)
            r_fit = r_simple(D_v, *popt)
        else:
            raise ValueError(f"unknown closure {closure!r}")
        mse = float(np.mean((r_v - r_fit) ** 2))
        return popt[0], r_fit, mse
    except Exception as e:
        return None, None, float("nan")


def main():
    """Validate the three closures against observed (<r_loc>, EKE)
    scatter for the North Atlantic basin.

    Uses cached arrays from the existing dcps/cache/eke_eddy_resolving
    directory (1/2-deg matched-resolution version).
    """
    import xarray as xr
    cache = Path("/home/bijanf/Documents/NEW_Theory/dcps/cache/eke_eddy_resolving")
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = cache / "rl_mean_2thdeg.nc"
    eke_path = cache / "eke_2thdeg.nc"
    if not (r_path.exists() and eke_path.exists()):
        print(f"Cached arrays not found; need to compute first.")
        return
    rl = xr.open_dataset(r_path)
    rl_var = rl[list(rl.data_vars)[0]]
    eke = xr.open_dataset(eke_path)
    eke_var = eke[list(eke.data_vars)[0]]

    r_arr = rl_var.values.ravel()
    D_arr = eke_var.values.ravel()
    valid = np.isfinite(r_arr) & np.isfinite(D_arr) & (D_arr > 0)
    r_v = r_arr[valid]
    D_v = D_arr[valid]
    print(f"  n cells: {r_v.size}; r range [{r_v.min():.2f}, {r_v.max():.2f}]; "
          f"D range [{D_v.min():.2e}, {D_v.max():.2e}]")

    results = {}
    for closure in ("bessel", "weak", "simple"):
        param, r_fit, mse = fit_and_score(D_v, r_v, closure=closure)
        results[closure] = dict(best_param=param, residual_mse=mse,
                                 n=int(valid.sum()))
        print(f"  closure='{closure}'  best param = {param}  MSE = {mse:.4f}")

    with open(out_dir / "theoretical_curve_fit.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_dir / 'theoretical_curve_fit.json'}")

    # Figure: observed scatter + fitted curves
    D_sort_idx = np.argsort(D_v)
    D_sorted = D_v[D_sort_idx]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.scatter(D_v, r_v, s=10, alpha=0.25, color="0.5", edgecolors="none",
                  label="observed (NA, 1/2 deg)")
    if results["bessel"]["best_param"] is not None:
        ax.plot(D_sorted, r_bessel(D_sorted, results["bessel"]["best_param"]),
                  color="C0", lw=2.5,
                  label=fr"Bessel closure, $K = {results['bessel']['best_param']:.2e}$, MSE={results['bessel']['residual_mse']:.4f}")
    if results["weak"]["best_param"] is not None:
        ax.plot(D_sorted, r_weak(D_sorted, results["weak"]["best_param"]),
                  color="C1", lw=2, linestyle="--",
                  label=fr"Weak-coupling, $K = {results['weak']['best_param']:.2e}$, MSE={results['weak']['residual_mse']:.4f}")
    if results["simple"]["best_param"] is not None:
        ax.plot(D_sorted, r_simple(D_sorted, results["simple"]["best_param"]),
                  color="C3", lw=2, linestyle=":",
                  label=fr"$1/\sqrt{{1+\tau D}}$, $\tau = {results['simple']['best_param']:.2e}$, MSE={results['simple']['residual_mse']:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$D$ = EKE  (m$^2$/s$^2$)")
    ax.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "fig_T2_theoretical_curve.pdf")
    plt.close(fig)
    print(f"Wrote {out_dir / 'fig_T2_theoretical_curve.pdf'}")


if __name__ == "__main__":
    main()
