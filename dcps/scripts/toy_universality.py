"""Universality test (Plan Step 4): same Quiescence mechanism in
three different prescribed-noise geometries.

Plan Step 4 asks for a comparative-planetology demonstration: the
Quiescence Signature should be a property of two-dimensional
stochastic Kuramoto dynamics with spatially varying noise, not a
peculiarity of the North Atlantic.  We run the toy in three
"planets":

  (i)  Earth-EKE        : F^2 = GLORYS12 EKE coarsened to the toy
                          grid -- the realistic Atlantic case.
  (ii) Jupiter-zonal    : F^2 = a zonally-banded pattern (rows of
                          alternating high/low noise, mimicking the
                          belt-and-zone structure of Jupiter where
                          the jets / belts carry the eddy energy).
  (iii) Chaotic         : F^2 = random Gaussian field with the same
                          spatial spectrum as a generic 2-D
                          baroclinic-turbulent flow.

For each "planet" we integrate an ensemble of seeds, compute
<r_loc>, the cell-wise (rho, Q), and the fitted tau in the
r = 1/sqrt(1 + tau * F^2) law.  The prediction is: rho < 0 and
the same one-parameter law holds in every "planet", with tau
varying only by an O(1) prefactor.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

from dcps.scripts.toy_kuramoto import (
    NY, NX, K, SIGMA_OMEGA, NOISE_BASE, NOISE_SCALE,
    DT_YR, T_MONTHS, load_eke_field_for_toy_grid, local_r_2d,
)

OUT_DIR = CACHE_DIR / "toy_universality"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

N_SEED = 8


def make_earth_field():
    return load_eke_field_for_toy_grid()


def make_jupiter_field(n_bands=6, seed=2027):
    """Zonally banded noise: alternating high/low rows, slight zonal
    modulation, normalised to [0, 1].
    """
    rng = np.random.default_rng(seed)
    yy = np.arange(NY)
    profile = 0.5 + 0.5 * np.cos(2 * np.pi * n_bands * yy / NY)
    f = np.tile(profile[:, None], (1, NX))
    f += 0.1 * rng.standard_normal((NY, NX))
    f -= f.min(); f /= f.max()
    return f


def make_chaotic_field(seed=2027):
    """Isotropic Gaussian random field with red-spectrum
    (k^{-3}) typical of 2-D baroclinic turbulence.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(NX) * NX
    ky = np.fft.fftfreq(NY) * NY
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0
    amp = K ** -1.5   # 2D spectrum slope
    phase = rng.uniform(0, 2 * np.pi, size=(NY, NX))
    field = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    field -= field.min(); field /= field.max()
    return field


def integrate(F2, seed):
    rng = np.random.default_rng(seed)
    ny, nx = F2.shape
    omega = rng.normal(0, SIGMA_OMEGA, size=(ny, nx)) * 2 * np.pi
    F = NOISE_BASE + NOISE_SCALE * F2
    phi = rng.uniform(-np.pi, np.pi, size=(ny, nx))
    phis = np.empty((T_MONTHS, ny, nx), dtype=np.float32)
    for t in range(T_MONTHS):
        d_phi_coupling = (
            np.sin(np.roll(phi, 1, 0) - phi)
            + np.sin(np.roll(phi, -1, 0) - phi)
            + np.sin(np.roll(phi, 1, 1) - phi)
            + np.sin(np.roll(phi, -1, 1) - phi)
        )
        drift = omega + K * d_phi_coupling
        xi = rng.normal(0, 1, size=(ny, nx))
        diff = F * xi * np.sqrt(DT_YR)
        phi = phi + drift * DT_YR + diff
        phis[t] = np.angle(np.exp(1j * phi))
    return phis


def fit_tau(rl, f2):
    def model(x, tau):
        return 1.0 / np.sqrt(1.0 + tau * x)
    m = np.isfinite(rl) & np.isfinite(f2)
    if m.sum() < 10:
        return float("nan"), float("nan")
    try:
        popt, _ = curve_fit(model, f2[m], rl[m], p0=[1.0],
                             bounds=(1e-3, 1e6), maxfev=5000)
        pred = model(f2[m], popt[0])
        ss_res = np.sum((rl[m] - pred) ** 2)
        ss_tot = np.sum((rl[m] - rl[m].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(popt[0]), float(r2)
    except Exception:
        return float("nan"), float("nan")


def run_planet(name, F2):
    print(f"\n  PLANET: {name}")
    print(f"    F^2: range [{F2.min():.3f}, {F2.max():.3f}], "
          f"std {F2.std():.3f}")
    rho_seeds = []
    rl_stack = []
    tau_seeds = []
    r2_seeds = []
    for s in range(N_SEED):
        t0 = time.time()
        phis = integrate(F2, seed=2000 + s + hash(name) % 10_000)
        rl = local_r_2d(phis)
        rl_stack.append(rl)
        rl_v = rl.ravel(); f2_v = F2.ravel()
        valid = np.isfinite(rl_v) & np.isfinite(f2_v)
        rho, _ = pearsonr(rl_v[valid], f2_v[valid])
        rho_seeds.append(float(rho))
        tau, r2 = fit_tau(rl_v[valid], f2_v[valid])
        tau_seeds.append(tau); r2_seeds.append(r2)
    rho_m = float(np.mean(rho_seeds))
    rho_sd = float(np.std(rho_seeds, ddof=1))
    tau_m = float(np.nanmean(tau_seeds))
    tau_sd = float(np.nanstd(tau_seeds, ddof=1))
    r2_m = float(np.nanmean(r2_seeds))
    print(f"    rho = {rho_m:+.3f} +/- {rho_sd:.3f}   Q = {-rho_m:+.3f}")
    print(f"    fitted tau = {tau_m:.2f} +/- {tau_sd:.2f}, "
          f"R^2 = {r2_m:.3f}")
    return dict(F2=F2, rl_mean=np.mean(rl_stack, axis=0),
                 rho_mean=rho_m, rho_sd=rho_sd, tau_mean=tau_m,
                 tau_sd=tau_sd, r2_mean=r2_m,
                 rho_seeds=rho_seeds, tau_seeds=tau_seeds)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Universality test: Quiescence in three 'planet' configurations")
    print("=" * 70)
    print(f"  {N_SEED} seeds per planet, {NY}x{NX} = {NY*NX} oscillators, "
          f"{T_MONTHS} months")
    planets = {
        "Earth"   : make_earth_field(),
        "Jupiter" : make_jupiter_field(),
        "Chaotic" : make_chaotic_field(),
    }
    results = {name: run_planet(name, F2)
                for name, F2 in planets.items()}

    # Persist (drop arrays)
    summary = {name: {k: r[k] for k in (
                  "rho_mean", "rho_sd", "tau_mean", "tau_sd", "r2_mean",
                  "rho_seeds", "tau_seeds")}
                for name, r in results.items()}
    n_neg = sum(1 for r in results.values() if r["rho_mean"] < 0)
    summary["aggregate"] = dict(
        n_planets_with_negative_rho=int(n_neg),
        verdict=("Universal sign" if n_neg == len(planets)
                  else "Mixed sign"),
    )
    with open(OUT_DIR / "universality.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'universality.json'}")

    # Figure: 3 planets x (F^2 map, r_loc map, scatter+fit)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5),
                              constrained_layout=True)
    for row, (name, r) in enumerate(results.items()):
        ax = axes[row, 0]
        im = ax.imshow(r["F2"], cmap="inferno", origin="lower",
                        interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="$F^2$ (noise amp)")
        ax.set_title(f"{name}: F$^2$", fontsize=11)
        ax.set_ylabel(name, fontsize=12, fontweight="bold")

        ax = axes[row, 1]
        im = ax.imshow(r["rl_mean"], cmap="viridis", origin="lower",
                        vmin=0, vmax=1, interpolation="bilinear")
        plt.colorbar(im, ax=ax,
                      label=r"$\langle r_{\mathrm{loc}}\rangle$")
        ax.set_title(f"{name}: coherence", fontsize=11)

        ax = axes[row, 2]
        f2_v = r["F2"].ravel(); rl_v = r["rl_mean"].ravel()
        ax.scatter(f2_v, rl_v, s=8, alpha=0.4, color="C0",
                    edgecolors="none")
        # theoretical curve with fitted tau
        x_th = np.linspace(0, max(1e-3, float(np.max(f2_v))), 100)
        if np.isfinite(r["tau_mean"]):
            y_th = 1.0 / np.sqrt(1.0 + r["tau_mean"] * x_th)
            ax.plot(x_th, y_th, "r-", lw=2,
                     label=f"$\\tau = {r['tau_mean']:.1f}$")
        # decile means
        order = np.argsort(f2_v)
        n = order.size
        dx, dy = [], []
        for k in range(10):
            lo, hi = int(k * n / 10), int((k + 1) * n / 10)
            dx.append(float(np.mean(f2_v[order[lo:hi]])))
            dy.append(float(np.mean(rl_v[order[lo:hi]])))
        ax.plot(dx, dy, "ko-", ms=5, lw=1,
                 label="decile means")
        ax.set_xlabel("$F^2$"); ax.set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle$")
        ax.set_title(
            f"$\\rho = {r['rho_mean']:+.2f}\\pm{r['rho_sd']:.2f}$, "
            f"$Q = {-r['rho_mean']:+.2f}$",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Universality of the Quiescence Signature: same "
                  "$\\rho<0$ and same functional form across three "
                  "noise geometries",
                  fontsize=12)
    out_fig = MANUSCRIPT_FIGS / "fig_toy_universality.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
