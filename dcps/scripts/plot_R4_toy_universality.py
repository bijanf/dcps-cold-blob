"""Plot R4 (reviewer-driven): 2x3 universality panel for the 2-D toy Kuramoto.

Top row (a, b, c): prescribed noise amplitude F^2 for Earth-EKE,
Jupiter-like (zonal bands), and Chaotic (isotropic k^-3 turbulence).

Bottom row (d, e, f): synthetic time-mean <r_loc> from the stochastic
2-D Kuramoto integration on each noise geometry.  Each panel has its
own colour bar attached to the right via make_axes_locatable.

Linear units only.  Panel letters outside the axes (top-left corner,
above the axes spine).  Per-column rho and Q go in the figure caption,
not in the axes themselves.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style, DOUBLE_COL_IN
apply_nature_style()

# toy_kuramoto.py exposes integrate_kuramoto, local_r_2d, NX/NY,
# DT_YR/T_MONTHS, SIGMA_OMEGA, NOISE_BASE, NOISE_SCALE, K.  Reuse them
# rather than re-deriving constants.
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from toy_kuramoto import (  # noqa: E402
    NY, NX, K, SIGMA_OMEGA, NOISE_BASE, NOISE_SCALE, DT_YR, T_MONTHS,
    local_r_2d, load_eke_field_for_toy_grid,
)


MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

N_SEED = 8


def make_earth_field():
    return load_eke_field_for_toy_grid()


def make_jupiter_field(n_bands=6, seed=2027):
    rng = np.random.default_rng(seed)
    yy = np.arange(NY)
    profile = 0.5 + 0.5 * np.cos(2 * np.pi * n_bands * yy / NY)
    f = np.tile(profile[:, None], (1, NX))
    f += 0.1 * rng.standard_normal((NY, NX))
    f -= f.min(); f /= f.max()
    return f


def make_chaotic_field(seed=2027):
    """Isotropic Gaussian random field with k^-3 spatial spectrum."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(NX) * NX
    ky = np.fft.fftfreq(NY) * NY
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K_mag = np.sqrt(KX ** 2 + KY ** 2)
    K_mag[0, 0] = 1.0
    amp = K_mag ** -1.5
    phase = rng.uniform(0, 2 * np.pi, size=(NY, NX))
    field = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    field -= field.min(); field /= field.max()
    return field


GEOMETRIES = (
    ("Earth EKE",    make_earth_field),
    ("Jupiter-like", make_jupiter_field),
    ("Chaotic",      make_chaotic_field),
)


def _integrate(F2, seed):
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


def _attach_cb(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.07,
                                axes_class=plt.Axes)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return cb


def _ensemble(name, make_field):
    F2 = make_field()
    rl_stack = []
    rho_seeds = []
    for s in range(N_SEED):
        phis = _integrate(F2, seed=2000 + s + hash(name) % 10_000)
        rl = local_r_2d(phis)
        rl_stack.append(rl)
        rl_v = rl.ravel(); f2_v = F2.ravel()
        m = np.isfinite(rl_v) & np.isfinite(f2_v)
        rho, _ = pearsonr(rl_v[m], f2_v[m])
        rho_seeds.append(float(rho))
    rl_mean = np.mean(rl_stack, axis=0)
    return F2, rl_mean, float(np.mean(rho_seeds)), float(np.std(rho_seeds, ddof=1))


def main():
    runs = {}
    for name, make_field in GEOMETRIES:
        F2, rl_mean, rho_m, rho_sd = _ensemble(name, make_field)
        runs[name] = dict(F2=F2, rl=rl_mean, rho=rho_m, rho_sd=rho_sd)
        print(f"  {name}: rho = {rho_m:+.3f} +/- {rho_sd:.3f}, "
              f"Q = {-rho_m:+.3f}")

    fig, axes = plt.subplots(
        2, 3, figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.55),
        constrained_layout=True,
    )

    panel_letters = (("a", "b", "c"), ("d", "e", "f"))

    for col, (name, _) in enumerate(GEOMETRIES):
        ax = axes[0, col]
        F2 = runs[name]["F2"]
        im = ax.imshow(F2, cmap="inferno", origin="lower",
                        interpolation="nearest",
                        vmin=float(np.nanmin(F2)),
                        vmax=float(np.nanmax(F2)))
        _attach_cb(fig, ax, im, r"$F^{2}$  (linear)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(-0.06, 1.04, panel_letters[0][col], transform=ax.transAxes,
                 fontweight="bold", fontsize=10, va="bottom", ha="left")

    for col, (name, _) in enumerate(GEOMETRIES):
        ax = axes[1, col]
        rl = runs[name]["rl"]
        im = ax.imshow(rl, cmap="viridis", origin="lower",
                        interpolation="nearest",
                        vmin=float(np.nanpercentile(rl, 2)),
                        vmax=float(np.nanpercentile(rl, 98)))
        _attach_cb(fig, ax, im,
                    r"$\langle r_{\mathrm{loc}}\rangle$  (linear)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(-0.06, 1.04, panel_letters[1][col], transform=ax.transAxes,
                 fontweight="bold", fontsize=10, va="bottom", ha="left")

    out = MANUSCRIPT_FIGS / "fig_R4_toy_universality.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    summary = {name: {"rho_mean": runs[name]["rho"],
                        "rho_sd":   runs[name]["rho_sd"],
                        "Q_mean":   -runs[name]["rho"]}
                for name, _ in GEOMETRIES}
    out_json = CACHE_DIR / "toy_universality" / "plot_R4_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
