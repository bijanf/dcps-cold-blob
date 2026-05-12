"""Causal test of the eddy-noise mechanism.

Builds on toy_kuramoto.py.  Two configurations:

  CONTROL    : 2-D stochastic Kuramoto with EKE-prescribed noise
               F = NOISE_BASE + NOISE_SCALE * F2(x, y)   (as in main toy)
  EDDY-NULL  : identical mean flow / coupling / natural frequencies /
               random seed, but with noise driven to NOISE_BASE inside
               the high-EKE patch (top-quartile cells).  Outside the
               patch the noise is unchanged.

If EKE acts causally as a phase-decohering noise, then <r_loc>_t in
the patch must rise from its CONTROL value to ~ its low-EKE value
under EDDY-NULL.  This is the causal counterpart of the basin-wide
spatial anti-correlation reported in the main text.

We bootstrap an ensemble of N_SEED seeds and report Delta r in the
patch with a 95% CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

from dcps.scripts.toy_kuramoto import (
    NY, NX, K, SIGMA_OMEGA, NOISE_BASE, NOISE_SCALE, RADIUS_CELLS,
    DT_YR, T_MONTHS, load_eke_field_for_toy_grid, local_r_2d,
)

OUT_DIR = CACHE_DIR / "toy_kuramoto_causal"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

N_SEED = 24                  # ensemble size for bootstrap
PATCH_QUARTILE = 0.75        # define "high-EKE patch" as cells in this F^2 quantile


def integrate(F2_field, F_override_mask=None, seed=0):
    """Integrate the toy Kuramoto.

    If F_override_mask is given, cells where mask is True have their
    noise amplitude held at NOISE_BASE regardless of local F2 (i.e.
    eddy-nulled).  All other cells use the normal EKE-prescribed F.
    """
    rng = np.random.default_rng(seed)
    ny, nx = F2_field.shape
    omega = rng.normal(0, SIGMA_OMEGA, size=(ny, nx)) * 2 * np.pi
    F = NOISE_BASE + NOISE_SCALE * F2_field
    if F_override_mask is not None:
        F = np.where(F_override_mask, NOISE_BASE, F)
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Causal test: noise-toggle 2-D Kuramoto")
    print("=" * 70)

    F2 = load_eke_field_for_toy_grid()
    patch_thresh = float(np.quantile(F2, PATCH_QUARTILE))
    patch_mask = F2 > patch_thresh
    n_patch = int(patch_mask.sum())
    print(f"  Patch defined as top {(1-PATCH_QUARTILE)*100:.0f}% of F2 "
          f"({n_patch} of {NY*NX} cells).")

    # Single-seed reference for the figure
    print("  Integrating CONTROL  ...")
    t0 = time.time()
    phis_ctrl = integrate(F2, F_override_mask=None, seed=0)
    rl_ctrl = local_r_2d(phis_ctrl)
    print(f"    done in {time.time()-t0:.1f}s")
    print("  Integrating EDDY-NULL...")
    t0 = time.time()
    phis_null = integrate(F2, F_override_mask=patch_mask, seed=0)
    rl_null = local_r_2d(phis_null)
    print(f"    done in {time.time()-t0:.1f}s")

    delta_r_map = rl_null - rl_ctrl
    print(f"  CTRL   <r_loc> in patch: {rl_ctrl[patch_mask].mean():.3f}")
    print(f"  NULL   <r_loc> in patch: {rl_null[patch_mask].mean():.3f}")
    print(f"  delta  <r_loc> in patch: "
          f"{delta_r_map[patch_mask].mean():+.3f}")

    # Bootstrap over seeds
    print(f"\n  Bootstrap over {N_SEED} seeds for patch mean delta_r ...")
    deltas = np.empty(N_SEED, dtype=float)
    for s in range(N_SEED):
        phi_c = integrate(F2, F_override_mask=None, seed=10 + s)
        phi_n = integrate(F2, F_override_mask=patch_mask, seed=10 + s)
        rc = local_r_2d(phi_c)
        rn = local_r_2d(phi_n)
        deltas[s] = float((rn[patch_mask] - rc[patch_mask]).mean())
        print(f"    seed {10+s:3d}: delta = {deltas[s]:+.3f}")

    d_mean = float(deltas.mean())
    d_std = float(deltas.std(ddof=1))
    d_se = d_std / np.sqrt(N_SEED)
    ci95 = (d_mean - 1.96 * d_se, d_mean + 1.96 * d_se)
    t_stat = d_mean / d_se
    print(f"\n  Bootstrap mean delta_r in patch = {d_mean:+.3f} "
          f"+/- {d_se:.3f} (95% CI {ci95[0]:+.3f} ... {ci95[1]:+.3f})")
    print(f"  one-sample t against zero: t = {t_stat:.2f}, n = {N_SEED}")
    print(f"  Causal verdict: "
          f"{'EKE causally suppresses coherence' if t_stat > 2 else 'inconclusive'}")

    # Persist
    summary = dict(
        n_patch_cells=n_patch, n_total_cells=int(NY * NX),
        patch_quantile=PATCH_QUARTILE,
        patch_threshold_F2=patch_thresh,
        control_mean_r_in_patch=float(rl_ctrl[patch_mask].mean()),
        eddynull_mean_r_in_patch=float(rl_null[patch_mask].mean()),
        delta_r_seed0=float(delta_r_map[patch_mask].mean()),
        bootstrap_n_seeds=N_SEED,
        delta_r_mean=d_mean, delta_r_sd=d_std, delta_r_se=d_se,
        delta_r_ci95=list(ci95), t_stat_vs_zero=float(t_stat),
        causal_verdict=("EKE causally suppresses coherence"
                         if t_stat > 2 else "inconclusive"),
        seeds_all=deltas.tolist(),
    )
    with open(OUT_DIR / "causal.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'causal.json'}")

    # Figure: F2, r_ctrl, r_null, delta, plus bootstrap distribution
    fig = plt.figure(figsize=(13.0, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax_f2 = fig.add_subplot(gs[0, 0])
    im0 = ax_f2.imshow(F2, cmap="inferno", origin="lower",
                        interpolation="bilinear")
    plt.colorbar(im0, ax=ax_f2, label="$F^2$ (EKE proxy)")
    ax_f2.contour(patch_mask.astype(float), levels=[0.5], colors="white",
                   linewidths=1.0)
    ax_f2.set_title("a. EKE field with patch outlined", fontsize=11)

    ax_c = fig.add_subplot(gs[0, 1])
    vmin = min(rl_ctrl.min(), rl_null.min())
    vmax = max(rl_ctrl.max(), rl_null.max())
    im1 = ax_c.imshow(rl_ctrl, cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
    plt.colorbar(im1, ax=ax_c, label=r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax_c.contour(patch_mask.astype(float), levels=[0.5], colors="white",
                  linewidths=1.0)
    ax_c.set_title("b. CONTROL coherence", fontsize=11)

    ax_n = fig.add_subplot(gs[0, 2])
    im2 = ax_n.imshow(rl_null, cmap="viridis", origin="lower",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
    plt.colorbar(im2, ax=ax_n, label=r"$\langle r_{\mathrm{loc}}\rangle_t$")
    ax_n.contour(patch_mask.astype(float), levels=[0.5], colors="white",
                  linewidths=1.0)
    ax_n.set_title("c. EDDY-NULL coherence", fontsize=11)

    ax_d = fig.add_subplot(gs[0, 3])
    dmax = float(np.nanmax(np.abs(delta_r_map)))
    im3 = ax_d.imshow(delta_r_map, cmap="RdBu_r", origin="lower",
                       vmin=-dmax, vmax=dmax, interpolation="bilinear")
    plt.colorbar(im3, ax=ax_d, label=r"$\Delta r_{\mathrm{loc}}$ (NULL$-$CTRL)")
    ax_d.contour(patch_mask.astype(float), levels=[0.5], colors="black",
                  linewidths=1.0)
    ax_d.set_title("d. Coherence response", fontsize=11)

    ax_bs = fig.add_subplot(gs[1, :2])
    ax_bs.hist(deltas, bins=12, color="C2", edgecolor="black", alpha=0.7)
    ax_bs.axvline(0, color="k", linestyle="--", lw=1)
    ax_bs.axvline(d_mean, color="C3", lw=2, label=f"mean = {d_mean:+.3f}")
    ax_bs.axvspan(ci95[0], ci95[1], color="C3", alpha=0.15,
                   label=f"95% CI ({ci95[0]:+.3f}, {ci95[1]:+.3f})")
    ax_bs.set_xlabel(r"in-patch $\overline{\Delta r_{\mathrm{loc}}}$ per seed")
    ax_bs.set_ylabel("ensemble count")
    ax_bs.set_title(
        f"e. Bootstrap over {N_SEED} seeds: "
        f"$t = {t_stat:.1f}$ vs zero",
        fontsize=11,
    )
    ax_bs.legend(loc="upper left", fontsize=9)
    ax_bs.grid(alpha=0.3)

    ax_box = fig.add_subplot(gs[1, 2:])
    r_lo_ctrl = rl_ctrl[~patch_mask]
    r_hi_ctrl = rl_ctrl[patch_mask]
    r_hi_null = rl_null[patch_mask]
    data = [r_lo_ctrl, r_hi_ctrl, r_hi_null]
    bp = ax_box.boxplot(data, labels=["low-EKE\n(CTRL)",
                                       "high-EKE\n(CTRL)",
                                       "high-EKE\n(NULL)"],
                         patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], ["#bbbbbb", "#d6604d", "#92c5de"]):
        patch.set_facecolor(c)
    ax_box.set_ylabel(r"cell-wise $\langle r_{\mathrm{loc}}\rangle_t$")
    ax_box.set_title(
        "f. Coherence recovers in the patch when noise is removed",
        fontsize=11,
    )
    ax_box.grid(alpha=0.3, axis="y")

    fig.savefig(MANUSCRIPT_FIGS / "fig_toy_kuramoto_causal.pdf")
    plt.close(fig)
    print(f"Wrote {MANUSCRIPT_FIGS / 'fig_toy_kuramoto_causal.pdf'}")


if __name__ == "__main__":
    main()
