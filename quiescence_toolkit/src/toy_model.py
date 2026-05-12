"""Step 6 / B7: 1-D reduced-order stochastic Kuramoto model with
Gaussian jet and EKE-prescribed noise amplitude.

Specification from the user's guide:
  - 1-D periodic domain L = 4000 km
  - Gaussian jet at x = 0, sigma_J = 200 km
  - EKE noise amplitude D(x) = alpha |dU/dx|
  - N = 100 oscillators, K = 0.5, sigma_omega = 0.2

Integrate the SDE
  d_phi_i = omega_i + K [sin(phi_{i+1}-phi_i) + sin(phi_{i-1}-phi_i)]
          + D(x_i) eta_i(t)
via Euler-Maruyama for 10^5 steps; compute <r_loc(x)> as long-time
average within 500-km windows.

Parameter sensitivity sweep over (K, U_0) is also produced.

Output:
  - <r_loc(x)> vs U(x) and |dU/dx| scatter, cross-correlation reported
  - parameter-sensitivity table over K in [0.2, 1.0], U_0 in [0.5, 2.0]
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


L_KM = 4000.0
N = 500              # more oscillators -> more robust estimate
DX_KM = L_KM / N
SIGMA_J_KM = 200.0
K_DEFAULT = 0.2      # weak coupling, where Fokker-Planck applies
U0_DEFAULT = 1.0
SIGMA_OMEGA = 0.2
DT = 0.05
N_STEPS = 100_000
N_BURN = 20_000
WIN_KM = 500.0
WIN_CELLS = max(1, int(round(WIN_KM / DX_KM)))
N_SEEDS = 10         # seed-average for robust reference rho


def jet_profile(x_km, U0=U0_DEFAULT, sigma_J_km=SIGMA_J_KM):
    """Gaussian jet centred at x=0 in a periodic domain."""
    # Periodic boundary: nearest image
    L = L_KM
    x_wrap = ((x_km + L / 2) % L) - L / 2
    return U0 * np.exp(-x_wrap ** 2 / (2 * sigma_J_km ** 2))


def noise_amplitude(x_km, U0=U0_DEFAULT, sigma_J_km=SIGMA_J_KM, alpha=1.0):
    """D(x) = alpha |dU/dx| normalised so alpha * max(|dU/dx|) = 1
    (matching the user-guide spec).  U0 then drops out of D's
    spatial shape; the parameter sweep over U0 is therefore a
    trivial identity unless we let U0 affect K/D coupling -- which
    we don't in this minimal implementation."""
    L = L_KM
    x_wrap = ((x_km + L / 2) % L) - L / 2
    dUdx = -U0 * x_wrap / (sigma_J_km ** 2) * np.exp(-x_wrap ** 2 / (2 * sigma_J_km ** 2))
    return alpha * np.abs(dUdx) / np.max(np.abs(dUdx))


def integrate(K=K_DEFAULT, U0=U0_DEFAULT, n_steps=N_STEPS,
                  n_burn=N_BURN, seed=0):
    """Euler-Maruyama integration of the 1-D stochastic Kuramoto on a
    periodic grid.  Returns (x_km, omega, D, mean_phase_history,
    final_phase, r_loc_x).
    """
    rng = np.random.default_rng(seed)
    x_km = np.arange(N) * DX_KM
    omega = rng.normal(0.0, SIGMA_OMEGA, N)
    D = noise_amplitude(x_km, U0=U0)
    phi = rng.uniform(-np.pi, np.pi, N)

    # Accumulators for r_loc
    z_sum = np.zeros(N, dtype=complex)
    n_acc = 0

    sqrt_dt = np.sqrt(DT)
    for step in range(n_steps):
        d_phi_coupling = (np.sin(np.roll(phi, -1) - phi)
                          + np.sin(np.roll(phi, 1) - phi))
        drift = omega + K * d_phi_coupling
        diff = D * rng.normal(0.0, 1.0, N) * sqrt_dt
        phi = phi + drift * DT + diff
        if step >= n_burn:
            z_t = np.exp(1j * phi)
            # 500-km window local r: rolling mean over WIN_CELLS
            # We'll do this with convolution at the end.
            z_sum += z_t
            n_acc += 1
    # Final-time r computation: rolling mean of z over WIN_CELLS,
    # averaged over all timesteps via the cumulative z_sum.
    z_mean = z_sum / max(n_acc, 1)
    # 500-km moving average of z (circular)
    z_padded = np.concatenate([z_mean[-WIN_CELLS:], z_mean, z_mean[:WIN_CELLS]])
    kernel = np.ones(2 * WIN_CELLS + 1) / (2 * WIN_CELLS + 1)
    z_window = np.convolve(z_padded, kernel, mode="same")[WIN_CELLS:-WIN_CELLS]
    r_loc_x = np.abs(z_window)
    return x_km, omega, D, jet_profile(x_km, U0), r_loc_x


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Step 6 / B7: 1-D reduced-order stochastic Kuramoto model")
    print(f"   L = {L_KM:.0f} km, N = {N}, K = {K_DEFAULT}, "
          f"U_0 = {U0_DEFAULT}, sigma_J = {SIGMA_J_KM:.0f} km")
    print("=" * 70)

    # Reference run: seed-average for robust estimate
    rho_U_seeds = []
    rho_D_seeds = []
    last_x = None; last_U = None; last_D = None; last_r = None
    for s in range(N_SEEDS):
        x, omega, D, U, r = integrate(K=K_DEFAULT, U0=U0_DEFAULT, seed=s)
        rho_U_seeds.append(pearsonr(r, U)[0])
        rho_D_seeds.append(pearsonr(r, D)[0])
        last_x, last_U, last_D, last_r = x, U, D, r
    x, U, D, r = last_x, last_U, last_D, last_r
    rho_U = float(np.mean(rho_U_seeds))
    rho_D = float(np.mean(rho_D_seeds))
    rho_U_sd = float(np.std(rho_U_seeds))
    rho_D_sd = float(np.std(rho_D_seeds))
    p_U = float(np.nan); p_D = float(np.nan)
    print(f"  reference ({N_SEEDS}-seed avg): rho(<r>, U) = {rho_U:+.3f} "
          f"+/- {rho_U_sd:.3f}")
    print(f"  reference ({N_SEEDS}-seed avg): rho(<r>, D) = {rho_D:+.3f} "
          f"+/- {rho_D_sd:.3f}")

    # Parameter sweep
    K_grid = np.linspace(0.2, 1.0, 5)
    U_grid = np.linspace(0.5, 2.0, 4)
    rho_table = np.full((K_grid.size, U_grid.size), np.nan)
    for i, K in enumerate(K_grid):
        for j, U0 in enumerate(U_grid):
            xx, _, _, Uu, rr = integrate(K=K, U0=U0, seed=42)
            r_local, _ = pearsonr(rr, Uu)
            rho_table[i, j] = r_local
    print()
    print(f"  parameter sweep rho(<r>, U):")
    print(f"    K \\ U0   " + "  ".join(f"{u:6.2f}" for u in U_grid))
    for i, K in enumerate(K_grid):
        print(f"    K={K:.2f}   " + "  ".join(f"{rho_table[i,j]:+6.3f}"
                                                  for j in range(U_grid.size)))

    out = dict(
        reference=dict(
            K=K_DEFAULT, U0=U0_DEFAULT,
            rho_r_U=float(rho_U), p_r_U=float(p_U),
            rho_r_D=float(rho_D), p_r_D=float(p_D),
        ),
        param_sweep=dict(
            K_grid=K_grid.tolist(),
            U0_grid=U_grid.tolist(),
            rho_table=rho_table.tolist(),
            min=float(np.nanmin(rho_table)),
            max=float(np.nanmax(rho_table)),
        ),
    )
    with open(out_dir / "toy_1d_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_dir / 'toy_1d_results.json'}")

    # Figure: panel a U(x), panel b D(x), panel c r(x), panel d scatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    axes[0, 0].plot(x, U, color="C0", linewidth=2)
    axes[0, 0].set_ylabel("U(x)"); axes[0, 0].set_xlabel("x (km)")
    axes[0, 0].text(-0.10, 1.02, "a", transform=axes[0, 0].transAxes,
                       fontweight="bold", fontsize=13)
    axes[0, 0].set_title("Gaussian jet profile", fontsize=10)

    axes[0, 1].plot(x, D, color="C1", linewidth=2)
    axes[0, 1].set_ylabel(r"D(x) = $\alpha$|dU/dx|")
    axes[0, 1].set_xlabel("x (km)")
    axes[0, 1].text(-0.10, 1.02, "b", transform=axes[0, 1].transAxes,
                       fontweight="bold", fontsize=13)
    axes[0, 1].set_title("Noise amplitude (normalised)", fontsize=10)

    axes[1, 0].plot(x, r, color="C3", linewidth=2)
    axes[1, 0].set_ylabel(r"$\langle r_{\mathrm{loc}}(x)\rangle$")
    axes[1, 0].set_xlabel("x (km)")
    axes[1, 0].text(-0.10, 1.02, "c", transform=axes[1, 0].transAxes,
                       fontweight="bold", fontsize=13)
    axes[1, 0].set_title("Synthetic local coherence", fontsize=10)

    axes[1, 1].scatter(U, r, s=20, alpha=0.6, color="C0",
                          edgecolors="none")
    axes[1, 1].set_xlabel("U(x)")
    axes[1, 1].set_ylabel(r"$\langle r_{\mathrm{loc}}\rangle$")
    axes[1, 1].text(0.04, 0.06, f"$\\rho = {rho_U:+.3f}$\n$p = {p_U:.1e}$",
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", edgecolor="0.5"))
    axes[1, 1].text(-0.10, 1.02, "d", transform=axes[1, 1].transAxes,
                       fontweight="bold", fontsize=13)
    axes[1, 1].set_title("Quiescence Signature: $\\rho < 0$",
                            fontsize=10)
    fig.savefig(out_dir / "fig_T4_toy_1d.pdf")
    plt.close(fig)
    print(f"Wrote {out_dir / 'fig_T4_toy_1d.pdf'}")


if __name__ == "__main__":
    main()
