"""Multi-basin winding-number defect-density test (H1*-c-topo).

Pre-registered topological cross-check of the Quiescence Hypothesis:
across NA, NP, ACC, the Pearson correlation between time-mean phase-defect
density D(x) and the climatological |grad SSH| should be >= +0.30 with
p <= 0.01 in at least 2 of 3 basins (mirror image of the negative
correlation we found between <r_loc>(x) and |grad SSH|).

Sanity check (reported before the H1*-c-topo verdict): rho(<r_loc>, D)
should be strongly negative -- defects coincide with low coherence.

Reuses the load/regrid/preprocess/Hilbert/local_r helpers from
``multi_basin_quiescence`` so the upstream pipeline is identical to the
H1*-c run.
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.winding import plaquette_winding, defect_density_field
from dcps.nature_style import apply_nature_style
apply_nature_style()

from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
    instantaneous_phase,
    local_r_mean,
)


WINDING_DIR = CACHE_DIR / "winding"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def run_basin(basin: str) -> dict:
    print(f"\n{'='*60}\n{basin.upper()}: {BASINS[basin]['label']}\n{'='*60}")
    WINDING_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
    ssh, _, _ = load_oras5_basin("sossheig", basin)
    print(f"  loaded SST shape={tuple(sst.sizes.values())} in {time.time()-t0:.1f}s")

    sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
    ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
    sst_anom = preprocess_anomaly(sst_rg)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    phi = phi.isel(time=slice(6, n_t - 6))   # Hilbert edge trim
    print(f"  phase field {phi.sizes['time']} months x "
          f"{phi.sizes['lat']} lat x {phi.sizes['rlon']} rlon")

    # Local Kuramoto coherence (reused for sanity correlation)
    rl_mean = local_r_mean(phi, radius_km=500.0)
    ssh_mean = ssh_rg.mean("time")
    grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                        + ssh_mean.differentiate("rlon") ** 2)

    # Plaquette winding number per timestep, then defect density on the same
    # 500 km circular window used elsewhere.
    t0 = time.time()
    W = plaquette_winding(phi)
    print(f"  plaquette winding {tuple(W.sizes.values())} in {time.time()-t0:.1f}s")
    t0 = time.time()
    D = defect_density_field(W, target_lat=phi.lat.values,
                              target_lon=phi.rlon.values, radius_km=500.0)
    print(f"  defect-density field in {time.time()-t0:.1f}s")

    # ----- pre-registered correlation -------------------------------------
    a = D.values.ravel()
    b = grad_mag.values.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    rho_topo, p_topo = pearsonr(a[mask], b[mask])
    n_cells = int(mask.sum())

    # ----- sanity correlation: D vs <r_loc> -------------------------------
    a = D.values.ravel()
    c = rl_mean.values.ravel()
    mask = np.isfinite(a) & np.isfinite(c)
    rho_sanity, p_sanity = pearsonr(a[mask], c[mask])

    print(f"  rho(D, |grad SSH|)  = {rho_topo:+.3f}  p = {p_topo:.2e}  n={n_cells}")
    print(f"  rho(D, <r_loc>)     = {rho_sanity:+.3f}  p = {p_sanity:.2e}  (sanity)")

    verdict = "SUPPORTED" if (rho_topo >= 0.30 and p_topo < 0.01) else "falsified"
    print(f"  -> H1*-c-topo per-basin: {verdict}")

    np.savez_compressed(
        WINDING_DIR / f"{basin}.npz",
        D=D.values, grad_SSH=grad_mag.values, rl_mean=rl_mean.values,
        lat=D.lat.values, rlon=D.rlon.values,
        rho_topo=rho_topo, p_topo=p_topo, n_cells=n_cells,
        rho_sanity=rho_sanity, p_sanity=p_sanity,
    )

    return {
        "basin": basin,
        "rho_topo": float(rho_topo), "p_topo": float(p_topo),
        "rho_sanity": float(rho_sanity), "p_sanity": float(p_sanity),
        "n_cells": n_cells, "verdict": verdict,
        "D": D, "grad_mag": grad_mag, "rl_mean": rl_mean,
    }


def aggregate_verdict(results: dict[str, dict]) -> str:
    n_supp = sum(1 for r in results.values() if r["verdict"] == "SUPPORTED")
    return "SUPPORTED" if n_supp >= 2 else "falsified"


def main():
    results: dict[str, dict] = {}
    for b in BASINS:
        try:
            results[b] = run_basin(b)
        except Exception:
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print(" Multi-basin H1*-c-topo (winding defect-density) summary")
    print("=" * 70)
    print(f"{'basin':<12} {'rho_topo':>9} {'p_topo':>10} {'rho_sanity':>11} "
          f"{'n_cells':>8}  {'verdict'}")
    print("-" * 70)
    for b, r in results.items():
        print(f"{b:<12} {r['rho_topo']:+.3f} {r['p_topo']:.2e} "
              f"{r['rho_sanity']:+.3f} {r['n_cells']:>8}  {r['verdict']}")

    overall = aggregate_verdict(results)
    print(f"\nAggregate (>= 2/3 basins SUPPORTED): {overall}")

    out_json = WINDING_DIR / "winding_basin.json"
    with open(out_json, "w") as f:
        json.dump({
            "aggregate": overall,
            "per_basin": {b: {k: r[k] for k in
                              ("rho_topo", "p_topo", "rho_sanity",
                               "p_sanity", "n_cells", "verdict")}
                          for b, r in results.items()},
        }, f, indent=2)
    print(f"Wrote {out_json}")

    # ----- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(11, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.3])
    ax_scatter = fig.add_subplot(gs[3, :])

    def _native_tick(rlon_value: float, lon_offset: float) -> str:
        native = ((rlon_value + lon_offset + 180) % 360) - 180
        return f"{native:.0f}°E" if native >= 0 else f"{-native:.0f}°W"

    panel_letters = "abcdefg"
    for row, (b, r) in enumerate(results.items()):
        lon_offset = BASINS[b]["lon_offset"]
        rlon = r["D"].rlon.values

        ax_d = fig.add_subplot(gs[row, 0])
        im = ax_d.pcolormesh(rlon, r["D"].lat, r["D"].values,
                              cmap="magma", shading="auto")
        plt.colorbar(im, ax=ax_d, label=r"$D(x)$  [defect fraction]")
        ax_d.set_xlabel("longitude")
        ax_d.set_ylabel("latitude")
        ticks = np.linspace(rlon.min(), rlon.max(), 6)
        ax_d.set_xticks(ticks)
        ax_d.set_xticklabels([_native_tick(t, lon_offset) for t in ticks])
        ax_d.text(-0.18, 1.02, panel_letters[2 * row],
                   transform=ax_d.transAxes,
                   fontweight="bold", fontsize=11,
                   verticalalignment="bottom")

        ax_g = fig.add_subplot(gs[row, 1])
        im = ax_g.pcolormesh(rlon, r["grad_mag"].lat, r["grad_mag"].values,
                              cmap="magma", shading="auto")
        plt.colorbar(im, ax=ax_g, label=r"$|\nabla \overline{SSH}|$")
        ax_g.set_xlabel("longitude")
        ax_g.set_ylabel("latitude")
        ax_g.set_xticks(ticks)
        ax_g.set_xticklabels([_native_tick(t, lon_offset) for t in ticks])
        ax_g.text(-0.18, 1.02, panel_letters[2 * row + 1],
                   transform=ax_g.transAxes,
                   fontweight="bold", fontsize=11,
                   verticalalignment="bottom")

        a = r["D"].values.ravel()
        bb = r["grad_mag"].values.ravel()
        mask = np.isfinite(a) & np.isfinite(bb)
        ax_scatter.scatter(bb[mask], a[mask], s=4, alpha=0.30,
                            color=BASINS[b]["color"], label=BASINS[b]["label"])

    ax_scatter.set_xlabel(r"$|\nabla \overline{SSH}|$")
    ax_scatter.set_ylabel(r"$D(x)$")
    ax_scatter.legend(loc="upper left", fontsize=9, frameon=False)
    ax_scatter.grid(alpha=0.25)
    ax_scatter.text(-0.08, 1.02, "g",
                     transform=ax_scatter.transAxes,
                     fontweight="bold", fontsize=11,
                     verticalalignment="bottom")

    out_fig = MANUSCRIPT_FIGS / "fig4b_winding_defects.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
