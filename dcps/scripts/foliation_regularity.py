"""Foliation-regularity diagnostic for the Quiescence relationship.

Inspired by Brin & Stuck (2002) Sec. 6.2 (absolute continuity of stable
and unstable foliations of an Anosov diffeomorphism). The Quiescence
hypothesis posits that the basin's bandpassed phase dynamics are foliated
by mean geostrophic flow |grad SSH|, with phase coherence varying smoothly
along the foliation.

Operationally: within each spatial super-tile, regress <r_loc>(x) on
|grad SSH|(x) and compute the Hessian condition number kappa of the
ordinary-least-squares design matrix X^T X (X = [1, |grad SSH|]). A small
kappa indicates a well-conditioned local fit (foliation is regular); a
large kappa flags tiles where the linear ansatz breaks down (e.g.
near-constant |grad SSH| across the tile, or strongly non-linear local
relationship between coherence and mean flow).

This is reported as an *exploratory* diagnostic, NOT a pre-registered
falsification test. The output is per-basin distribution statistics of
kappa(x) and a 95th-percentile flag.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT


CACHE_WINDING = CACHE_DIR / "winding"
CACHE_FOLIATION = CACHE_DIR / "foliation"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


SUPER_TILE_DEG = 10.0
MIN_CELLS_PER_TILE = 12


def _condition_number(X: np.ndarray) -> float:
    """OLS Hessian condition number of design matrix X (rows = obs, cols = coefs)."""
    H = X.T @ X
    w = np.linalg.eigvalsh(H)
    if w.min() <= 0:
        return float("inf")
    return float(w.max() / w.min())


def kappa_field(rl_mean: np.ndarray, grad_mag: np.ndarray,
                lat: np.ndarray, rlon: np.ndarray,
                tile_deg: float = SUPER_TILE_DEG) -> tuple[np.ndarray, dict]:
    """Cell-wise condition number of the local OLS Hessian.

    For each cell, take all valid neighbours within a tile_deg x tile_deg
    box and compute kappa(X) where X = [1, |grad SSH|]. Cells whose tile
    contains < MIN_CELLS_PER_TILE valid samples become NaN.
    """
    n_lat, n_rlon = rl_mean.shape
    out = np.full_like(rl_mean, np.nan, dtype=np.float32)
    half = 0.5 * tile_deg
    for i, la in enumerate(lat):
        for j, lo in enumerate(rlon):
            ii = np.where(np.abs(lat - la) <= half)[0]
            jj = np.where(np.abs(rlon - lo) <= half)[0]
            blk_r = rl_mean[ii[:, None], jj[None, :]].ravel()
            blk_g = grad_mag[ii[:, None], jj[None, :]].ravel()
            mask = np.isfinite(blk_r) & np.isfinite(blk_g)
            if mask.sum() < MIN_CELLS_PER_TILE:
                continue
            x = blk_g[mask]
            X = np.column_stack([np.ones_like(x), x])
            out[i, j] = _condition_number(X)

    finite = out[np.isfinite(out)]
    stats = {
        "n_tiles_with_kappa": int(finite.size),
        "kappa_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "kappa_median": float(np.median(finite)) if finite.size else float("nan"),
        "kappa_p95": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "kappa_max": float(np.max(finite)) if finite.size else float("nan"),
    }
    return out, stats


def main():
    CACHE_FOLIATION.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    for ax, basin in zip(axs, ("atlantic", "pacific", "southern")):
        cache_file = CACHE_WINDING / f"{basin}.npz"
        z = np.load(cache_file)
        rl, gm = z["rl_mean"], z["grad_SSH"]
        lat, rlon = z["lat"], z["rlon"]
        kap, stats = kappa_field(rl, gm, lat, rlon)
        summary[basin] = stats
        np.savez_compressed(CACHE_FOLIATION / f"{basin}.npz",
                             kappa=kap, lat=lat, rlon=rlon, **stats)

        finite = kap[np.isfinite(kap)]
        log_kap = np.log10(finite)
        ax.hist(log_kap, bins=40, color="C0", alpha=0.85)
        ax.axvline(np.log10(stats["kappa_p95"]),
                    color="C3", linestyle="--",
                    label=fr"95th pctile: $\kappa = {stats['kappa_p95']:.1e}$")
        ax.set_title(f"{basin} ({stats['n_tiles_with_kappa']} tiles)")
        ax.set_xlabel(r"$\log_{10}\kappa$")
        ax.legend(fontsize=8)

    axs[0].set_ylabel("count")
    out_fig = MANUSCRIPT_FIGS / "figS_foliation.pdf"
    fig.savefig(out_fig)
    plt.close(fig)

    print("=" * 70)
    print(" Foliation-regularity diagnostic summary (exploratory)")
    print("=" * 70)
    for b, s in summary.items():
        print(f"  {b:<10}  median kappa = {s['kappa_median']:.2e}, "
              f"95th pctile = {s['kappa_p95']:.2e}, n_tiles = {s['n_tiles_with_kappa']}")

    out_json = CACHE_FOLIATION / "foliation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
