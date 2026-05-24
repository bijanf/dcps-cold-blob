"""MAJOR 13 of the peer review: independent cross-check via
correlation-network node degree.

Reviewer A8: the defect-density and Hilbert local Kuramoto coherence
are computed from the same phase field; they are not independent.
The Morlet wavelet cross-check failed its pre-registered method-
independence threshold (r_s >= 0.7).

This script builds a graph-theoretic test that is independent of
both Hilbert phase and Kuramoto vocabulary:
  1. For each cell, build the 1-10 yr bandpassed SST anomaly
     time series (the same input as the phase pipeline).
  2. Compute Pearson cross-correlation r_ij between every pair of
     cells at lag 0 on the bandpassed time series.
  3. Threshold |r_ij| > 0.5 to obtain a binary adjacency matrix.
  4. Each cell's node-degree k_i = number of cells to which it is
     significantly correlated.
  5. Correlate the node-degree spatial field with the
     climatological |grad SSH| field.

Expected sign: NEGATIVE.  In the Quiescence framework, high-EKE
cells should have lower coherence with their neighbours (so lower
node-degree); low-EKE cells should have higher coherence (so higher
node-degree).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.spatial_stats import spatial_block_permutation
from dcps.nature_style import apply_nature_style
apply_nature_style()

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import (
    BASINS,
    load_oras5_basin,
    regrid_basin,
    preprocess_anomaly,
)


OUT_DIR = CACHE_DIR / "graph_network"
CORR_THRESHOLD = 0.5
B = 500


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" MAJOR 13: independent graph-network cross-check")
    print(f"   threshold |r| > {CORR_THRESHOLD}, block_km = 500, B = {B}")
    print("=" * 70)

    results = {}
    for basin in BASINS:
        print(f"\n  basin: {basin}")
        t0 = time.time()
        sst, lat2d, rlon2d = load_oras5_basin("sosstsst", basin)
        ssh, _, _ = load_oras5_basin("sossheig", basin)
        sst_rg = regrid_basin(sst, lat2d, rlon2d, basin)
        ssh_rg = regrid_basin(ssh, lat2d, rlon2d, basin)
        # Bandpass SST anomaly (same preprocessing as phase pipeline)
        sst_anom = preprocess_anomaly(sst_rg)
        # Build correlation matrix on the bandpassed series, lag 0
        arr = sst_anom.transpose("time", "lat", "rlon").values
        T, ny, nx = arr.shape
        flat = arr.reshape(T, -1)
        valid = np.isfinite(flat).all(axis=0)
        flat_valid = flat[:, valid]
        # Standardise
        flat_z = (flat_valid - flat_valid.mean(axis=0)) / flat_valid.std(axis=0)
        # Pearson cross-correlation matrix (N x N)
        corr = (flat_z.T @ flat_z) / T
        # Adjacency: |r| > threshold, exclude self
        adj = (np.abs(corr) > CORR_THRESHOLD).astype(int)
        np.fill_diagonal(adj, 0)
        node_degree = adj.sum(axis=1)
        # Map back to 2D
        node_degree_2d = np.full(flat.shape[1], np.nan)
        node_degree_2d[valid] = node_degree.astype(float)
        node_degree_2d = node_degree_2d.reshape(ny, nx)
        print(f"    correlation matrix done in {time.time()-t0:.0f}s; "
              f"mean degree = {np.nanmean(node_degree_2d):.1f}")

        # |grad SSH|
        ssh_mean = ssh_rg.mean("time")
        grad_mag = np.sqrt(ssh_mean.differentiate("lat") ** 2
                           + ssh_mean.differentiate("rlon") ** 2)

        # Get coords for spatial perm
        lat = sst_anom["lat"].values
        rlon = sst_anom["rlon"].values
        LAT, RLON = np.meshgrid(lat, rlon, indexing="ij")
        lon_offset = BASINS[basin]["lon_offset"]
        actual_lon = ((RLON + lon_offset + 180.0) % 360.0) - 180.0

        a = node_degree_2d.ravel()
        b = grad_mag.values.ravel()
        lat_flat = LAT.ravel()
        lon_flat = actual_lon.ravel()

        result = spatial_block_permutation(
            a, b, lat_flat, lon_flat, block_km=500, B=B, seed=42)
        print(f"    rho(node_degree, |grad SSH|) = "
              f"{result['rho_observed']:+.3f}  "
              f"n_cells = {result['n_cells']}  "
              f"n_blocks = {result['n_blocks']}  "
              f"p_perm = {result['p_perm']:.4f}")
        results[basin] = {
            "rho": result["rho_observed"],
            "p_perm": result["p_perm"],
            "n_cells": result["n_cells"],
            "n_blocks": result["n_blocks"],
            "mean_node_degree": float(np.nanmean(node_degree_2d)),
            "threshold": CORR_THRESHOLD,
        }

    # Compare to Hilbert-derived rho
    pub = json.loads(
        (CACHE_DIR / "multi_basin" / "multi_basin_quiescence.json").read_text())
    print()
    print("=" * 70)
    print(" Comparison: Hilbert <r_loc> vs graph-network node degree")
    print("=" * 70)
    print(f"{'basin':<10} {'Hilbert rho':>14} {'graph rho':>14} "
          f"{'agreement':>12}")
    print("-" * 56)
    for b, r in results.items():
        rho_pub = pub[b]["rho"]
        rho_graph = r["rho"]
        # Sign agreement
        agree = "yes" if np.sign(rho_pub) == np.sign(rho_graph) else "no"
        print(f"{b:<10} {rho_pub:>14.3f} {rho_graph:>14.3f} {agree:>12}")

    with open(OUT_DIR / "graph_network.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'graph_network.json'}")


if __name__ == "__main__":
    main()
