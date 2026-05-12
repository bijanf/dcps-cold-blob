"""Step 3 / B3: Compute Quiescence Index Q from CMIP6 historical
tos for whatever models the Pangeo fetch returned.

The original spec was: high-res group (EC-Earth3P-HR, HadGEM3-GC31-HM,
CNRM-CM6-1-HR) vs low-res group (CanESM5, CNRM-CM6-1, IPSL-CM6A-LR),
Mann-Whitney U on Q.  Pangeo coverage of HighResMIP is incomplete;
the fetch returned only the two low-res models (CanESM5, CNRM-CM6-1).
We compute Q for each and report the gap honestly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import pearsonr


REPO = Path("/home/bijanf/Documents/NEW_Theory")
CACHE = REPO / "dcps/cache/highresmip"
OUT_DIR = REPO / "quiescence_toolkit" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_Q_from_tos_annual(tos_da):
    """Compute Q = -rho(<r_loc>, |grad SST_climatology|) from annual
    tos.  Uses 2-yr to 5-yr bandpass on the annual cadence (fs = 1/yr,
    Nyquist = 0.5/yr, so cutoffs must give Wn < 1.0).
    """
    arr = tos_da.values.astype(np.float64)
    n_t = arr.shape[0]
    if n_t < 30:
        return None
    # Climatology + detrend
    arr_anom = arr - np.nanmean(arr, axis=0, keepdims=True)
    t = np.arange(n_t).astype(float)
    flat = arr_anom.reshape(n_t, -1)
    for j in range(flat.shape[1]):
        if np.isfinite(flat[:, j]).all():
            coef = np.polyfit(t, flat[:, j], 1)
            flat[:, j] -= np.polyval(coef, t)
    arr_anom = flat.reshape(arr.shape)
    # Bandpass 4-10 yr on annual fs=1/yr; nyq=0.5; need Wn strictly < 1
    fs = 1.0; nyq = 0.5
    b, a = butter(4, [1.0 / 10.0 / nyq, 1.0 / 4.0 / nyq], btype="band")
    flat = arr_anom.reshape(n_t, -1)
    valid = np.isfinite(flat).all(axis=0)
    bp = np.full_like(flat, np.nan)
    if valid.any():
        bp[:, valid] = filtfilt(b, a, flat[:, valid], axis=0).astype(arr.dtype)
    arr_bp = bp.reshape(arr.shape)
    # Hilbert phase
    phi = np.full(flat.shape, np.nan, dtype=np.float64)
    if valid.any():
        z = scipy_hilbert(bp[:, valid], axis=0)
        phi[:, valid] = np.angle(z)
    phi = phi.reshape(arr.shape)
    # Time-mean local r_loc on 500 km windows
    # The tos arrays are on irregular CMIP6 grids; treat as flat with
    # nominal 1-deg resolution.  We use a simple 5-cell box average
    # in lat-lon for the local window (~500 km at midlatitudes).
    z_e = np.exp(1j * phi)
    z_e[~np.isfinite(arr)] = 0
    # Average over time
    z_mean = np.nanmean(z_e, axis=0)
    # Spatial window box of side 5 cells (~500 km at midlatitudes)
    from scipy.signal import convolve2d
    kernel = np.ones((5, 5)) / 25.0
    z_re = convolve2d(np.real(z_mean), kernel, mode="same", boundary="symm")
    z_im = convolve2d(np.imag(z_mean), kernel, mode="same", boundary="symm")
    rl_mean = np.sqrt(z_re ** 2 + z_im ** 2)
    # |grad SST_climatology|
    sst_mean = np.nanmean(arr, axis=0)
    dlat = np.gradient(sst_mean, axis=0)
    dlon = np.gradient(sst_mean, axis=1)
    grad_mag = np.sqrt(dlat ** 2 + dlon ** 2)
    # Pearson correlation
    a_flat = rl_mean.ravel()
    b_flat = grad_mag.ravel()
    m = np.isfinite(a_flat) & np.isfinite(b_flat)
    if m.sum() < 30:
        return None
    rho, p = pearsonr(a_flat[m], b_flat[m])
    return dict(rho=float(rho), p=float(p), n_cells=int(m.sum()),
                Q=float(-rho))


def main():
    inventory_file = CACHE / "highresmip_inventory.json"
    if not inventory_file.exists():
        print(f"No inventory at {inventory_file}; fetch first.")
        return
    inv = json.loads(inventory_file.read_text())

    Q_table = {"high_res": {}, "low_res": {}}
    for group, models in inv.items():
        print(f"\n--- {group} ---")
        for source_id, info in models.items():
            fpath = Path(info["file"])
            if not fpath.exists():
                print(f"  {source_id}: file missing")
                continue
            try:
                ds = xr.open_dataset(fpath)
                tos = ds["tos"]
                result = compute_Q_from_tos_annual(tos)
                if result is None:
                    print(f"  {source_id}: too few years to compute Q")
                    continue
                Q_table[group][source_id] = result
                print(f"  {source_id}: Q = {result['Q']:+.3f}  "
                      f"(rho = {result['rho']:+.3f}, p = {result['p']:.2e}, "
                      f"n = {result['n_cells']})")
            except Exception as e:
                print(f"  {source_id}: FAILED -- {e}")

    # Mann-Whitney U if we have both groups
    hi_Q = [r["Q"] for r in Q_table["high_res"].values()]
    lo_Q = [r["Q"] for r in Q_table["low_res"].values()]
    print()
    if hi_Q and lo_Q:
        from scipy.stats import mannwhitneyu
        u_stat, u_p = mannwhitneyu(hi_Q, lo_Q, alternative="less")
        print(f"  Mann-Whitney U (high_Q < low_Q): U = {u_stat:.2f}, p = {u_p:.3f}")
    else:
        print(f"  Mann-Whitney U test: NOT RUNNABLE")
        print(f"    high_res: {len(hi_Q)} model(s) -- {list(Q_table['high_res'].keys())}")
        print(f"    low_res:  {len(lo_Q)} model(s) -- {list(Q_table['low_res'].keys())}")
        print(f"  HighResMIP coverage on Pangeo is incomplete; the")
        print(f"  resolution-dependence test cannot be run as specified.")
        print(f"  P3 (resolution dependence) remains pending.")

    summary = dict(
        Q_per_model=Q_table,
        n_high_res=len(hi_Q),
        n_low_res=len(lo_Q),
        mann_whitney_runnable=bool(hi_Q and lo_Q),
        note=("HighResMIP hist-1950 experiment_id is not yet on Pangeo "
              "for the requested high-res models (EC-Earth3P-HR, "
              "HadGEM3-GC31-HM, CNRM-CM6-1-HR). The test reports Q "
              "for the available low-res models and acknowledges the "
              "high-res gap."),
    )
    with open(OUT_DIR / "cmip6_quiescence.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cmip6_quiescence.json'}")


if __name__ == "__main__":
    main()
