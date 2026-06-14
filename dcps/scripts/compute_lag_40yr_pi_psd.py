"""
Compute multi-model Welch power spectral densities of basin-mean Q and EKE
from the cached piControl 30-yr-window series, for SI Fig. fig_si_lag_40yr_decomposition
panel (e).

Inputs:
  dcps/cache/eke_timeseries/<MODEL>_atlantic_eke_ts.json       -- pi_eke list
  dcps/cache/holocene_exit/bulk/<MODEL>_atlantic.json          -- pi_Q list

Both sources record values from 30-yr non-overlapping windows at 30-yr stride,
so the effective sampling interval is dt = 30 yr.  This sets a Nyquist period
of 60 yr.  We therefore compute the PSD only over periods 60 -- 600 yr.  The
13--28 yr North Atlantic decadal band (Arthun 2021) and the ~24 yr SPG
eigenmode (Sevellec & Fedorov 2013) are shown as physical-reference
annotations on the figure, with a clear caption note that those processes
lie below our resolved frequency range.

Per-model PSD is normalised to unit area (so we are comparing spectral
shapes, not absolute power) and aggregated into a multi-model median + IQR
on a common log-period grid.

Output:
  dcps/cache/lag_40yr/pi_psd_atlantic.json
"""
from __future__ import annotations
import json
import glob
from pathlib import Path

import numpy as np
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[2]
EKE_DIR = REPO / "dcps" / "cache" / "eke_timeseries"
Q_DIR = REPO / "dcps" / "cache" / "holocene_exit" / "bulk"
OUT = REPO / "dcps" / "cache" / "lag_40yr" / "pi_psd_atlantic.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT_YR = 30.0  # sampling interval in years (one 30-yr-window-mean per step)
PERIOD_LO = 60.0   # Nyquist period
PERIOD_HI = 600.0  # longest period retained on the common grid
N_GRID = 40
MIN_N_EKE = 6
MIN_N_Q = 4


def _model_psd(arr, min_n):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < min_n:
        return None
    nperseg = min(n, 16)
    f, P = welch(arr - np.nanmean(arr),
                 fs=1.0 / DT_YR, nperseg=nperseg,
                 noverlap=nperseg // 2,
                 detrend="linear",
                 window="hann", scaling="spectrum")
    mask = f > 0
    return f[mask], P[mask]


def _interp_to_grid(f, P, fg):
    # area-normalise first, then log-interpolate to the common grid
    area = np.trapezoid(P, f)
    if not np.isfinite(area) or area <= 0:
        return None
    Pn = P / area
    logf = np.log(f)
    logP = np.log(Pn + 1e-30)
    return np.exp(np.interp(np.log(fg), logf, logP))


def main():
    period_grid = np.geomspace(PERIOD_LO, PERIOD_HI, N_GRID)
    freq_grid = 1.0 / period_grid

    out = {
        "dt_yr": DT_YR,
        "period_grid_yr": period_grid.tolist(),
        "freq_grid_cy_per_yr": freq_grid.tolist(),
        "min_n_eke": MIN_N_EKE,
        "min_n_q": MIN_N_Q,
        "eke": {"models": [], "psd_norm": []},
        "q":   {"models": [], "psd_norm": []},
    }

    # --- EKE ---
    for path in sorted(EKE_DIR.glob("*_atlantic_eke_ts.json")):
        d = json.loads(path.read_text())
        r = _model_psd(d.get("pi_eke", []), MIN_N_EKE)
        if r is None:
            continue
        f, P = r
        P_on_grid = _interp_to_grid(f, P, freq_grid)
        if P_on_grid is None:
            continue
        out["eke"]["models"].append(d.get("model", path.stem))
        out["eke"]["psd_norm"].append(P_on_grid.tolist())

    # --- Q ---
    for path in sorted(Q_DIR.glob("*_atlantic.json")):
        name = path.stem
        if "ssp" in name or "pacific" in name:
            continue
        d = json.loads(path.read_text())
        r = _model_psd(d.get("pi_Q", []), MIN_N_Q)
        if r is None:
            continue
        f, P = r
        P_on_grid = _interp_to_grid(f, P, freq_grid)
        if P_on_grid is None:
            continue
        out["q"]["models"].append(d.get("model", name))
        out["q"]["psd_norm"].append(P_on_grid.tolist())

    eke_arr = np.array(out["eke"]["psd_norm"])
    q_arr = np.array(out["q"]["psd_norm"])
    out["eke"]["n_models"] = int(eke_arr.shape[0])
    out["q"]["n_models"] = int(q_arr.shape[0])
    out["eke"]["median"] = np.median(eke_arr, axis=0).tolist()
    out["eke"]["q25"] = np.percentile(eke_arr, 25, axis=0).tolist()
    out["eke"]["q75"] = np.percentile(eke_arr, 75, axis=0).tolist()
    out["q"]["median"] = np.median(q_arr, axis=0).tolist()
    out["q"]["q25"] = np.percentile(q_arr, 25, axis=0).tolist()
    out["q"]["q75"] = np.percentile(q_arr, 75, axis=0).tolist()

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    print(f"  EKE: {out['eke']['n_models']} models")
    print(f"  Q  : {out['q']['n_models']} models")


if __name__ == "__main__":
    main()
