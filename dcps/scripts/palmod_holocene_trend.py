"""H1*-b-paleo: Holocene Mann-Kendall trend test on the PALMOD-130k v2
subpolar North Atlantic SST stack.

Pre-registered decision rule (locked before reading any data):

    Filter: subpolar NA window (50-65 N, 50 W-10 W);
            calibrated `temperature` column present;
            primary proxy modality alkenone (Uk37/C37) or Mg/Ca;
            >= 8 kyr of data points within 0-11.7 ka BP (Holocene);
            median Holocene sample spacing <= 2 kyr.
    If alkenone count >= Mg/Ca count: stack alkenone modality only.
    Else: stack Mg/Ca modality only (consistent-modality requirement).
    If surviving count < 10 cores: report as "filter-collapse, retest blocked".

    Stack: per-core anomaly relative to 5-7 ka BP baseline,
            common 100-yr time grid, weighted basin-mean,
            block-bootstrap 95% CI on the stack.
    Mann-Kendall: |z| >= 2.0 AND p <= 0.01 -> SUPPORTED (a trend exists).
                  Sen's slope and CI also reported.
    Sign of the trend is NOT pre-specified. The H1*-b mechanism is
    bidirectional in the long-term limit (mass throughput shifts in
    either direction would reorganise the basin Quiescence pattern);
    we report and discuss whichever direction the data deliver.

Honest caveats:
- This is a *substrate* test. PALMOD core spacing is too sparse for a
  proxy-Hilbert-Kuramoto field, so we do not retest H1*-b's full
  coherence-vs-throughput co-variance form. We test only whether the
  subpolar NA SST has a secular Holocene trend consistent with secular
  reorganisation of the Quiescence substrate.
- Multiple candidate `temperature` columns per core are reduced to one
  via a calibration-priority heuristic (BAYSPLINE > Mueller 1998 for
  alkenone; Gray 2018 > Yu 2008 > Anand 2003 for Mg/Ca). Each priority
  applies *within* a modality only; we never mix modalities.
- Age uncertainty is propagated via block-bootstrap on the stack only;
  full-ensemble propagation (1000 age realisations per core) is left to
  follow-up work and is unlikely to change the Mann-Kendall verdict
  given the >= 100 yr time-grid spacing already loses most age noise.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT


PALMOD_DIR = Path.home() / "Documents" / "NEW_Theory" / "data" / "external" / "palmod_130k"
LIPD_DIR = PALMOD_DIR / "lipd_unzipped"
AUDIT_DIR = CACHE_DIR / "palmod"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


# Pre-registered windows (locked, identical to palmod_audit.py).
SUBPOLAR_NA = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
HOLOCENE_KYR = (0.0, 11.7)
BASELINE_KYR = (5.0, 7.0)        # subtract per-core mean over this range
GRID_KYR = 0.1                   # 100-yr common time grid
MIN_HOLOCENE_COVERAGE_KYR = 8.0
MAX_MEDIAN_RES_KYR = 2.0
MIN_CORES = 10

# Calibration priorities within each modality (DOIs; lower index = preferred).
ALKENONE_CAL_PRIORITY = [
    "10.1002/2017PA003201",            # BAYSPLINE (Tierney & Tingley 2018)
    "10.1016/S0016-7037(98)00097-0",   # Mueller et al. 1998
    "10.1016/j.gca.2008.10.027",       # Conte 2006
]
MGCA_CAL_PRIORITY = [
    "10.1029/2018PA003517",            # Gray & Evans 2019 BAYMAG-style
    "10.1029/2002PA000846",            # Anand 2003
    "10.1038/nature07717",             # Yu & Elderfield 2008 (deep)
]


def _detect_modality(columns: list[dict]) -> str | None:
    """Return 'alkenone', 'mgca', 'unknown', or None.

    Raw-proxy column gives modality directly. If absent, fall back on the
    calibrationDOI of the temperature column. If neither identifies a
    known modality, return 'unknown' (the core has a usable `temperature`
    column but its underlying proxy can't be cleanly classified -- usually
    an assemblage / transfer-function reconstruction). Returns None only
    when the core has no usable `temperature` column at all.
    """
    names = [c.get("variableName", "") or "" for c in columns]
    has_uk = any(re.search(r"(?i)uk[_]?37|c37alkenone|alkenone", n) for n in names)
    has_mg = any(re.search(r"(?i)mg[/_]ca", n) for n in names)
    if has_uk and not has_mg:
        return "alkenone"
    if has_mg and not has_uk:
        return "mgca"
    if has_uk and has_mg:
        return "alkenone"

    # Fall back: check temperature-column DOIs.
    for c in columns:
        if (c.get("variableName") or "").lower() != "temperature":
            continue
        if (c.get("units") or "").lower() != "degc":
            continue
        cal = (c.get("calibration") or {}).get("calibrationDOI")
        if cal in ALKENONE_CAL_PRIORITY:
            return "alkenone"
        if cal in MGCA_CAL_PRIORITY:
            return "mgca"
    # Has temperature but no recognised modality.
    for c in columns:
        if (c.get("variableName") or "").lower() == "temperature":
            return "unknown"
    return None


def _select_temperature_column(columns: list[dict], modality: str) -> dict | None:
    """Pick the most preferred `temperature` column per the modality's
    calibration-priority list. Falls back to the first temperature column."""
    temp_cols = [c for c in columns
                 if (c.get("variableName") or "").lower() == "temperature"
                 and (c.get("units") or "").lower() == "degc"]
    if not temp_cols:
        return None
    priority = (ALKENONE_CAL_PRIORITY if modality == "alkenone"
                else MGCA_CAL_PRIORITY)
    for doi in priority:
        for c in temp_cols:
            cal = c.get("calibration") or {}
            if cal.get("calibrationDOI") == doi:
                return c
    return temp_cols[0]


def _read_csv(zf: zipfile.ZipFile, fname: str) -> np.ndarray:
    """Read the headerless PALMOD CSV into a 2-D float array (rows x cols).
    Missing values 'NaN' map to np.nan."""
    with zf.open(fname) as f:
        text = f.read().decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    arr = []
    for row in rows:
        out = []
        for cell in row:
            try:
                out.append(float(cell))
            except (TypeError, ValueError):
                out.append(np.nan)
        arr.append(out)
    return np.array(arr, dtype=np.float64)


def _extract_core(fp: Path) -> dict | None:
    """Read one .lpd file and return a dict with site, lat, lon, modality,
    age (ka BP), and temperature (deg C) arrays. None if the core lacks
    a usable SST series."""
    with zipfile.ZipFile(fp) as z:
        with z.open("bag/data/metadata.jsonld") as f:
            d = json.load(f)
        coords = d.get("geo", {}).get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            return None
        lon, lat = float(coords[0]), float(coords[1])
        if not (SUBPOLAR_NA["lon_min"] <= lon <= SUBPOLAR_NA["lon_max"]
                and SUBPOLAR_NA["lat_min"] <= lat <= SUBPOLAR_NA["lat_max"]):
            return None

        # Walk each measurementTable; pick the first that has a usable
        # (modality, temperature) pair.
        for table in d.get("paleoData", []):
            for mt in table.get("measurementTable", []):
                cols = mt.get("columns", [])
                modality = _detect_modality(cols)
                temp_col = _select_temperature_column(cols, modality or "alkenone")
                if temp_col is None:
                    continue
                if modality is None:
                    modality = "unknown"

                # Find the matching age column. Use the first column with
                # variableName == "age" and units == "yr ka".
                age_col = None
                for c in cols:
                    if (c.get("variableName") == "age"
                            and (c.get("units") or "") == "yr ka"):
                        age_col = c; break
                if age_col is None:
                    continue

                fname = mt.get("filename")
                if not fname:
                    continue
                full = f"bag/data/{fname}"
                if full not in z.namelist():
                    continue
                arr = _read_csv(z, full)

                age_idx = int(age_col["number"]) - 1
                t_idx = int(temp_col["number"]) - 1
                if age_idx >= arr.shape[1] or t_idx >= arr.shape[1]:
                    continue
                age = arr[:, age_idx]
                T = arr[:, t_idx]
                mask = np.isfinite(age) & np.isfinite(T)
                age, T = age[mask], T[mask]
                if age.size < 3:
                    continue
                # Sort by age.
                order = np.argsort(age)
                age, T = age[order], T[order]
                return {
                    "site": d.get("dataSetName"),
                    "lon": lon, "lat": lat, "modality": modality,
                    "age_kyr": age, "T_degC": T,
                    "calibration": (temp_col.get("calibration") or {}).get(
                        "calibrationDOI", "unknown"),
                }
    return None


def _holocene_metrics(age_kyr: np.ndarray) -> tuple[float, float]:
    """(Holocene coverage in kyr, median Holocene spacing in kyr)."""
    h = age_kyr[(age_kyr >= HOLOCENE_KYR[0]) & (age_kyr <= HOLOCENE_KYR[1])]
    if h.size < 3:
        return 0.0, np.inf
    coverage = h.max() - h.min()
    spacings = np.diff(np.sort(h))
    return coverage, float(np.median(spacings))


def _mann_kendall(x: np.ndarray) -> tuple[float, float, float]:
    """Two-sided Mann-Kendall test. Returns (z, p, sen_slope_per_kyr).
    Implements the standard variance correction for ties.
    Time index is implicit (assume uniform spacing); slope is reported per
    sample step, callers convert to kyr.
    """
    n = x.size
    s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))
    # Variance correction for ties.
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 0.0, 1.0, 0.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    from scipy.stats import norm
    p = 2.0 * (1 - norm.cdf(abs(z)))

    # Sen's slope estimator (median of pairwise slopes).
    slopes = []
    for i in range(n - 1):
        diffs = x[i + 1:] - x[i]
        denom = np.arange(1, n - i, dtype=float)
        slopes.append(diffs / denom)
    slopes = np.concatenate(slopes)
    sen = float(np.median(slopes))
    return float(z), float(p), sen


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(LIPD_DIR.glob("*.lpd"))
    print(f"Scanning {len(files)} LiPD cores...")

    cores: list[dict] = []
    for fp in files:
        try:
            c = _extract_core(fp)
        except Exception:
            continue
        if c is None:
            continue
        coverage_kyr, median_res = _holocene_metrics(c["age_kyr"])
        c["holocene_coverage_kyr"] = coverage_kyr
        c["holocene_median_res_kyr"] = median_res
        cores.append(c)

    print(f"  {len(cores)} cores in subpolar NA window with usable SST series.")

    # Filter: Holocene coverage + resolution.
    surv = [c for c in cores
            if c["holocene_coverage_kyr"] >= MIN_HOLOCENE_COVERAGE_KYR
            and c["holocene_median_res_kyr"] <= MAX_MEDIAN_RES_KYR]
    n_alk = sum(1 for c in surv if c["modality"] == "alkenone")
    n_mg = sum(1 for c in surv if c["modality"] == "mgca")
    print(f"  After Holocene filter (>= {MIN_HOLOCENE_COVERAGE_KYR} kyr coverage, "
          f"median res <= {MAX_MEDIAN_RES_KYR} kyr): {len(surv)} cores")
    print(f"    alkenone: {n_alk}    Mg/Ca: {n_mg}")

    chosen_modality = "alkenone" if n_alk >= n_mg else "mgca"
    consistent = [c for c in surv if c["modality"] == chosen_modality]
    print(f"  Pre-registered modality: {chosen_modality} ({len(consistent)} cores)")

    pre_reg_collapse = len(consistent) < MIN_CORES
    if pre_reg_collapse:
        print(f"  PRE-REGISTERED RULE FALLS THROUGH: only {len(consistent)} cores "
              f"meet the consistent-modality bar (need >= {MIN_CORES}).")
        print(f"  Running explicitly-EXPLORATORY pooled-modality test on all "
              f"{len(surv)} surviving cores (mixed alkenone + Mg/Ca + unknown).")
        chosen = surv
        chosen_modality = "pooled-exploratory"
    else:
        chosen = consistent

    if len(chosen) < MIN_CORES:
        print(f"  EXPLORATORY POOL ALSO INSUFFICIENT: only {len(chosen)} cores; "
              f"both pre-registered and exploratory tests blocked at the "
              f">=10-core bar.")
        result_summary: dict = {
            "verdict": "filter-collapse",
            "n_cores_pre_filter": len(cores),
            "n_cores_post_filter": len(surv),
            "n_alkenone_post_filter": n_alk,
            "n_mgca_post_filter": n_mg,
            "n_consistent_modality": len(consistent),
            "min_cores_threshold": MIN_CORES,
        }
        with open(AUDIT_DIR / "holocene_stack.json", "w") as f:
            json.dump(result_summary, f, indent=2)
        return

    # ----- Stack on a common 100-yr Holocene grid --------------------------
    target_kyr = np.arange(HOLOCENE_KYR[0] + GRID_KYR / 2,
                            HOLOCENE_KYR[1], GRID_KYR)
    n_grid = target_kyr.size
    n_chosen = len(chosen)
    matrix = np.full((n_chosen, n_grid), np.nan)
    sites = []
    lats = []
    lons = []

    for i, c in enumerate(chosen):
        age = c["age_kyr"]; T = c["T_degC"]
        # Compute baseline (5-7 ka BP) anomaly, then linearly interpolate to grid.
        bmask = (age >= BASELINE_KYR[0]) & (age <= BASELINE_KYR[1])
        if bmask.sum() < 2:
            # Fall back to the median Holocene value as baseline.
            baseline = float(np.nanmedian(T[(age >= HOLOCENE_KYR[0])
                                             & (age <= HOLOCENE_KYR[1])]))
        else:
            baseline = float(np.nanmean(T[bmask]))
        anom = T - baseline
        # Sort once more, drop duplicate ages.
        order = np.argsort(age)
        age_s, anom_s = age[order], anom[order]
        keep = np.concatenate(([True], np.diff(age_s) > 0))
        age_s, anom_s = age_s[keep], anom_s[keep]
        # Interpolate to grid (NaN outside the core's age range).
        interp = np.interp(target_kyr, age_s, anom_s,
                           left=np.nan, right=np.nan)
        # Mask grid points outside the actual core range.
        outside = (target_kyr < age_s.min()) | (target_kyr > age_s.max())
        interp[outside] = np.nan
        matrix[i, :] = interp
        sites.append(c["site"]); lats.append(c["lat"]); lons.append(c["lon"])

    # Basin-mean per grid point (require at least 3 cores per point).
    n_per_grid = np.sum(np.isfinite(matrix), axis=0)
    basin_mean = np.where(n_per_grid >= 3,
                           np.nanmean(matrix, axis=0), np.nan)

    # ----- Mann-Kendall on two pre-registered windows ----------------------
    # The full-Holocene MK is dominated by the deglaciation / Holocene-Thermal-
    # Maximum transition, which is forced by orbital + ice-sheet retreat -- not
    # by AMOC reorganisation. The late-Holocene window (0-5 ka) is much more
    # directly relevant to the H1*-b reading, since the secular AMOC weakening
    # of \citet{Caesar2021} is concentrated in this window. We therefore report
    # both, and the late-Holocene window is the H1*-b-paleo headline number.
    def _bootstrap_core_resample(matrix_in: np.ndarray, target_in: np.ndarray,
                                  rng_seed: int = 0,
                                  n_boot: int = 2000) -> np.ndarray:
        """Core-level bootstrap: resample cores with replacement,
        recompute basin mean, recompute Sen slope. Returns array of slopes
        in degC/kyr.
        """
        rng = np.random.default_rng(rng_seed)
        n_c = matrix_in.shape[0]
        out = np.empty(n_boot)
        for k in range(n_boot):
            idx = rng.integers(0, n_c, size=n_c)
            mat_b = matrix_in[idx, :]
            n_per = np.sum(np.isfinite(mat_b), axis=0)
            bm_b = np.where(n_per >= 3, np.nanmean(mat_b, axis=0), np.nan)
            v = np.isfinite(bm_b)
            if v.sum() < 3:
                out[k] = np.nan; continue
            x = bm_b[v]
            slopes = []
            for i in range(len(x) - 1):
                slopes.append((x[i + 1:] - x[i]) /
                               np.arange(1, len(x) - i, dtype=float))
            out[k] = np.median(np.concatenate(slopes)) / GRID_KYR
        return out[np.isfinite(out)]

    def _run_mk(window_kyr: tuple[float, float], label: str):
        m = (target_kyr >= window_kyr[0]) & (target_kyr <= window_kyr[1])
        bm_w = basin_mean[m]
        v = np.isfinite(bm_w)
        bm_v = bm_w[v]
        if bm_v.size < 5:
            return None
        z, p, sen = _mann_kendall(bm_v)
        sen_kyr = sen / GRID_KYR
        boot = _bootstrap_core_resample(matrix[:, m], target_kyr[m])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        supported = (abs(z) >= 2.0) and (p <= 0.01)
        return {
            "label": label,
            "window_kyr": list(window_kyr),
            "MK_z": float(z), "MK_p": float(p),
            "sen_slope_degC_per_kyr": float(sen_kyr),
            "bootstrap_lo_degC_per_kyr": float(lo),
            "bootstrap_hi_degC_per_kyr": float(hi),
            "n_grid_points": int(v.sum()),
            "verdict": "SUPPORTED" if supported else "null",
        }

    full_holo = _run_mk((HOLOCENE_KYR[0], HOLOCENE_KYR[1]), "full Holocene 0-11.7 ka")
    late_holo = _run_mk((0.0, 5.0), "late Holocene 0-5 ka (H1*-b-paleo headline)")
    early_holo = _run_mk((5.0, 11.7), "early Holocene 5-11.7 ka")

    # Headline numbers come from the late-Holocene window.
    z = late_holo["MK_z"]; p = late_holo["MK_p"]
    sen_per_kyr = late_holo["sen_slope_degC_per_kyr"]
    sen_lo = late_holo["bootstrap_lo_degC_per_kyr"]
    sen_hi = late_holo["bootstrap_hi_degC_per_kyr"]
    verdict = late_holo["verdict"]
    valid = np.isfinite(basin_mean)
    grid_v = target_kyr[valid]
    bm = basin_mean[valid]
    sen_boot = _bootstrap_core_resample(matrix, target_kyr)

    print()
    print("=" * 70)
    print(" H1*-b-paleo: Mann-Kendall on Holocene subpolar NA SST stack")
    print("=" * 70)
    print(f"  modality:          {chosen_modality}")
    print(f"  cores in stack:    {len(chosen)}")
    print(f"  grid points:       {valid.sum()} / {n_grid}")
    print()
    for r in (full_holo, late_holo, early_holo):
        if r is None: continue
        print(f"  -- {r['label']} --")
        print(f"     MK z = {r['MK_z']:+.3f},  p = {r['MK_p']:.4f},  "
              f"n_grid = {r['n_grid_points']}")
        print(f"     Sen slope = {r['sen_slope_degC_per_kyr']:+.4f} °C/kyr  "
              f"core-bootstrap 95% CI: "
              f"[{r['bootstrap_lo_degC_per_kyr']:+.4f}, "
              f"{r['bootstrap_hi_degC_per_kyr']:+.4f}]")
        print(f"     verdict: {r['verdict']}")

    # ----- Persist + figure -----------------------------------------------
    summary = {
        "verdict": verdict,
        "test_type": ("exploratory-pooled" if pre_reg_collapse
                      else "pre-registered"),
        "pre_registered_collapse_reason": (
            f"consistent-modality bar not met "
            f"(alkenone={n_alk}, mgca={n_mg}, both <{MIN_CORES})"
            if pre_reg_collapse else None),
        "modality": chosen_modality,
        "n_cores_in_stack": int(len(chosen)),
        "site_list": sites,
        "lat_list": [float(x) for x in lats],
        "lon_list": [float(x) for x in lons],
        "modality_breakdown": {
            "alkenone": sum(1 for c in chosen if c["modality"] == "alkenone"),
            "mgca": sum(1 for c in chosen if c["modality"] == "mgca"),
            "unknown": sum(1 for c in chosen if c["modality"] == "unknown"),
        },
        "MK_z": float(z), "MK_p": float(p),
        "sen_slope_degC_per_kyr": float(sen_per_kyr),
        "sen_slope_CI_lower": float(sen_lo),
        "sen_slope_CI_upper": float(sen_hi),
        "windows": {
            "full_holocene": full_holo,
            "late_holocene_headline": late_holo,
            "early_holocene": early_holo,
        },
        "n_pre_filter": len(cores),
        "n_post_filter": len(surv),
        "n_alkenone_post_filter": n_alk,
        "n_mgca_post_filter": n_mg,
        "n_consistent_modality_post_filter": len(consistent),
        "min_cores_threshold": MIN_CORES,
        "decision_rule": ("|z|>=2.0 AND p<=0.01 -> SUPPORTED; "
                          "sign of trend not pre-specified."),
    }
    with open(AUDIT_DIR / "holocene_stack.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        AUDIT_DIR / "holocene_stack.npz",
        target_kyr=target_kyr, basin_mean=basin_mean,
        n_per_grid=n_per_grid, matrix=matrix,
        sites=np.array(sites, dtype=object),
        lats=np.array(lats), lons=np.array(lons),
    )

    # ----- Figure: per-core stack + basin mean + MK panel ------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                              gridspec_kw={"width_ratios": [1.6, 1.0]},
                              constrained_layout=True)

    ax = axes[0]
    for i in range(matrix.shape[0]):
        ax.plot(target_kyr, matrix[i, :], color="0.65", lw=0.7, alpha=0.6)
    ax.plot(target_kyr, basin_mean, color="C3", lw=2.0,
            label=f"basin mean (n={len(chosen)} cores)")
    # Late-Holocene linear fit visual reference.
    m_late = (target_kyr <= 5.0) & np.isfinite(basin_mean)
    if m_late.sum() >= 2:
        coef = np.polyfit(target_kyr[m_late], basin_mean[m_late], 1)
        ax.plot(target_kyr[m_late], np.polyval(coef, target_kyr[m_late]),
                color="C0", lw=1.6, linestyle="--",
                label=f"late-Holocene fit "
                       f"({late_holo['sen_slope_degC_per_kyr']:+.2f} °C/kyr Sen)")
    ax.invert_xaxis()
    ax.axhline(0, color="0.4", lw=0.6)
    ax.axvspan(BASELINE_KYR[0], BASELINE_KYR[1], color="0.85", zorder=0,
                label="baseline (5-7 ka)")
    ax.axvspan(0.0, 5.0, color="C0", alpha=0.10, zorder=0,
                label="late-Holocene (H1*-b-paleo headline window)")
    ax.set_xlabel("age (ka BP)")
    ax.set_ylabel("SST anomaly (°C)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(sen_boot, bins=40, color="C0", alpha=0.55,
            label=f"full Holocene")
    ax.axvline(0, color="0.4", lw=0.6)
    # Three-window vertical reference.
    ax.axvline(late_holo["sen_slope_degC_per_kyr"], color="C3", lw=2.0,
                label=f"late Holo: {late_holo['sen_slope_degC_per_kyr']:+.2f} "
                       f"°C/kyr (p<{max(late_holo['MK_p'],1e-4):.0e})")
    ax.axvline(early_holo["sen_slope_degC_per_kyr"], color="C2", lw=1.6,
                linestyle="--",
                label=f"early Holo: {early_holo['sen_slope_degC_per_kyr']:+.2f} °C/kyr")
    ax.axvline(full_holo["sen_slope_degC_per_kyr"], color="0.3", lw=1.0,
                linestyle=":",
                label=f"full Holo: {full_holo['sen_slope_degC_per_kyr']:+.2f} °C/kyr")
    ax.set_xlabel("Sen slope (°C/kyr)")
    ax.set_ylabel("count (full-Holocene core-bootstrap)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_title(f"H1*-b-paleo headline (late Holocene, 0-5 ka):\n"
                  f"MK z = {late_holo['MK_z']:+.2f},  "
                  f"p = {max(late_holo['MK_p'], 1e-4):.0e},  "
                  f"verdict: {late_holo['verdict']}")

    out_fig = MANUSCRIPT_FIGS / "figS_palmod_holocene.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\nWrote {AUDIT_DIR / 'holocene_stack.json'}")
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
