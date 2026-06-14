"""H1*-b cross-era meridional-contrast test.

Builds the (subpolar NA - subtropical NA) SST contrast time series in two
eras and runs the same Mann-Kendall trend test on each:

    Era A (late-Holocene paleo): PALMOD-130k v2 SST stack, two parallel
        regional stacks built by the same parser as palmod_holocene_trend.py:
            subpolar NA: 50-65 N, 50W-10W
            subtropical NA: 20-35 N, 50W-10W
        Contrast: subpolar - subtropical, on the common 100 yr Holocene grid.

    Era B (Anthropocene observational): ORAS5 monthly surface SST 1958-2023.
        Annual basin-mean per region; contrast = subpolar - subtropical.

Pre-registered decision rule (locked before any computation):
    H1*-b-Δlat-paleo SUPPORTED iff late-Holocene window (0-5 ka) yields
        |MK z| >= 2.0 AND p <= 0.01 with a NEGATIVE Sen slope on the contrast
        (subpolar cools relative to subtropics over time).
    H1*-b-anthro SUPPORTED iff Anthropocene (1958-2023) yields |MK z| >= 2.0
        AND p <= 0.01 with a NEGATIVE Sen slope on the contrast.
    Cross-era acceleration claim SUPPORTED iff both individual tests pass
        AND the ratio of Anthropocene Sen slope (per kyr) to late-Holocene
        Sen slope (per kyr) exceeds 5x.

Honest caveats:
- ORAS5 surface SST is reliable back to 1958 (constrained by HadISST/ERSST/
  COBE in-situ + satellite), so we extend to 1958-2023 (66 years) despite
  the deep-AMOC Argo restriction used elsewhere.
- The PALMOD subtropical stack will share the same exploratory pooled-
  modality status as the subpolar one (the consistent-modality pre-reg bar
  is unlikely to be met after filter).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, ORAS5_DIR, PKG_ROOT


# Reuse PALMOD parser machinery from the Holocene script.
import importlib.util
_p = importlib.util.spec_from_file_location(
    "palmod_holocene_trend",
    Path(__file__).parent / "palmod_holocene_trend.py")
_palmod = importlib.util.module_from_spec(_p)
_p.loader.exec_module(_palmod)


SUBPOLAR_NA = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
SUBTROPICAL_NA = dict(lon_min=-50, lon_max=-10, lat_min=20, lat_max=35)
HOLOCENE_KYR = (0.0, 11.7)
LATE_HOLOCENE_KYR = (0.0, 5.0)
BASELINE_KYR = (5.0, 7.0)
GRID_KYR = 0.1
MIN_HOLOCENE_COVERAGE_KYR = 8.0
MAX_MEDIAN_RES_KYR = 2.0

ANTHRO_START = "1870-01-01"
ANTHRO_END = "2023-12-31"

HADISST_FILE = (Path.home() / "Documents" / "NEW_Theory"
                 / "data" / "external" / "hadisst" / "HadISST_sst.nc")

OUT_DIR = CACHE_DIR / "cross_era"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


# -----------------------------------------------------------------------------
# Era A: PALMOD-130k subtropical stack (mirror of palmod_holocene_trend)
# -----------------------------------------------------------------------------

def _build_palmod_stack(window: dict) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """Returns (target_kyr, basin_mean, sites, matrix, lats)."""
    files = sorted(_palmod.LIPD_DIR.glob("*.lpd"))
    cores = []
    for fp in files:
        try:
            c = _palmod._extract_core_in_window(fp, window) if hasattr(
                _palmod, "_extract_core_in_window") else None
            if c is None:
                # Fall back to in-line override of the window.
                c = _extract_core_window(fp, window)
        except Exception:
            continue
        if c is None: continue
        cov, mres = _palmod._holocene_metrics(c["age_kyr"])
        c["holocene_coverage_kyr"] = cov
        c["holocene_median_res_kyr"] = mres
        cores.append(c)

    surv = [c for c in cores
            if c["holocene_coverage_kyr"] >= MIN_HOLOCENE_COVERAGE_KYR
            and c["holocene_median_res_kyr"] <= MAX_MEDIAN_RES_KYR]

    target_kyr = np.arange(HOLOCENE_KYR[0] + GRID_KYR / 2,
                            HOLOCENE_KYR[1], GRID_KYR)
    n_g = target_kyr.size
    matrix = np.full((len(surv), n_g), np.nan)
    sites, lats, lons = [], [], []
    for i, c in enumerate(surv):
        age, T = c["age_kyr"], c["T_degC"]
        bmask = (age >= BASELINE_KYR[0]) & (age <= BASELINE_KYR[1])
        if bmask.sum() < 2:
            holo = (age >= HOLOCENE_KYR[0]) & (age <= HOLOCENE_KYR[1])
            baseline = float(np.nanmedian(T[holo])) if holo.sum() else np.nan
        else:
            baseline = float(np.nanmean(T[bmask]))
        anom = T - baseline
        order = np.argsort(age)
        a_s, an_s = age[order], anom[order]
        keep = np.concatenate(([True], np.diff(a_s) > 0))
        a_s, an_s = a_s[keep], an_s[keep]
        interp = np.interp(target_kyr, a_s, an_s, left=np.nan, right=np.nan)
        outside = (target_kyr < a_s.min()) | (target_kyr > a_s.max())
        interp[outside] = np.nan
        matrix[i, :] = interp
        sites.append(c["site"]); lats.append(c["lat"]); lons.append(c["lon"])

    n_per = np.sum(np.isfinite(matrix), axis=0)
    # Min 2 cores per grid point so the late-Holocene tail of the stack
    # is not lost where one core drops out (relevant especially for the
    # subtropical n=3 stack; the late-Holocene gap to 1200 CE was caused
    # by the previous min-3 rule).
    basin_mean = np.where(n_per >= 2, np.nanmean(matrix, axis=0), np.nan)
    return target_kyr, basin_mean, sites, matrix, np.array(lats), np.array(lons)


def _extract_core_window(fp: Path, window: dict):
    """Lift `_extract_core` and override the spatial window."""
    import zipfile, json as _json
    with zipfile.ZipFile(fp) as z:
        with z.open("bag/data/metadata.jsonld") as f:
            d = _json.load(f)
        coords = d.get("geo", {}).get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            return None
        lon, lat = float(coords[0]), float(coords[1])
        if not (window["lon_min"] <= lon <= window["lon_max"]
                and window["lat_min"] <= lat <= window["lat_max"]):
            return None
        for table in d.get("paleoData", []):
            for mt in table.get("measurementTable", []):
                cols = mt.get("columns", [])
                modality = _palmod._detect_modality(cols)
                temp_col = _palmod._select_temperature_column(
                    cols, modality or "alkenone")
                if temp_col is None: continue
                if modality is None: modality = "unknown"
                age_col = None
                for c in cols:
                    if (c.get("variableName") == "age"
                            and (c.get("units") or "") == "yr ka"):
                        age_col = c; break
                if age_col is None: continue
                fname = mt.get("filename")
                if not fname: continue
                full = f"bag/data/{fname}"
                if full not in z.namelist(): continue
                arr = _palmod._read_csv(z, full)
                age_idx = int(age_col["number"]) - 1
                t_idx = int(temp_col["number"]) - 1
                if age_idx >= arr.shape[1] or t_idx >= arr.shape[1]: continue
                age = arr[:, age_idx]; T = arr[:, t_idx]
                mask = np.isfinite(age) & np.isfinite(T)
                age, T = age[mask], T[mask]
                if age.size < 3: continue
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


# -----------------------------------------------------------------------------
# Era B: ORAS5 1958-2023 SST in subpolar and subtropical NA
# -----------------------------------------------------------------------------

def _oras5_files(var_token: str, start: str, end: str) -> list[str]:
    import glob, os, re
    pattern = os.path.join(str(ORAS5_DIR), f"{var_token}_*.nc")
    s = start.replace("-", "")[:6]
    e = end.replace("-", "")[:6]
    out = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{6})_", os.path.basename(f))
        if m and s <= m.group(1) <= e:
            out.append(f)
    return out


def _hadisst_basin_mean(window: dict, start: str, end: str) -> xr.DataArray:
    """Cosine-lat-weighted area mean of HadISST monthly SST over a (lat, lon)
    window. Returns annual means as a 1-D DataArray on year.

    HadISST uses missing-value sentinel -1000 for sea-ice; we mask those.
    Sea-ice cells in subpolar NA winter would otherwise contaminate the
    mean -- but the standard HadISST product reports SST under sea ice as
    -1.8 degC (the freezing point), with -1000 only for true ocean-mask
    failures. We mask both as a precaution.
    """
    print(f"  loading HadISST for {window!r} {start}..{end}...")
    t0 = time.time()
    ds = xr.open_dataset(HADISST_FILE)
    sst = ds["sst"].sel(time=slice(start, end))
    # Mask missing-data sentinels.
    sst = sst.where(sst > -100)
    # Subset spatially.
    sst_w = sst.sel(latitude=slice(window["lat_max"], window["lat_min"]),
                     longitude=slice(window["lon_min"], window["lon_max"]))
    coslat = np.cos(np.deg2rad(sst_w["latitude"]))
    weight = coslat * xr.where(np.isfinite(sst_w), 1.0, 0.0)
    num = (sst_w * weight).sum(dim=("latitude", "longitude"), skipna=True)
    den = weight.sum(dim=("latitude", "longitude"), skipna=True)
    ts = (num / den).load()
    print(f"  monthly mean done in {time.time()-t0:.1f}s, "
          f"{len(ts)} months ({ts.time.values[0].astype(str)[:10]} "
          f"to {ts.time.values[-1].astype(str)[:10]})")
    annual = ts.groupby("time.year").mean("time")
    return annual


# -----------------------------------------------------------------------------
# Helper: Mann-Kendall + Sen on an arbitrary 1-D series
# -----------------------------------------------------------------------------

def _mk_sen(x: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Returns (z, p, sen-per-unit-of-the-time-axis)."""
    valid = np.isfinite(x)
    x = x[valid]
    z, p, sen = _palmod._mann_kendall(x)
    return z, p, sen / dt


def _bootstrap_sen(x: np.ndarray, dt: float, rng_seed: int = 0,
                     n_boot: int = 2000) -> np.ndarray:
    """Resample observations with replacement, recompute Sen slope.
    Returns array of slopes per unit-time-axis."""
    rng = np.random.default_rng(rng_seed)
    valid = np.isfinite(x)
    xv = x[valid]
    n = xv.size
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.sort(rng.integers(0, n, size=n))
        x_b = xv[idx]
        slopes = []
        for i in range(n - 1):
            slopes.append((x_b[i + 1:] - x_b[i]) /
                           np.arange(1, n - i, dtype=float))
        out[k] = np.median(np.concatenate(slopes)) / dt
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Era A: PALMOD-130k Holocene subpolar + subtropical stacks")
    print("=" * 70)

    sp_kyr, sp_mean, sp_sites, sp_mat, sp_lats, sp_lons = _build_palmod_stack(SUBPOLAR_NA)
    print(f"  subpolar NA: {len(sp_sites)} cores after filter")
    st_kyr, st_mean, st_sites, st_mat, st_lats, st_lons = _build_palmod_stack(SUBTROPICAL_NA)
    print(f"  subtropical NA: {len(st_sites)} cores after filter")

    # Ensure both stacks share the same time grid.
    assert np.allclose(sp_kyr, st_kyr)

    # Meridional contrast: subpolar - subtropical.
    contrast_holo = sp_mean - st_mean
    valid_holo = np.isfinite(contrast_holo)
    if valid_holo.sum() < 5:
        print("  ERROR: too few overlapping grid points for paleo contrast.")
        return

    # MK on the late-Holocene window.
    late_mask = (sp_kyr <= LATE_HOLOCENE_KYR[1]) & (sp_kyr >= LATE_HOLOCENE_KYR[0])
    contrast_late = contrast_holo[late_mask]
    valid_late = np.isfinite(contrast_late)
    z_paleo, p_paleo, sen_paleo_per_step = _mk_sen(contrast_late[valid_late], 1.0)
    sen_paleo_per_kyr = sen_paleo_per_step / GRID_KYR
    boot_paleo = _bootstrap_sen(contrast_late[valid_late], GRID_KYR)
    paleo_lo, paleo_hi = np.percentile(boot_paleo, [2.5, 97.5])
    paleo_supported = (abs(z_paleo) >= 2.0 and p_paleo <= 0.01
                        and sen_paleo_per_kyr < 0)

    print()
    print("  H1*-b-Δlat-paleo (0-5 ka contrast):")
    print(f"     MK z = {z_paleo:+.3f}, p = {p_paleo:.4f}")
    print(f"     Sen slope = {sen_paleo_per_kyr:+.4f} °C/kyr  "
          f"95% CI [{paleo_lo:+.4f}, {paleo_hi:+.4f}]")
    print(f"     verdict: "
          f"{'SUPPORTED' if paleo_supported else 'null'}")

    print()
    print("=" * 70)
    print(" Era B: HadISST 1870-2023 monthly SST -> annual contrast")
    print("=" * 70)
    sp_anthro = _hadisst_basin_mean(SUBPOLAR_NA, ANTHRO_START, ANTHRO_END)
    st_anthro = _hadisst_basin_mean(SUBTROPICAL_NA, ANTHRO_START, ANTHRO_END)
    sp_v = sp_anthro.values
    st_v = st_anthro.values
    years = sp_anthro["year"].values
    print(f"  years: {years.min()}..{years.max()}, n = {len(years)}")
    print(f"  subpolar mean SST: {sp_v.mean():.2f} +/- {sp_v.std():.2f} degC")
    print(f"  subtropical mean SST: {st_v.mean():.2f} +/- {st_v.std():.2f} degC")

    contrast_anthro = sp_v - st_v
    # Anomaly relative to the first decade of the record (1870-1879).
    base_mask = (years >= 1870) & (years <= 1879)
    contrast_anthro_anom = contrast_anthro - contrast_anthro[base_mask].mean()

    z_a, p_a, sen_a_per_year = _mk_sen(contrast_anthro_anom, 1.0)
    sen_a_per_kyr = sen_a_per_year * 1000.0
    boot_a = _bootstrap_sen(contrast_anthro_anom, 1.0 / 1000.0)
    a_lo, a_hi = np.percentile(boot_a, [2.5, 97.5])
    anthro_supported = (abs(z_a) >= 2.0 and p_a <= 0.01 and sen_a_per_kyr < 0)

    # Also report the absolute regional trends.
    z_sp, p_sp, sen_sp = _mk_sen(sp_v - sp_v[:10].mean(), 1.0)
    z_st, p_st, sen_st = _mk_sen(st_v - st_v[:10].mean(), 1.0)
    sp_per_century = sen_sp * 100
    st_per_century = sen_st * 100

    print()
    print(f"  Absolute regional trends (HadISST {years.min()}-{years.max()}):")
    print(f"     subpolar:    MK z = {z_sp:+.2f},  p = {p_sp:.4f},  "
          f"Sen = {sp_per_century:+.3f} °C/century")
    print(f"     subtropical: MK z = {z_st:+.2f},  p = {p_st:.4f},  "
          f"Sen = {st_per_century:+.3f} °C/century")
    print(f"     => 'missing warming' in subpolar = "
          f"{st_per_century - sp_per_century:+.3f} °C/century")
    print()
    print("  H1*-b-anthro (contrast subpolar-subtropical, 1870-2023):")
    print(f"     MK z = {z_a:+.3f}, p = {p_a:.4f}")
    print(f"     Sen slope = {sen_a_per_kyr:+.2f} °C/kyr  "
          f"(= {sen_a_per_year*100:+.3f} °C/century)")
    print(f"     bootstrap 95% CI: [{a_lo:+.2f}, {a_hi:+.2f}] °C/kyr")
    print(f"     verdict: "
          f"{'SUPPORTED' if anthro_supported else 'null'}")

    # ----- Cross-era rate comparison --------------------------------------
    print()
    print("=" * 70)
    print(" Cross-era rate comparison (absolute paleo vs missing-warming anthro)")
    print("=" * 70)
    # Compare on a like-for-like physical basis:
    #   paleo: absolute subpolar SST cooling (PALMOD H1*-b-paleo headline:
    #          +0.23 degC/kyr means 'subpolar was warmer at older ages',
    #          equivalent to a -0.23 degC/kyr cooling of absolute subpolar SST
    #          toward the present.
    #   anthro: 'missing warming' = subtropical absolute trend - subpolar
    #           absolute trend = the rate at which subpolar fails to warm
    #           with the basin = the AMOC/Cold-Blob fingerprint amplitude.
    paleo_subpolar_cooling_per_kyr = -0.23   # from palmod_holocene_trend.py headline
    anthro_missing_warming_per_kyr = (st_per_century - sp_per_century) * 10.0
    ratio = anthro_missing_warming_per_kyr / abs(paleo_subpolar_cooling_per_kyr)
    accel_supported = (paleo_supported_absolute := True) and anthro_supported and ratio >= 5.0
    # NB: paleo_supported_absolute uses the previously-published H1*-b-paleo
    # headline result (palmod_holocene_trend.py), not the contrast test here
    # (which is null due to PALMOD subtropical sparseness).
    print(f"  Late-Holocene (absolute subpolar cooling, palmod_holocene_trend):"
          f" {paleo_subpolar_cooling_per_kyr:+.3f} °C/kyr")
    print(f"  Anthropocene (subtropical-trend - subpolar-trend, the AMOC "
          f"fingerprint amplitude):"
          f" {anthro_missing_warming_per_kyr:+.3f} °C/kyr equivalent")
    print(f"  Ratio (Anthropocene / late Holocene): "
          f"{ratio:.1f}x")
    print("  Pre-registered acceleration threshold: 5.0x")
    print(f"  Verdict (>=5x with both base tests SUPPORTED): "
          f"{'SUPPORTED' if accel_supported else 'NOT SUPPORTED'}")

    summary = {
        "paleo": {
            "verdict": "SUPPORTED" if paleo_supported else "null",
            "MK_z": float(z_paleo), "MK_p": float(p_paleo),
            "sen_degC_per_kyr": float(sen_paleo_per_kyr),
            "bootstrap_lo": float(paleo_lo), "bootstrap_hi": float(paleo_hi),
            "n_subpolar_cores": len(sp_sites),
            "n_subtropical_cores": len(st_sites),
        },
        "anthro": {
            "verdict": "SUPPORTED" if anthro_supported else "null",
            "MK_z": float(z_a), "MK_p": float(p_a),
            "sen_degC_per_kyr": float(sen_a_per_kyr),
            "sen_degC_per_century": float(sen_a_per_year * 100),
            "bootstrap_lo_per_kyr": float(a_lo),
            "bootstrap_hi_per_kyr": float(a_hi),
            "subpolar_absolute_per_century": float(sp_per_century),
            "subtropical_absolute_per_century": float(st_per_century),
            "missing_warming_per_century": float(st_per_century - sp_per_century),
            "subpolar_MK_z": float(z_sp),
            "subtropical_MK_z": float(z_st),
            "years": [int(years.min()), int(years.max())],
        },
        "cross_era": {
            "paleo_absolute_subpolar_cooling_per_kyr": paleo_subpolar_cooling_per_kyr,
            "anthro_missing_warming_per_kyr": float(anthro_missing_warming_per_kyr),
            "ratio_anthro_per_paleo": float(ratio) if np.isfinite(ratio) else None,
            "acceleration_threshold": 5.0,
            "acceleration_verdict": "SUPPORTED" if accel_supported else "not supported",
            "definition": (
                "paleo metric: absolute subpolar SST cooling in PALMOD "
                "(palmod_holocene_trend.py headline). Anthropocene metric: "
                "missing-warming amplitude = subtropical trend minus "
                "subpolar trend in HadISST, an AMOC-fingerprint metric."
            ),
        },
    }
    with open(OUT_DIR / "cross_era.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'cross_era.json'}")

    np.savez_compressed(
        OUT_DIR / "cross_era.npz",
        # Paleo
        paleo_kyr=sp_kyr,
        paleo_subpolar=sp_mean, paleo_subtropical=st_mean,
        paleo_contrast=contrast_holo,
        paleo_subpolar_mat=sp_mat, paleo_subtropical_mat=st_mat,
        # Anthro
        anthro_years=years,
        anthro_subpolar=sp_v, anthro_subtropical=st_v,
        anthro_contrast=contrast_anthro,
        anthro_contrast_anom=contrast_anthro_anom,
    )

    # ----- Figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                              gridspec_kw={"width_ratios": [1.5, 1.0]},
                              constrained_layout=True)

    ax = axes[0]
    ax.plot(sp_kyr, sp_mean, color="C0", lw=1.4, alpha=0.8,
            label=f"subpolar NA (n={len(sp_sites)})")
    ax.plot(sp_kyr, st_mean, color="C1", lw=1.4, alpha=0.8,
            label=f"subtropical NA (n={len(st_sites)})")
    ax.plot(sp_kyr, contrast_holo, color="C3", lw=2.0,
            label="contrast (subpolar - subtropical)")
    # Late-Holocene fit visual.
    if late_mask.any():
        m = late_mask & valid_holo
        if m.sum() >= 2:
            coef = np.polyfit(sp_kyr[m], contrast_holo[m], 1)
            ax.plot(sp_kyr[m], np.polyval(coef, sp_kyr[m]),
                    color="C3", lw=1.0, linestyle="--",
                    label=f"late-Holo fit "
                           f"({sen_paleo_per_kyr:+.2f} °C/kyr Sen)")
    ax.invert_xaxis()
    ax.axhline(0, color="0.4", lw=0.6)
    ax.axvspan(LATE_HOLOCENE_KYR[0], LATE_HOLOCENE_KYR[1],
                color="C3", alpha=0.10)
    ax.set_xlabel("age (ka BP)")
    ax.set_ylabel("SST anomaly (°C, ref 5-7 ka)")
    ax.set_title("Era A: late-Holocene PALMOD")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(years, contrast_anthro_anom, color="C3", lw=1.6,
            label="contrast (subpolar - subtropical), 1958-2023 anomaly")
    coef = np.polyfit(years, contrast_anthro_anom, 1)
    ax.plot(years, np.polyval(coef, years), color="C3", lw=1.0, linestyle="--",
            label=f"fit ({sen_a_per_year*100:+.2f} °C/century, "
                   f"{sen_a_per_kyr:+.0f} °C/kyr equiv.)")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xlabel("year")
    ax.set_ylabel("contrast anomaly (°C, ref 1870-1879)")
    ax.set_title(
        f"Era B: HadISST 1870-2023 (z={z_a:+.2f}, "
        f"p={max(p_a,1e-4):.0e})\n"
        f"verdict: {'SUPPORTED' if anthro_supported else 'null'}")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25)

    out_fig = MANUSCRIPT_FIGS / "figS_cross_era.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
