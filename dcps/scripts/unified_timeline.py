"""Build the unified-timeline figure: subpolar NA SST anomaly from
12 ka BP to 2100 CE on a single y-axis (anomaly vs 1870-1950 baseline).

Sources:
    (i)   PALMOD-130k v2 11-core basin-mean (paleo, 0-11.7 ka BP).
    (ii)  HadISST 1870-2023 subpolar NA area-mean.
    (iii) CMIP6 historical + ssp245 + ssp585 ensemble (10 models, Pangeo).

All anomalies are referenced to a 1870-1950 pre-industrial baseline:
    - PALMOD: re-anchored so basin-mean at 0-1 ka BP = 0 (the 0-1 ka
      window straddles the 1870-1950 reference period).
    - HadISST: subtract 1870-1950 mean.
    - CMIP6: each model's anomaly = model SST minus model's own
      historical 1870-1950 mean (so SSP anomalies use the parent
      historical baseline, not their own first decade).

Time axis: year CE throughout. PALMOD ages converted via
    year_CE = 1950 - 1000 * age_ka.
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

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


CACHE_PALMOD = CACHE_DIR / "palmod"
CACHE_CROSS  = CACHE_DIR / "cross_era"
CACHE_OUT    = CACHE_DIR / "unified_timeline"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


SUBPOLAR_NA  = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)
BASELINE = (1870, 1950)

HADISST_FILE = (Path.home() / "Documents" / "NEW_Theory" / "data"
                / "external" / "hadisst" / "HadISST_sst.nc")

MODELS = [
    "ACCESS-CM2", "CanESM5", "CESM2", "CMCC-CM2-SR5", "CNRM-CM6-1",
    "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0",
    "NorESM2-MM", "UKESM1-0-LL",
]

EXPERIMENTS = ["historical", "ssp245", "ssp585"]


# ---------------------------------------------------------------------------
# Source 1: PALMOD basin-mean (re-anchored)
# ---------------------------------------------------------------------------

def palmod_unified() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (years_CE, basin_mean_anomaly_vs_1870_1950, matrix_per_core)."""
    z = np.load(CACHE_PALMOD / "holocene_stack.npz", allow_pickle=True)
    target_kyr = z["target_kyr"]
    basin_mean = z["basin_mean"].copy()         # anomaly vs 5-7 ka
    matrix = z["matrix"].copy()

    # Re-anchor: subtract basin-mean over 0-1 ka BP so that window = 0,
    # which is the rough proxy of the 1870-1950 modern baseline.
    re_anchor_mask = (target_kyr >= 0.0) & (target_kyr <= 1.0)
    shift = float(np.nanmean(basin_mean[re_anchor_mask]))
    basin_mean -= shift
    matrix -= shift

    # Convert age (ka BP) -> year CE (BP measured against 1950).
    years_ce = 1950.0 - 1000.0 * target_kyr
    return years_ce, basin_mean, matrix


# ---------------------------------------------------------------------------
# Source 2: HadISST subpolar absolute anomaly vs 1870-1950
# ---------------------------------------------------------------------------

def hadisst_unified() -> tuple[np.ndarray, np.ndarray]:
    """Return (years, anomaly_vs_1870_1950)."""
    print("Loading HadISST subpolar absolute SST...")
    t0 = time.time()
    ds = xr.open_dataset(HADISST_FILE)
    sst = ds["sst"].sel(time=slice("1870-01-01", "2023-12-31"))
    sst = sst.where(sst > -100)
    sst_w = sst.sel(latitude=slice(SUBPOLAR_NA["lat_max"], SUBPOLAR_NA["lat_min"]),
                     longitude=slice(SUBPOLAR_NA["lon_min"], SUBPOLAR_NA["lon_max"]))
    coslat = np.cos(np.deg2rad(sst_w["latitude"]))
    weight = coslat * xr.where(np.isfinite(sst_w), 1.0, 0.0)
    num = (sst_w * weight).sum(dim=("latitude", "longitude"), skipna=True)
    den = weight.sum(dim=("latitude", "longitude"), skipna=True)
    ts_monthly = (num / den).load()
    annual = ts_monthly.groupby("time.year").mean("time")
    years = annual["year"].values
    values = annual.values
    base_mask = (years >= BASELINE[0]) & (years <= BASELINE[1])
    baseline = float(values[base_mask].mean())
    print(f"  HadISST 1870-1950 baseline = {baseline:.3f} °C "
          f"({time.time()-t0:.1f}s)")
    return years, values - baseline


# ---------------------------------------------------------------------------
# Source 3: CMIP6 subpolar absolute anomaly vs each model's own 1870-1950
# ---------------------------------------------------------------------------

def _cmip6_subpolar_annual(ds: xr.Dataset) -> xr.DataArray:
    tos = ds["tos"]
    lat_name = next((n for n in ("lat", "latitude") if n in tos.coords), None)
    lon_name = next((n for n in ("lon", "longitude") if n in tos.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"no lat/lon: {list(tos.coords)}")
    lat = tos[lat_name]; lon = tos[lon_name]
    lon180 = ((lon + 180) % 360) - 180
    mask = ((lat >= SUBPOLAR_NA["lat_min"]) & (lat <= SUBPOLAR_NA["lat_max"])
            & (lon180 >= SUBPOLAR_NA["lon_min"]) & (lon180 <= SUBPOLAR_NA["lon_max"]))
    coslat = np.cos(np.deg2rad(lat))
    weight = (coslat * mask).where(mask)
    masked = tos.where(mask)
    spatial_dims = [d for d in tos.dims if d != "time"]
    num = (masked * weight).sum(dim=spatial_dims, skipna=True)
    den = weight.where(np.isfinite(masked)).sum(dim=spatial_dims, skipna=True)
    return (num / den).groupby("time.year").mean("time")


def cmip6_unified() -> dict[str, dict[str, dict]]:
    """For each model: subpolar absolute annual mean per experiment, plus
    the model's 1870-1950 historical baseline; subpolar anomaly per
    experiment = absolute - baseline."""
    print("Connecting to Pangeo CMIP6 catalog...")
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

    out: dict[str, dict[str, dict]] = {}

    # ----- First pass: historical baselines -------------------------------
    print("\n-- historical baselines --")
    for member_id in ("r1i1p1f1", "r1i1p1f2"):
        q = cat.search(source_id=MODELS, experiment_id="historical",
                       variable_id="tos", table_id="Omon", member_id=member_id)
        for source_id in q.df["source_id"].unique():
            if source_id in out: continue
            t0 = time.time()
            sub = q.df[q.df["source_id"] == source_id]
            if "gn" in sub["grid_label"].values:
                sub = sub[sub["grid_label"] == "gn"]
            zstore = sub.iloc[0]["zstore"]
            try:
                ds = xr.open_zarr(zstore, consolidated=True, chunks={})
                sp_ann = _cmip6_subpolar_annual(ds).load()
                yrs = sp_ann["year"].values
                vals = sp_ann.values
                base_mask = (yrs >= BASELINE[0]) & (yrs <= BASELINE[1])
                baseline = float(vals[base_mask].mean()) if base_mask.any() else np.nan
                out[source_id] = {
                    "member_id": member_id,
                    "baseline_1870_1950": baseline,
                    "experiments": {
                        "historical": {
                            "years": [int(y) for y in yrs],
                            "anomaly_degC": [float(v - baseline) for v in vals],
                        },
                    },
                }
                print(f"  {source_id:<20}  member {member_id}  "
                      f"baseline {baseline:.3f}°C  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {source_id}: FAILED -- {type(e).__name__}: {e}")

    # ----- Second pass: SSPs anchored to historical baseline --------------
    for exp in ("ssp245", "ssp585"):
        print(f"\n-- {exp} (anchored to model historical baseline) --")
        for member_id in ("r1i1p1f1", "r1i1p1f2"):
            q = cat.search(source_id=list(out.keys()), experiment_id=exp,
                           variable_id="tos", table_id="Omon",
                           member_id=member_id)
            for source_id in q.df["source_id"].unique():
                if exp in out.get(source_id, {}).get("experiments", {}):
                    continue
                t0 = time.time()
                sub = q.df[q.df["source_id"] == source_id]
                if "gn" in sub["grid_label"].values:
                    sub = sub[sub["grid_label"] == "gn"]
                zstore = sub.iloc[0]["zstore"]
                try:
                    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
                    sp_ann = _cmip6_subpolar_annual(ds).load()
                    yrs = sp_ann["year"].values
                    vals = sp_ann.values
                    baseline = out[source_id]["baseline_1870_1950"]
                    out[source_id]["experiments"][exp] = {
                        "years": [int(y) for y in yrs],
                        "anomaly_degC": [float(v - baseline) for v in vals],
                    }
                    print(f"  {source_id:<20}  member {member_id}  "
                          f"{len(yrs)} yr  ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  {source_id}: FAILED -- {type(e).__name__}: {e}")

    return out


def ensemble_band(data: dict, experiment: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (years, p50, p16, p84) across models for one experiment."""
    arrs = []
    yrs_all = []
    for model, info in data.items():
        if experiment not in info["experiments"]: continue
        yrs_all.append(np.array(info["experiments"][experiment]["years"]))
        arrs.append(np.array(info["experiments"][experiment]["anomaly_degC"]))
    if not arrs:
        return None, None, None, None
    yr_min = min(y.min() for y in yrs_all)
    yr_max = max(y.max() for y in yrs_all)
    grid_years = np.arange(yr_min, yr_max + 1)
    grid = np.full((len(arrs), len(grid_years)), np.nan)
    for i, (y, a) in enumerate(zip(yrs_all, arrs)):
        idx = np.searchsorted(grid_years, y)
        grid[i, idx] = a
    p50 = np.nanmedian(grid, axis=0)
    p16 = np.nanpercentile(grid, 16, axis=0)
    p84 = np.nanpercentile(grid, 84, axis=0)
    return grid_years, p50, p16, p84


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(palmod_yrs, palmod_mean, palmod_mat,
                  hadisst_yrs, hadisst_anom,
                  cmip6_data):
    """Two-panel figure mirroring fig:envelope_breakout layout:
       (a) full 12-kyr continuous axis with all sources overlaid; the
           modern cliff is visible at the right edge.
       (b) zoom on the dashed box of (a) for modern + projections.
    """
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.18},
    )

    p16 = np.nanpercentile(palmod_mat, 16, axis=0)
    p84 = np.nanpercentile(palmod_mat, 84, axis=0)
    yrs_h, h_p50, h_p16, h_p84 = ensemble_band(cmip6_data, "historical")
    yrs_24, s24_p50, s24_p16, s24_p84 = ensemble_band(cmip6_data, "ssp245")
    yrs_58, s58_p50, s58_p16, s58_p84 = ensemble_band(cmip6_data, "ssp585")

    # ----- Panel (a): continuous time axis 10 000 BCE to 2100 CE ----------
    ax_a.fill_between(palmod_yrs, p16, p84, color="C0", alpha=0.20)
    ax_a.plot(palmod_yrs, palmod_mean, color="C0", lw=1.6,
              label="PALMOD-130k basin mean")
    if yrs_h is not None:
        ax_a.fill_between(yrs_h, h_p16, h_p84, color="0.55", alpha=0.28)
        ax_a.plot(yrs_h, h_p50, color="0.30", lw=1.1)
    if yrs_24 is not None:
        m24 = yrs_24 <= 2100
        ax_a.fill_between(yrs_24[m24], s24_p16[m24], s24_p84[m24],
                           color="C1", alpha=0.18)
        ax_a.plot(yrs_24[m24], s24_p50[m24], color="C1", lw=1.2)
    if yrs_58 is not None:
        m58 = yrs_58 <= 2100
        ax_a.fill_between(yrs_58[m58], s58_p16[m58], s58_p84[m58],
                           color="C3", alpha=0.22)
        ax_a.plot(yrs_58[m58], s58_p50[m58], color="C3", lw=1.4)
    ax_a.plot(hadisst_yrs, hadisst_anom, color="k", lw=1.6)
    ax_a.axhline(0, color="0.4", lw=0.5)
    ax_a.set_xlim(-10000, 2100)
    ax_a.set_ylim(-2.5, 5.5)
    ax_a.set_xlabel("year CE")
    ax_a.set_ylabel("subpolar NA SST anomaly (°C, vs 1870--1950)")
    ax_a.set_title("(a) Full 12-kyr arc: continuous time axis",
                   loc="left", fontsize=10)
    ax_a.grid(alpha=0.20)

    # Dashed zoom box on panel (a)
    from matplotlib.patches import Rectangle
    zoom_rect = Rectangle((1850, -2.5), 250, 8.0,
                           fill=False, edgecolor="0.2", lw=0.8,
                           linestyle="--", zorder=10)
    ax_a.add_patch(zoom_rect)

    # ----- Panel (b): zoom 1850 to 2100 ------------------------------------
    if yrs_h is not None:
        ax_b.fill_between(yrs_h, h_p16, h_p84, color="0.55", alpha=0.30)
        ax_b.plot(yrs_h, h_p50, color="0.30", lw=1.4,
                   label="CMIP6 historical")
    if yrs_24 is not None:
        m24 = yrs_24 <= 2100
        ax_b.fill_between(yrs_24[m24], s24_p16[m24], s24_p84[m24],
                           color="C1", alpha=0.20)
        ax_b.plot(yrs_24[m24], s24_p50[m24], color="C1", lw=1.4,
                   label="CMIP6 ssp245")
    if yrs_58 is not None:
        m58 = yrs_58 <= 2100
        ax_b.fill_between(yrs_58[m58], s58_p16[m58], s58_p84[m58],
                           color="C3", alpha=0.22)
        ax_b.plot(yrs_58[m58], s58_p50[m58], color="C3", lw=1.6,
                   label="CMIP6 ssp585")
    ax_b.plot(hadisst_yrs, hadisst_anom, color="k", lw=1.8,
               label="HadISST observed")
    ax_b.axhline(0, color="0.4", lw=0.5)
    ax_b.axvline(2014.5, color="0.5", lw=0.7, linestyle=":")
    ax_b.set_xlim(1850, 2100)
    ax_b.set_ylim(-2.5, 5.5)
    ax_b.set_xlabel("year CE")
    ax_b.set_title("(b) Zoom on dashed box of panel (a)",
                   loc="left", fontsize=10)
    ax_b.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax_b.grid(alpha=0.20)

    out_fig = MANUSCRIPT_FIGS / "fig_unified_timeline.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\nWrote {out_fig}")


def main():
    CACHE_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Source 1: PALMOD")
    print("=" * 60)
    palmod_yrs, palmod_mean, palmod_mat = palmod_unified()
    print(f"  range: {palmod_yrs.min():.0f} CE to {palmod_yrs.max():.0f} CE")
    print(f"  re-anchored basin mean: range {np.nanmin(palmod_mean):+.3f} to "
          f"{np.nanmax(palmod_mean):+.3f} °C")

    print("\n" + "=" * 60)
    print(" Source 2: HadISST 1870-2023")
    print("=" * 60)
    had_yrs, had_anom = hadisst_unified()
    print(f"  range: {had_yrs.min()}-{had_yrs.max()}, "
          f"anomaly {np.min(had_anom):+.2f} to {np.max(had_anom):+.2f} °C")

    print("\n" + "=" * 60)
    print(" Source 3: CMIP6 ensemble")
    print("=" * 60)
    cache_file = CACHE_OUT / "cmip6_subpolar_unified.json"
    if cache_file.exists():
        print(f"  using cached {cache_file}")
        with open(cache_file) as f:
            cmip6_data = json.load(f)
    else:
        cmip6_data = cmip6_unified()
        with open(cache_file, "w") as f:
            json.dump(cmip6_data, f, indent=2)
        print(f"\nWrote {cache_file}")

    print("\n" + "=" * 60)
    print(" Building figure")
    print("=" * 60)
    build_figure(palmod_yrs, palmod_mean, palmod_mat,
                  had_yrs, had_anom, cmip6_data)


if __name__ == "__main__":
    main()
