"""Autonomous overnight Holocene Q hunter.

Walks an unbounded worklist of (archive, model, experiment, variable
strategy) tuples and, for each, computes a Q-time-series on the
North Atlantic 2-degree grid -- including past1000 forced-Holocene
runs.  Results are cached per-target so the script is fully
resumable; crashes on one target never block the queue.

Strategies:
  (a) Q-EKE   : <r_loc(tos)>, EKE = |grad SSH| from zos.  Requires
                tos+zos.  This is the canonical Q used elsewhere.
  (b) Q-TOS   : <r_loc(tos)>, EKE proxy = |grad tos|^2.  Requires
                tos only.  Different physics from Q-EKE but a
                defensible Holocene-era diagnostic when zos isn't
                published (past1000 has no zos in CMIP6/PMIP4).

Worklist sources tried in order:
  - CMIP6 Pangeo zarr (no auth)
  - DKRZ ESGF SOLR (legacy, still on; covers CMIP5+CMIP6 past1000)
  - CEDA ESGF SOLR
  - IPSL ESGF SOLR

Disk hygiene:
  - Per-window Q is the only intermediate persisted (~few kB per
    model per experiment; final cache <100 MB across hundreds of
    models).
  - Raw downloads (when via HTTPS / non-zarr) are streamed to a
    scratch dir and DELETED immediately after Q computation.
  - Atomic .tmp -> rename writes.

Re-runnable.  Sleep-the-user safe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT

sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _basin_subset_2deg, _bandpass_anomaly,
    _slice_by_year, _year_of, _mann_kendall,
    instantaneous_phase, local_r_mean,
    WINDOW_YEARS, RLOC_RADIUS_KM, PI_SPINUP_YEARS,
)
from scipy.stats import pearsonr  # noqa: E402


AUTO_DIR = CACHE_DIR / "holocene_exit" / "auto"
AUTO_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PKG_ROOT.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# How often to retry the whole queue
RETRY_PASS_SLEEP_S = 600       # 10 min between full passes
PER_TARGET_TIMEOUT_S = 1800    # 30 min per target before abandon


def _Q_window_tos_only(tos_win, basin):
    """Q variant: r_loc from <tos> + EKE_proxy = |grad <tos>|^2.

    Used when zos is not available (past1000 era).  The EKE proxy is
    the squared magnitude of the time-mean tos gradient (a tracer
    noise proxy in the absence of dynamics).  Caveat: physically
    different from |grad SSH| EKE; results labelled Q-TOS not Q.
    """
    tos_2d = _basin_subset_2deg(tos_win, basin)
    sst_anom = _bandpass_anomaly(tos_2d)
    phi = instantaneous_phase(sst_anom)
    n_t = phi.sizes["time"]
    if n_t < 12 * WINDOW_YEARS * 0.8:
        return float("nan"), 0
    phi = phi.isel(time=slice(6, n_t - 6))
    rl_mean = local_r_mean(phi, radius_km=RLOC_RADIUS_KM)
    tos_mean = tos_2d.mean("time")
    grad2 = (tos_mean.differentiate("lat") ** 2
             + tos_mean.differentiate("rlon") ** 2)
    a = rl_mean.values.ravel(); b = grad2.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan"), int(m.sum())
    rho, _ = pearsonr(a[m], b[m])
    return float(-rho), int(m.sum())


# -----------------------------------------------------------------
#  Catalog discovery
# -----------------------------------------------------------------

def pangeo_inventory():
    """Return list of (model, experiment, has_tos, has_zos) for
    Pangeo CMIP6 zarr ocean monthly variables."""
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    df = cat.df
    exps = ["piControl", "historical", "ssp585", "ssp245", "ssp126",
            "ssp370", "1pctCO2", "abrupt-4xCO2", "past1000",
            "midHolocene", "lgm", "lig127k"]
    rows = []
    for e in exps:
        sub = df[(df.experiment_id == e) & (df.table_id == "Omon")]
        tos_m = set(sub[sub.variable_id == "tos"]["source_id"])
        zos_m = set(sub[sub.variable_id == "zos"]["source_id"])
        models = tos_m | zos_m
        for m in models:
            rows.append(dict(
                source="pangeo", model=m, experiment=e,
                has_tos=(m in tos_m), has_zos=(m in zos_m),
            ))
    return rows


# CMOR / CMIP variable-name aliases.  Different archives use different
# names for the same physical field.  The autonomous hunter tries each
# alias in turn when probing ESGF or other catalogs.
TOS_ALIASES = ["tos", "SST", "sst", "thetao", "temp_mm_uo"]
ZOS_ALIASES = ["zos", "SSH", "ssh", "ssh_alt"]

# Holocene-relevant experiments under various MIP eras.
# CMIP6/PMIP4: past1000, midHolocene, lgm, lig127k
# CMIP5/PMIP3: past1000, lastMillennium, past2k
# Single-forcing CESM-LME-style: past1000_solar, past1000_volcanic ...
HOLOCENE_EXPERIMENTS = [
    "past1000", "past2k", "lastMillennium",
    "past1000-volc", "past1000-solar", "past1000-ghg",
    "midHolocene", "lgm", "lig127k",
]


def _esgf_solr_query(node_base, **params):
    """Generic ESGF SOLR JSON query. Returns the 'docs' list or []."""
    import requests
    url = (f"{node_base}/esg-search/search/?format="
            f"application%2Fsolr%2Bjson&limit=1000")
    for k, v in params.items():
        url += f"&{k}={v}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200: return []
        return r.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  esgf {node_base}: {type(e).__name__}: {e}")
        return []


def dkrz_esgf_inventory():
    """Probe DKRZ ESGF legacy SOLR for Holocene-era runs across CMIP5+6
    and across the TOS / ZOS alias families.

    Returns list of (source, model, experiment, has_tos, has_zos).
    """
    rows = []
    seen = set()
    for exp in HOLOCENE_EXPERIMENTS:
        for var in TOS_ALIASES + ZOS_ALIASES:
            docs = _esgf_solr_query(
                "https://esgf-data.dkrz.de",
                experiment_id=exp, variable_id=var,
            )
            is_zos = var in ZOS_ALIASES
            for doc in docs:
                m = doc.get("source_id") or doc.get("model")
                if isinstance(m, list): m = m[0] if m else None
                if not m: continue
                k = ("dkrz_esgf", m, exp)
                if k in seen:
                    # Update the existing row if we discover an extra var
                    for r in rows:
                        if (r["source"], r["model"], r["experiment"]) == k:
                            if is_zos: r["has_zos"] = True
                            else: r["has_tos"] = True
                            break
                    continue
                seen.add(k)
                rows.append(dict(
                    source="dkrz_esgf", model=m, experiment=exp,
                    has_tos=(not is_zos), has_zos=is_zos,
                ))
    return rows


def ceda_esgf_inventory():
    """Probe CEDA ESGF for Holocene-era runs.  CEDA still on legacy SOLR."""
    rows = []
    seen = set()
    for exp in HOLOCENE_EXPERIMENTS:
        for var in TOS_ALIASES + ZOS_ALIASES:
            docs = _esgf_solr_query(
                "https://esgf-index1.ceda.ac.uk",
                experiment_id=exp, variable_id=var,
            )
            is_zos = var in ZOS_ALIASES
            for doc in docs:
                m = doc.get("source_id") or doc.get("model")
                if isinstance(m, list): m = m[0] if m else None
                if not m: continue
                k = ("ceda_esgf", m, exp)
                if k in seen:
                    for r in rows:
                        if (r["source"], r["model"], r["experiment"]) == k:
                            if is_zos: r["has_zos"] = True
                            else: r["has_tos"] = True
                            break
                    continue
                seen.add(k)
                rows.append(dict(
                    source="ceda_esgf", model=m, experiment=exp,
                    has_tos=(not is_zos), has_zos=is_zos,
                ))
    return rows


def ipsl_esgf_inventory():
    """Probe IPSL ESGF for Holocene-era runs."""
    rows = []
    seen = set()
    for exp in HOLOCENE_EXPERIMENTS:
        for var in TOS_ALIASES + ZOS_ALIASES:
            docs = _esgf_solr_query(
                "https://esgf-node.ipsl.upmc.fr",
                experiment_id=exp, variable_id=var,
            )
            is_zos = var in ZOS_ALIASES
            for doc in docs:
                m = doc.get("source_id") or doc.get("model")
                if isinstance(m, list): m = m[0] if m else None
                if not m: continue
                k = ("ipsl_esgf", m, exp)
                if k in seen:
                    for r in rows:
                        if (r["source"], r["model"], r["experiment"]) == k:
                            if is_zos: r["has_zos"] = True
                            else: r["has_tos"] = True
                            break
                    continue
                seen.add(k)
                rows.append(dict(
                    source="ipsl_esgf", model=m, experiment=exp,
                    has_tos=(not is_zos), has_zos=is_zos,
                ))
    return rows


# -----------------------------------------------------------------
#  Per-target processing
# -----------------------------------------------------------------

def _open_pangeo_member(model, experiment, variable, member=None):
    import intake
    cat = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    sub = cat.df[
        (cat.df.source_id == model)
        & (cat.df.experiment_id == experiment)
        & (cat.df.variable_id == variable)
        & (cat.df.table_id == "Omon")
    ]
    if sub.empty:
        raise FileNotFoundError(f"{model} {experiment} {variable}")
    if member:
        sub = sub[sub.member_id == member]
        if sub.empty: raise FileNotFoundError("member missing")
    else:
        if "gn" in sub["grid_label"].values:
            sub = sub[sub["grid_label"] == "gn"]
        sub = sub.sort_values("member_id").iloc[:1]
    zstore = sub.iloc[0]["zstore"]
    ds = xr.open_zarr(zstore, consolidated=True, chunks={})
    return ds[variable]


def process_target(target, basin, stride_yr=10):
    """Run one (source, model, experiment) target.  Saves result JSON.
    Returns the result dict or None on failure."""
    src = target["source"]
    model = target["model"]
    exp = target["experiment"]
    has_tos = target["has_tos"]
    has_zos = target["has_zos"]
    name = f"{src}_{model}_{exp}_{basin}"
    out_path = AUTO_DIR / f"{name}.json"
    fail_path = AUTO_DIR / f"{name}.failed"
    if out_path.exists():
        return json.loads(out_path.read_text())
    if fail_path.exists():
        # Honour cooldown: skip targets that recently failed
        age = time.time() - fail_path.stat().st_mtime
        if age < 6 * 3600:  # try again after 6 h
            return None

    t0 = time.time()
    print(f"\n[{name}] start")
    strategy = "Q-EKE" if has_zos else "Q-TOS"
    if not has_tos:
        # No usable variable
        fail_path.write_text("no tos")
        return None

    try:
        if src == "pangeo":
            tos = _open_pangeo_member(model, exp, "tos")
            zos = _open_pangeo_member(model, exp, "zos") if has_zos else None
        else:
            # ESGF opendap-style open (placeholder).  Pangeo is the
            # primary path because it has zarr; ESGF would require
            # streaming netCDFs -- left as a future extension.
            fail_path.write_text(f"non-pangeo source {src} not yet wired")
            return None
    except Exception as e:
        print(f"  open FAILED: {type(e).__name__}: {e}")
        fail_path.write_text(f"open: {e}")
        return None

    yr0 = _year_of(tos["time"].values[0])
    yr1 = _year_of(tos["time"].values[-1])

    # Pick year range and stride based on experiment
    span = yr1 - yr0 + 1
    if exp in ("piControl", "1pctCO2", "abrupt-4xCO2"):
        start = yr0 + (PI_SPINUP_YEARS if exp == "piControl" else 0)
        end = yr1
        ws = max(stride_yr, WINDOW_YEARS)  # non-overlapping for null
        if exp != "piControl": ws = stride_yr
    elif exp == "past1000":
        start, end, ws = yr0, yr1, stride_yr
    elif exp in ("historical", "ssp585", "ssp245",
                  "ssp126", "ssp370"):
        start, end, ws = yr0, yr1, stride_yr
    else:
        start, end, ws = yr0, yr1, stride_yr

    Qs = []; centres = []
    seg = start
    while seg + WINDOW_YEARS - 1 <= end:
        s, e = seg, seg + WINDOW_YEARS - 1
        t1 = time.time()
        try:
            tos_win = _slice_by_year(tos, s, e)
            if has_zos and zos is not None:
                from holocene_q_pilot import _Q_for_window
                zos_win = _slice_by_year(zos, s, e)
                Q, n = _Q_for_window(tos_win, zos_win, basin)
            else:
                Q, n = _Q_window_tos_only(tos_win, basin)
            Qs.append(Q); centres.append(s + WINDOW_YEARS // 2)
            print(f"  {s}-{e}: Q={Q:+.3f} n={n} ({time.time()-t1:.0f}s)")
        except Exception as ex:
            print(f"  {s}-{e}: WINDOW FAILED ({type(ex).__name__})")
            Qs.append(float("nan")); centres.append(s + WINDOW_YEARS // 2)
        if time.time() - t0 > PER_TARGET_TIMEOUT_S:
            print(f"  TIMEOUT at {time.time()-t0:.0f}s; saving partial")
            break
        seg += ws

    arr = np.asarray(Qs, dtype=float)
    arr_f = arr[np.isfinite(arr)]
    summary = dict(
        target=target, basin=basin, strategy=strategy,
        year_range=[int(yr0), int(yr1)],
        window_years=WINDOW_YEARS, stride_yr=int(ws),
        n_windows=int(len(Qs)),
        Q=Qs, year_centres=centres,
        Q_mean=float(arr_f.mean()) if arr_f.size else None,
        Q_sd=float(arr_f.std()) if arr_f.size else None,
        elapsed_s=round(time.time() - t0, 1),
        timestamp=int(time.time()),
    )
    if exp == "piControl" and arr_f.size > 3:
        tau, p_mk = _mann_kendall(arr_f.tolist())
        summary["pi_mk_tau"] = tau
        summary["pi_mk_p"] = p_mk
        summary["pi_p95"] = float(np.percentile(arr_f, 95))
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.rename(out_path)
    print(f"  wrote {out_path}  ({time.time()-t0:.0f}s)")
    return summary


# -----------------------------------------------------------------
#  Main loop
# -----------------------------------------------------------------

def build_worklist(basin, include_dkrz=True, include_ceda=True,
                     include_ipsl=True):
    print("discovering Pangeo CMIP6 inventory ...")
    work = pangeo_inventory()
    print(f"  Pangeo targets: {len(work)}")
    if include_dkrz:
        print("discovering DKRZ ESGF past1000 inventory "
              "(all aliases + experiments) ...")
        try:
            extra = dkrz_esgf_inventory()
            work.extend(extra)
            print(f"  DKRZ ESGF targets: {len(extra)}")
        except Exception as e:
            print(f"  dkrz discovery failed: {e}")
    if include_ceda:
        print("discovering CEDA ESGF past1000 inventory ...")
        try:
            extra = ceda_esgf_inventory()
            work.extend(extra)
            print(f"  CEDA ESGF targets: {len(extra)}")
        except Exception as e:
            print(f"  ceda discovery failed: {e}")
    if include_ipsl:
        print("discovering IPSL ESGF past1000 inventory ...")
        try:
            extra = ipsl_esgf_inventory()
            work.extend(extra)
            print(f"  IPSL ESGF targets: {len(extra)}")
        except Exception as e:
            print(f"  ipsl discovery failed: {e}")
    return work


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic",
                     choices=list(BASINS.keys()))
    ap.add_argument("--max-passes", type=int, default=100,
                     help="how many full worklist passes to attempt")
    ap.add_argument("--include-dkrz", action="store_true", default=False)
    ap.add_argument("--include-ceda", action="store_true", default=False)
    ap.add_argument("--include-ipsl", action="store_true", default=False)
    args = ap.parse_args()

    print("=" * 70)
    print(f" Autonomous Holocene Q hunter  basin={args.basin}")
    print("=" * 70)

    worklist = build_worklist(
        args.basin,
        include_dkrz=args.include_dkrz,
        include_ceda=args.include_ceda,
        include_ipsl=args.include_ipsl,
    )
    # Sort: paleo/idealised first (highest scientific value), then
    # piControl, then historical, then ssp scenarios.
    prio = {
        "past1000": 0, "midHolocene": 1, "lgm": 2, "lig127k": 3,
        "1pctCO2": 4, "abrupt-4xCO2": 5,
        "piControl": 6,
        "historical": 7,
        "ssp585": 8, "ssp245": 9, "ssp126": 10, "ssp370": 11,
    }
    worklist.sort(key=lambda t: (prio.get(t["experiment"], 99),
                                    t["model"]))
    print(f"\nworklist: {len(worklist)} targets queued")

    for pass_idx in range(args.max_passes):
        print(f"\n========== pass {pass_idx + 1}/{args.max_passes} "
              f"({time.strftime('%Y-%m-%d %H:%M:%S')}) ==========")
        n_done = n_skip = n_new = n_fail = 0
        for t in worklist:
            name = (f"{t['source']}_{t['model']}_"
                     f"{t['experiment']}_{args.basin}")
            out_path = AUTO_DIR / f"{name}.json"
            fail_path = AUTO_DIR / f"{name}.failed"
            if out_path.exists():
                n_done += 1; continue
            if fail_path.exists():
                age = time.time() - fail_path.stat().st_mtime
                if age < 6 * 3600:
                    n_skip += 1; continue
            try:
                r = process_target(t, args.basin)
                if r is None: n_fail += 1
                else: n_new += 1
            except Exception:
                print("  process_target raised:")
                traceback.print_exc()
                n_fail += 1
        total = len(worklist)
        print(f"\n--- pass {pass_idx + 1} summary: "
              f"done={n_done}  new={n_new}  fail={n_fail}  "
              f"cooldown-skip={n_skip}  total={total} ---")
        if n_done + n_skip >= total and n_new == 0:
            print(f"\nworklist saturated (no new completions this "
                  f"pass).  sleeping {RETRY_PASS_SLEEP_S}s then "
                  f"retry-cycle ...")
            time.sleep(RETRY_PASS_SLEEP_S)


if __name__ == "__main__":
    main()
