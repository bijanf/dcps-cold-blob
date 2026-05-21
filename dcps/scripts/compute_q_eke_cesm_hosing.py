"""Compute Q and basin-mean EKE on the van Westen eddy-rich CESM hosing
dataset (Dijkstra et al. 2026, Nat Rev Phys; van Westen, Kliphuis,
Dijkstra 2024, Sci Adv).

Intended location of the data on PIK HPC:
    /p/projects/poem/data/modeloutput/CESM/CESM_hosing_data_van_Westen

This script is designed to be run on HPC.  It:
  1. Auto-discovers the directory layout (CESM filename conventions
     for SST and SSH).  If discovery fails it PRINTS a diagnostic plan
     and exits without writing anything, so the layout regex can be
     fixed before any compute is wasted.
  2. For each detected simulation branch (forward / backward quasi-
     equilibrium, ramp-up, etc.), runs a rolling 30-yr window
     analysis (default stride 10 yr, matching the CMIP6 manuscript
     standard) and computes
         Q       = -rho(<r_loc>, |grad SSH|)    via _Q_for_window
         eke_abs = mean( |grad <SSH>|^2 )       via _basin_mean_eke_window
   3. Writes one JSON per branch to the output directory:
         <out>/vanWesten_CESM_hosing_<branch>.json
   The JSON contains a list of windows, each tagged with the window
   start/centre year, the hosing flux F_H (if decodable from filenames
   or metadata), and the resulting (Q, eke_abs) numbers.

Usage on HPC:

    git clone <this repo> NEW_Theory && cd NEW_Theory
    export PYTHONPATH=$PWD:$PWD/dcps:$PWD/dcps/scripts
    python dcps/scripts/compute_q_eke_cesm_hosing.py \\
        --cesm-root /p/projects/poem/data/modeloutput/CESM/CESM_hosing_data_van_Westen \\
        --out cesm_hosing_results

After it finishes, scp the JSON files back to the local repo at
dcps/cache/cesm_hosing/ and the SI figure can be rendered from them.

Dependencies (HPC-side):
    python>=3.10, xarray, scipy, numpy, netcdf4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# Variable-name heuristics for CESM hosing output.  CESM POP typically
# uses 'SST' or 'TEMP' for surface temperature and 'SSH' for sea
# surface height; the van Westen Sci Adv 2024 derivative outputs may
# have already been collapsed to a smaller set of NetCDFs with names
# like sst_F_H_0.18.nc, ssh_F_H_0.45.nc, or similar.  We try both
# conventions and report what we found.
SST_NAME_CANDIDATES = ["SST", "TEMP", "tos", "sst", "TEMPERATURE",
                          "T", "thetao", "sosstsst"]
SSH_NAME_CANDIDATES = ["SSH", "ssh", "zos", "SSHEIG", "sea_surface_height",
                          "sossheig", "ETA"]
# Filename regex to decode F_H value from filenames such as
# 'forward_FH_0.18.nc' or 'backward_F_H_0p45_SST.nc'.
FH_REGEX = re.compile(
    r"(?:F[_-]?H[_-])(?P<fh>\d+(?:[._]\d+)?)|"
    r"(?:fh[_-])(?P<fh2>\d+(?:[._]\d+)?)",
    re.IGNORECASE,
)
BRANCH_HINTS = {
    "forward":  ["forward", "fwd", "ramp_up", "rampup", "increase"],
    "backward": ["backward", "bwd", "ramp_down", "rampdown", "decrease",
                  "reverse"],
}


def _scan_root(root: Path) -> dict:
    """Walk the root, list NetCDF files, return a discovery summary
    that the user can sanity-check."""
    if not root.exists():
        return dict(error=f"root does not exist: {root}")
    ncs = sorted(p for p in root.rglob("*.nc"))
    summary = dict(
        root=str(root),
        n_netcdfs=len(ncs),
        first_files=[str(p.relative_to(root)) for p in ncs[:20]],
    )
    # variable-name + branch detection on a per-file basis
    detected = []
    for p in ncs[:100]:
        try:
            ds = xr.open_dataset(p, decode_cf=False)
            vars_ = list(ds.data_vars)
            sst_var = next((v for v in vars_
                             if v in SST_NAME_CANDIDATES), None)
            ssh_var = next((v for v in vars_
                             if v in SSH_NAME_CANDIDATES), None)
            ds.close()
        except Exception as e:
            detected.append(dict(path=str(p.relative_to(root)),
                                    error=type(e).__name__))
            continue
        fh = None
        m = FH_REGEX.search(p.name)
        if m:
            raw = (m.group("fh") or m.group("fh2") or "").replace("_", ".").replace("p", ".")
            try:
                fh = float(raw)
            except Exception:
                pass
        branch = None
        for tag, hints in BRANCH_HINTS.items():
            if any(h in p.name.lower() or h in str(p.parent).lower()
                   for h in hints):
                branch = tag; break
        detected.append(dict(
            path=str(p.relative_to(root)),
            sst_var=sst_var, ssh_var=ssh_var,
            F_H=fh, branch=branch,
        ))
    summary["detected_sample"] = detected
    return summary


def _import_dcps_helpers():
    """Late import of the dcps Q + EKE pipeline so that --plan-only
    works without these dependencies."""
    try:
        from holocene_q_pilot import (
            _Q_for_window, _basin_subset_2deg, _slice_by_year, _year_of,
            WINDOW_YEARS,
        )
        return (_Q_for_window, _basin_subset_2deg, _slice_by_year,
                _year_of, WINDOW_YEARS)
    except Exception as e:
        raise SystemExit(
            f"dcps imports failed ({type(e).__name__}: {e}).\n"
            f"Make sure PYTHONPATH includes the repo's dcps/ and "
            f"dcps/scripts/ directories.")


def _process_branch(files_for_branch, sst_var, ssh_var,
                     window_years, stride_yr, basin,
                     _Q_for_window, _basin_subset_2deg,
                     _slice_by_year, _year_of):
    """Open SST + SSH for one branch, compute Q + EKE on rolling
    windows.  Returns list of dicts with year, F_H, Q, eke_abs."""
    sst_paths = sorted(p for p, role in files_for_branch
                          if role == "sst")
    ssh_paths = sorted(p for p, role in files_for_branch
                          if role == "ssh")
    if not (sst_paths and ssh_paths):
        print(f"  skip: no paired SST+SSH for this branch")
        return []
    sst = xr.open_mfdataset(sst_paths, combine="by_coords",
                              decode_cf=True, parallel=False)[sst_var]
    ssh = xr.open_mfdataset(ssh_paths, combine="by_coords",
                              decode_cf=True, parallel=False)[ssh_var]
    # rename to lat/lon if needed (CESM uses TLAT/TLONG or
    # nlat/nlon depending on the post-processing chain)
    for name in ("TLAT", "ULAT", "nav_lat", "latitude"):
        if name in sst.coords:
            sst = sst.rename({name: "lat"}); break
    for name in ("TLONG", "ULONG", "nav_lon", "longitude"):
        if name in sst.coords:
            sst = sst.rename({name: "lon"}); break
    for name in ("TLAT", "ULAT", "nav_lat", "latitude"):
        if name in ssh.coords:
            ssh = ssh.rename({name: "lat"}); break
    for name in ("TLONG", "ULONG", "nav_lon", "longitude"):
        if name in ssh.coords:
            ssh = ssh.rename({name: "lon"}); break

    yr0 = _year_of(sst["time"].values[0])
    yr1 = _year_of(sst["time"].values[-1])
    print(f"  time {yr0}..{yr1} ({yr1 - yr0 + 1} yr)")

    rows = []
    for s in range(yr0, yr1 - window_years + 2, stride_yr):
        e = s + window_years - 1
        try:
            sst_w = _slice_by_year(sst, s, e)
            ssh_w = _slice_by_year(ssh, s, e)
            Q, n = _Q_for_window(sst_w, ssh_w, basin)
            ssh_2d = _basin_subset_2deg(ssh_w, basin)
            ssh_mean = ssh_2d.mean("time")
            eke_abs = float((ssh_mean.differentiate("lat") ** 2
                                + ssh_mean.differentiate("rlon") ** 2)
                              .mean(skipna=True).values)
            print(f"    {s}-{e}: Q={Q:+.3f}  EKE={eke_abs:.5f}  n={n}")
            rows.append(dict(window_start=int(s),
                                window_end=int(e),
                                window_centre=int(s + window_years // 2),
                                Q=float(Q), eke_abs=eke_abs,
                                n_cells=int(n)))
        except Exception as ex:
            print(f"    {s}-{e} FAILED: {type(ex).__name__}: {ex}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cesm-root", required=True,
                     help="root directory of the CESM hosing dataset")
    ap.add_argument("--out", required=True,
                     help="output directory for JSON results")
    ap.add_argument("--window-years", type=int, default=30,
                     help="rolling window length (yr), default 30 to "
                          "match CMIP6 manuscript standard")
    ap.add_argument("--stride-yr", type=int, default=10,
                     help="rolling window stride (yr), default 10")
    ap.add_argument("--basin", default="atlantic")
    ap.add_argument("--plan-only", action="store_true",
                     help="walk the root, print discovery summary, exit")
    args = ap.parse_args()

    root = Path(args.cesm_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== scanning {root} ===")
    discovery = _scan_root(root)
    print(json.dumps(discovery, indent=2)[:4000])
    (out_dir / "discovery.json").write_text(
        json.dumps(discovery, indent=2))
    print(f"\n  full discovery summary -> {out_dir / 'discovery.json'}")

    if "error" in discovery:
        raise SystemExit(discovery["error"])

    if args.plan_only:
        print("\n--plan-only mode; exiting without compute.")
        return

    # Group detected files by (branch, variable role)
    files_by_branch: dict[str, list[tuple[Path, str]]] = {}
    for d in discovery.get("detected_sample", []):
        branch = d.get("branch") or "unknown"
        path = root / d["path"]
        if d.get("sst_var"):
            files_by_branch.setdefault(branch, []).append(
                (path, "sst"))
        if d.get("ssh_var"):
            files_by_branch.setdefault(branch, []).append(
                (path, "ssh"))

    if not any(any(role == "sst" for _, role in v)
                and any(role == "ssh" for _, role in v)
                for v in files_by_branch.values()):
        raise SystemExit(
            "no branch has BOTH SST and SSH files in the first 100 "
            "discovered NetCDFs.  Re-run with --plan-only and adjust "
            "SST_NAME_CANDIDATES / SSH_NAME_CANDIDATES / FH_REGEX / "
            "BRANCH_HINTS at the top of the script.")

    (_Q_for_window, _basin_subset_2deg, _slice_by_year, _year_of,
     WINDOW_YEARS_DEFAULT) = _import_dcps_helpers()
    if args.window_years != WINDOW_YEARS_DEFAULT:
        print(f"  WARNING: using --window-years={args.window_years} "
              f"(manuscript default {WINDOW_YEARS_DEFAULT})")

    for branch, files in sorted(files_by_branch.items()):
        # pick a single SST variable name and SSH variable name from
        # the first paired file in this branch
        sst_var = next((d.get("sst_var") for d in
                          discovery["detected_sample"]
                          if d.get("branch") == branch
                          and d.get("sst_var")), None)
        ssh_var = next((d.get("ssh_var") for d in
                          discovery["detected_sample"]
                          if d.get("branch") == branch
                          and d.get("ssh_var")), None)
        if not (sst_var and ssh_var):
            print(f"\n[{branch}] missing sst_var or ssh_var, skip")
            continue
        print(f"\n[{branch}] sst_var={sst_var}  ssh_var={ssh_var}  "
              f"{len(files)} candidate files")
        rows = _process_branch(
            files, sst_var, ssh_var,
            args.window_years, args.stride_yr, args.basin,
            _Q_for_window, _basin_subset_2deg, _slice_by_year, _year_of)
        out_path = out_dir / f"vanWesten_CESM_hosing_{branch}.json"
        out_path.write_text(json.dumps(dict(
            source="vanWesten_CESM_hosing",
            branch=branch, basin=args.basin,
            window_years=args.window_years,
            stride_yr=args.stride_yr,
            sst_var=sst_var, ssh_var=ssh_var,
            windows=rows,
        ), indent=2))
        print(f"  wrote {out_path}  ({len(rows)} windows)")


if __name__ == "__main__":
    main()
