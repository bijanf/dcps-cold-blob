"""Streaming CEDA past1000 ingest -> Q time series.

For one CMIP5/PMIP3 model on CEDA, fetch its past1000 tos+zos
NetCDF chunks (typically ~100-yr each), regrid to the 2-deg NA basin
grid, compute Q on rolling 30-yr windows, save per-window Q to
JSON, and DELETE the raw NetCDFs.

Peak disk usage: ~200 MB (one tos chunk + one zos chunk in RAW_DIR
plus the partial Tier-1 cache).  Final cache per model: ~50 KB JSON.

Resumable: each chunk's processing is idempotent; downloads skip
existing files.

AUTH:
  CEDA does NOT accept HTTP Basic auth on its data servers; .netrc
  alone produces a 302 redirect to auth.ceda.ac.uk/account/signin.
  This script requires a CEDA Access Token (bearer).  To obtain one:

    1. log in at https://services.ceda.ac.uk/
    2. visit  https://services.ceda.ac.uk/access-token/  and click
       "Generate Access Token"; copy the resulting string.
    3. export CEDA_TOKEN='<paste-the-token>'         (in this shell)
       OR write the token to ~/.ceda_token  (single line, no quotes),
       and the script will pick it up automatically.

Usage:
  CEDA_TOKEN='...' python ceda_past1000_ingest.py \\
      --model MPI-ESM-P --member r1i1p1 --tos-version 20120625 \\
      --zos-version 20120625 --institute MPI-M
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from dcps.config import CACHE_DIR, PKG_ROOT
sys.path.insert(0, str(PKG_ROOT / "scripts"))
from multi_basin_quiescence import BASINS  # noqa: E402
from holocene_q_pilot import (  # noqa: E402
    _Q_for_window, _year_of,
    _slice_by_year, WINDOW_YEARS,
)


RAW_DIR = Path("/media/bijanf/writable/cmip5_past1000/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = CACHE_DIR / "cmip5_past1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ceda_token() -> str | None:
    """Return CEDA access token from $CEDA_TOKEN or ~/.ceda_token."""
    import os
    token = os.environ.get("CEDA_TOKEN")
    if token: return token.strip()
    p = Path.home() / ".ceda_token"
    if p.exists():
        t = p.read_text().strip()
        if t: return t
    return None


def _ceda_curl_args(token: str | None) -> list[str]:
    if token:
        return ["-H", f"Authorization: Bearer {token}"]
    return ["--netrc"]  # falls back to .netrc Basic (will 302 to login)


def _ceda_wget_args(token: str | None) -> list[str]:
    if token:
        return [f"--header=Authorization: Bearer {token}"]
    return ["--netrc"]


def _dap_url(institute, model, exp, table, member, var, version, fname):
    return (f"https://dap.ceda.ac.uk/badc/cmip5/data/cmip5/output1/"
            f"{institute}/{model}/{exp}/mon/ocean/{table}/{member}/"
            f"files/{var}_{version}/{fname}?download=1")


def _dkrz_url(institute, model, exp, table, member, var, version, fname):
    """DKRZ ESGF replica THREDDS URL (anonymous, no auth)."""
    return (f"https://esgf1.dkrz.de/thredds/fileServer/cmip5/cmip5/output1/"
            f"{institute}/{model}/{exp}/mon/ocean/{table}/{member}/"
            f"v{version}/{var}/{fname}")


def _esgf_ceda_url(institute, model, exp, table, member, var, version, fname):
    """CEDA THREDDS URL — a SEPARATE healthy host from the broken
    dap.ceda.ac.uk delivery server.  Works anonymously, also accepts
    the bearer token.  Path layout: /thredds/fileServer/esg_dataroot/
    <project>/<institute>/<model>/<exp>/<freq>/<realm>/<table>/<member>/
    v<version>/<var>/<fname>  (no ?download query string)."""
    return (f"https://esgf.ceda.ac.uk/thredds/fileServer/esg_dataroot/"
            f"cmip5/output1/{institute}/{model}/{exp}/mon/ocean/{table}/"
            f"{member}/v{version}/{var}/{fname}")


_HOST_CACHE: dict[tuple, str | None] = {}


def _probe_alt_host(institute, model, exp, member, var, version,
                      sample_fname) -> str | None:
    """Pick a working non-dap host for this dataset.  Tries
    esgf.ceda.ac.uk first (separate healthy CEDA THREDDS host) and
    DKRZ second; returns the URL template name to use, or None if the
    dataset is only on the broken dap.ceda.ac.uk.  Result is cached
    per (institute, model, member, var, version).
    """
    key = (institute, model, member, var, version)
    if key in _HOST_CACHE:
        return _HOST_CACHE[key]
    candidates = [
        ("esgf-ceda", _esgf_ceda_url(institute, model, exp, "Omon", member,
                                       var, version, sample_fname)),
        ("dkrz",      _dkrz_url(institute, model, exp, "Omon", member,
                                  var, version, sample_fname)),
    ]
    pick = None
    for tag, url in candidates:
        try:
            out = subprocess.run(
                ["curl", "--silent", "--head", "--location",
                 "--max-time", "15", "--connect-timeout", "10",
                 "--write-out", "%{http_code}", "-o", "/dev/null", url],
                capture_output=True, text=True, errors="replace", timeout=20,
            )
            code = (out.stdout or "").strip()
            if code in ("200", "206"):
                pick = tag
                break
        except Exception:
            continue
    _HOST_CACHE[key] = pick
    print(f"  alt-host probe {model} {member} {var} v{version}: "
          f"{('HIT '+pick+' — routing chunks via '+pick) if pick else 'miss — using dap.ceda.ac.uk'}")
    return pick


def _enumerate_chunks(institute, model, member, var, version, token):
    """Use CEDA's JSON index API to enumerate NetCDFs in the var/version
    directory.  Returns list of (fname, full_url)."""
    listing_url = (f"https://data.ceda.ac.uk/badc/cmip5/data/cmip5/output1/"
                    f"{institute}/{model}/past1000/mon/ocean/Omon/{member}/"
                    f"files/{var}_{version}/?json")
    print(f"  listing {listing_url}")
    out = subprocess.run(
        ["curl", *_ceda_curl_args(token), "--silent", "--max-time", "30",
          "-L", listing_url], capture_output=True, text=True, timeout=60,
    )
    try:
        idx = json.loads(out.stdout)
    except Exception as e:
        raise RuntimeError(
            f"CEDA JSON listing failed (auth?): {e}\n"
            f"first 300 chars of response: {out.stdout[:300]}")
    raw_chunks = []
    for it in idx.get("items", []):
        if it.get("type") != "file": continue
        if not it.get("name", "").endswith(".nc"): continue
        url = it.get("download") or (
            f"https://dap.ceda.ac.uk{it['path']}?download=1")
        raw_chunks.append((it["name"], url))
    raw_chunks.sort()
    if not raw_chunks:
        return raw_chunks
    # Probe alternative hosts: esgf.ceda.ac.uk (separate CEDA THREDDS,
    # healthy) and DKRZ ESGF replica.  Rewrite every chunk URL to the
    # first working host.  Falls back to the original dap.ceda.ac.uk
    # URLs only when both alternatives miss (BCC + CCSM4).
    sample_fname = raw_chunks[0][0]
    host = _probe_alt_host(institute, model, "past1000", member, var,
                             version, sample_fname)
    if host == "esgf-ceda":
        return [(fname,
                 _esgf_ceda_url(institute, model, "past1000", "Omon", member,
                                  var, version, fname))
                for (fname, _) in raw_chunks]
    if host == "dkrz":
        return [(fname,
                 _dkrz_url(institute, model, "past1000", "Omon", member,
                            var, version, fname))
                for (fname, _) in raw_chunks]
    return raw_chunks


def _download(url, dest, token, *, max_retries=6,
                initial_backoff_s=20):
    """Robust download via curl, with exponential backoff on
    transient 503 / connection errors.  Range-resumable.

    CEDA dap.ceda.ac.uk responds with HTTP 503 when an IP exceeds
    the burst threshold; the recovery is automatic but takes
    O(minute) per offence.  We honour Retry-After when given,
    else 20, 40, 80, ... seconds.
    """
    import time as _t
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return  # cached (>1 MB to skip incomplete)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    backoff = initial_backoff_s
    last_err = None
    for attempt in range(1, max_retries + 1):
        # Build curl argv cleanly; only add the Authorization header
        # if we have a token.  Earlier inline "-H ... if token else ..."
        # was a precedence bug that produced a stray empty URL arg and
        # caused curl rc=3 "Malformed input to a URL function".
        # We capture stderr (where curl's diagnostics live) and use
        # errors="replace" so a binary body never UTF-8-explodes.
        cmd = ["curl", "--silent", "--show-error", "--location",
               "--max-time", "900", "--connect-timeout", "30",
               "--write-out", "%{http_code}",
               "-C", "-"]
        # Only attach the CEDA bearer token for CEDA URLs.  DKRZ and
        # other ESGF replicas are anonymous and rejecting an unsolicited
        # Authorization header is at least cosmetically unwelcome.
        if token and "ceda.ac.uk" in url:
            cmd += ["-H", f"Authorization: Bearer {token}"]
        cmd += ["-o", str(tmp), url]
        out = subprocess.run(cmd, capture_output=True,
                              text=True, errors="replace",
                              timeout=960)
        # --write-out emits only the HTTP code on stdout.
        code = (out.stdout or "").strip()
        if out.returncode == 0 and tmp.exists() \
                and tmp.stat().st_size > 1_000_000:
            # success
            head = tmp.read_bytes()[:200]
            if b"<html" in head.lower() or b"signin" in head.lower():
                tmp.unlink()
                raise RuntimeError(
                    "CEDA returned an HTML login page instead of the "
                    "NetCDF.  Set CEDA_TOKEN to a valid access token "
                    "from https://services.ceda.ac.uk/access-token/ .")
            tmp.rename(dest)
            return
        # transient retry
        last_err = (out.returncode, code, out.stderr[:200])
        print(f"    attempt {attempt}/{max_retries}: rc={out.returncode} "
              f"http={code}  retry in {backoff}s")
        _t.sleep(backoff)
        backoff = min(backoff * 2, 600)  # cap at 10 min
    if tmp.exists(): tmp.unlink()
    raise RuntimeError(f"download exhausted retries for {url}: "
                         f"last={last_err}")


def _Q_windows_per_chunk(tos_path, zos_path, basin, year_start, year_end,
                           stride_yr):
    """Open tos+zos chunks (which may span multiple decades), apply Q on
    rolling 30-yr sub-windows entirely inside [year_start, year_end]."""
    tos = xr.open_dataset(tos_path, decode_times=True)["tos"]
    zos = xr.open_dataset(zos_path, decode_times=True)["zos"]
    yr_in = _year_of(tos["time"].values[0])
    yr_out = _year_of(tos["time"].values[-1])
    Qs = []; centres = []
    seg = max(year_start, yr_in)
    while seg + WINDOW_YEARS - 1 <= min(year_end, yr_out):
        s, e = seg, seg + WINDOW_YEARS - 1
        try:
            Q, n = _Q_for_window(_slice_by_year(tos, s, e),
                                   _slice_by_year(zos, s, e), basin)
            Qs.append(Q); centres.append(s + WINDOW_YEARS // 2)
        except Exception as ex:
            print(f"    WINDOW {s}-{e} FAILED ({type(ex).__name__})")
            Qs.append(float("nan")); centres.append(s + WINDOW_YEARS // 2)
        seg += stride_yr
    return Qs, centres


def _Q_windows_mfdataset(tos_paths, zos_paths, basin, year_start, year_end,
                          stride_yr):
    """Overlap-tolerant variant of _Q_windows_per_chunk.

    Opens lists of tos and zos paths as multi-file datasets (combine='by_coords'),
    then iterates 30-yr windows from year_start to year_end inclusive of the
    overlap between the two datasets' time coverage.
    """
    tos = xr.open_mfdataset(
        sorted(map(str, tos_paths)), combine="by_coords",
        decode_times=True, parallel=False,
    )["tos"]
    zos = xr.open_mfdataset(
        sorted(map(str, zos_paths)), combine="by_coords",
        decode_times=True, parallel=False,
    )["zos"]
    yr_in = max(
        _year_of(tos["time"].values[0]),
        _year_of(zos["time"].values[0]),
        year_start,
    )
    yr_out = min(
        _year_of(tos["time"].values[-1]),
        _year_of(zos["time"].values[-1]),
        year_end,
    )
    print(f"    mfdataset window: {yr_in}..{yr_out}  "
          f"(tos {len(tos_paths)} files, zos {len(zos_paths)} files)")
    Qs = []; centres = []
    seg = yr_in
    while seg + WINDOW_YEARS - 1 <= yr_out:
        s, e = seg, seg + WINDOW_YEARS - 1
        try:
            Q, n = _Q_for_window(_slice_by_year(tos, s, e),
                                   _slice_by_year(zos, s, e), basin)
            Qs.append(Q); centres.append(s + WINDOW_YEARS // 2)
        except Exception as ex:
            print(f"    WINDOW {s}-{e} FAILED ({type(ex).__name__})")
            Qs.append(float("nan")); centres.append(s + WINDOW_YEARS // 2)
        seg += stride_yr
    return Qs, centres


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--institute", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--member", default="r1i1p1")
    ap.add_argument("--tos-version", required=True)
    ap.add_argument("--zos-version", required=True)
    ap.add_argument("--basin", default="atlantic", choices=list(BASINS.keys()))
    ap.add_argument("--stride-yr", type=int, default=10)
    args = ap.parse_args()

    print("=" * 70)
    print(f" CEDA past1000 ingest  model={args.model}  member={args.member}  "
          f"basin={args.basin}")
    print("=" * 70)

    token = _ceda_token()
    if not token:
        print("WARNING: no CEDA bearer token in $CEDA_TOKEN or "
              "~/.ceda_token; downloads will be redirected to "
              "auth.ceda.ac.uk/account/signin and fail.")
    else:
        print(f"using CEDA bearer token "
              f"(len={len(token)}, last 4 = ...{token[-4:]})")

    out_json = OUT_DIR / f"{args.model}_{args.member}_{args.basin}.json"
    if out_json.exists():
        print(f"  result already cached: {out_json}")
        return

    # Discover chunk URLs for tos and zos
    print("\n[tos] enumerating chunks ...")
    tos_chunks = _enumerate_chunks(args.institute, args.model,
                                     args.member, "tos", args.tos_version,
                                     token)
    print(f"  {len(tos_chunks)} tos chunks")
    print("[zos] enumerating chunks ...")
    zos_chunks = _enumerate_chunks(args.institute, args.model,
                                     args.member, "zos", args.zos_version,
                                     token)
    print(f"  {len(zos_chunks)} zos chunks")

    # Pair chunks by year range (extract from filename)
    import re
    def _year_range(fname):
        m = re.search(r"_(\d{6})-(\d{6})\.nc", fname)
        if not m: return None
        ys = int(m.group(1)[:4]); ye = int(m.group(2)[:4])
        return (ys, ye)

    tos_by_yr = {_year_range(f): (f, u) for f, u in tos_chunks if _year_range(f)}
    zos_by_yr = {_year_range(f): (f, u) for f, u in zos_chunks if _year_range(f)}
    common = sorted(set(tos_by_yr) & set(zos_by_yr))
    all_Q = []; all_centres = []
    if not common:
        # Fallback: tos and zos chunks have different temporal boundaries
        # (e.g. GISS-E2-R has tos in 50-yr chunks, zos in 25-yr; CCSM4 has
        # 500-yr boundaries differing by 50 yr).  Download every chunk for
        # both vars, open as multi-file datasets, run windows over the
        # overlap of the two coverages.  Disk footprint stays bounded
        # because the chunks are small (~10-300 MB each).
        print(f"\n  no exact-tuple chunk pairs (tos={len(tos_by_yr)}, "
              f"zos={len(zos_by_yr)}); switching to mfdataset fallback.")
        tos_paths, zos_paths = [], []
        try:
            for fname, url in tos_chunks:
                p = RAW_DIR / fname
                _download(url, p, token)
                tos_paths.append(p)
            for fname, url in zos_chunks:
                p = RAW_DIR / fname
                _download(url, p, token)
                zos_paths.append(p)
            print(f"  downloaded {len(tos_paths)} tos + {len(zos_paths)} zos")
            try:
                Qs, centres = _Q_windows_mfdataset(
                    tos_paths, zos_paths, args.basin,
                    year_start=850, year_end=1849,
                    stride_yr=args.stride_yr,
                )
                for Q, c in zip(Qs, centres):
                    print(f"  Q centre={c}  Q={Q:+.3f}")
                all_Q.extend(Qs); all_centres.extend(centres)
            except Exception as ex:
                print(f"  mfdataset Q-windows FAILED: {ex}")
        finally:
            for p in tos_paths + zos_paths:
                p.unlink(missing_ok=True)
            print("  raw deleted (fallback path)")
        # jump straight to summary save
        common = []
    else:
        print(f"\n  {len(common)} matched century-chunks: "
              f"{common[0]}..{common[-1]}")

    for yr_range in common:
        ys, ye = yr_range
        print(f"\n[{ys}-{ye}]")
        tos_fname, tos_url = tos_by_yr[yr_range]
        zos_fname, zos_url = zos_by_yr[yr_range]
        tos_path = RAW_DIR / tos_fname
        zos_path = RAW_DIR / zos_fname
        t0 = time.time()
        try:
            _download(tos_url, tos_path, token)
            print(f"  tos: {tos_path.stat().st_size / 1e6:.1f} MB")
            _download(zos_url, zos_path, token)
            print(f"  zos: {zos_path.stat().st_size / 1e6:.1f} MB  ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"  download FAILED: {e}")
            for p in (tos_path, zos_path):
                if p.exists(): p.unlink()
            continue
        t1 = time.time()
        Qs, centres = _Q_windows_per_chunk(
            tos_path, zos_path, args.basin,
            year_start=ys, year_end=ye, stride_yr=args.stride_yr,
        )
        for Q, c in zip(Qs, centres):
            print(f"  Q centre={c}  Q={Q:+.3f}")
        all_Q.extend(Qs); all_centres.extend(centres)
        # Delete raw files to keep peak disk low
        tos_path.unlink(missing_ok=True)
        zos_path.unlink(missing_ok=True)
        print(f"  raw deleted ({time.time() - t1:.0f}s processing)")

    # Save summary
    arr = np.asarray(all_Q, dtype=float)
    arr_f = arr[np.isfinite(arr)]
    summary = dict(
        source="ceda_cmip5_past1000",
        institute=args.institute, model=args.model,
        member=args.member, basin=args.basin,
        n_windows=int(len(all_Q)),
        Q=all_Q,
        year_centres=all_centres,
        Q_mean=float(arr_f.mean()) if arr_f.size else None,
        Q_sd=float(arr_f.std()) if arr_f.size else None,
        timestamp=int(time.time()),
    )
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")
    if summary['Q_mean'] is not None:
        print(f"  Q mean={summary['Q_mean']:+.3f} "
              f"sd={summary['Q_sd']:.3f}")
    else:
        print("  no finite Q values produced")
    print(f"  n_windows={summary['n_windows']}")


if __name__ == "__main__":
    main()
