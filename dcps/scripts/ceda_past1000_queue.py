"""Sequentially ingest every PMIP3 past1000 model on CEDA that has
both tos and zos at Omon resolution.

Runs ceda_past1000_ingest.py as a subprocess per (institute, model,
member) target.  Sleeps INTER_MODEL_SLEEP_S between successive
models so CEDA's per-IP burst throttle on dap.ceda.ac.uk can reset.

Resumable: per-model JSONs in dcps/cache/cmip5_past1000/ are checked
first; targets whose JSON exists with at least one finite Q window
are skipped.  Empty / failed JSONs are removed first so the ingest
can retry.

Output: dcps/cache/cmip5_past1000/<model>_<member>_<basin>.json per
target; aggregate log at logs/ceda_past1000_queue.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dcps.config import CACHE_DIR, PKG_ROOT


INGEST_SCRIPT = (PKG_ROOT / "scripts" / "ceda_past1000_ingest.py")
OUT_DIR = CACHE_DIR / "cmip5_past1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTER_MODEL_SLEEP_S = 90
MAX_PASSES = 3                  # retry the whole queue this many times

TARGETS = [
    # (institute, model, member, tos_version, zos_version)
    # Five structurally distinct PMIP3 past1000 models
    ("MPI-M",    "MPI-ESM-P",   "r1i1p1",    "20120625", "20120625"),
    ("BCC",      "bcc-csm1-1",  "r1i1p1",    "20120606", "20120606"),
    ("NCAR",     "CCSM4",       "r1i1p1",    "20121128", "20121128"),
    ("MIROC",    "MIROC-ESM",   "r1i1p1",    "20130712", "20111228"),

    # GISS-E2-R single-forcing physics variants (Schmidt 2014 GMD).
    # Same model, different p-number = different combination of
    # transient Holocene forcings.  Together they give forced-
    # attribution of any Q shift.
    # Per-member version dates probed live from CEDA catalog 2026-05-21.
    ("NASA-GISS","GISS-E2-R",   "r1i1p121",  "20120531", "20120531"),  # all forcings
    ("NASA-GISS","GISS-E2-R",   "r1i1p122",  "20120605", "20120605"),  # natural-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p1221", "20130522", "20130522"),  # natural-only variant
    ("NASA-GISS","GISS-E2-R",   "r1i1p123",  "20120907", "20120907"),  # anthrop-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p124",  "20120329", "20120329"),  # solar-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p125",  "20120323", "20120323"),  # volcanic-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p126",  "20120913", "20120913"),  # orbital-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p127",  "20120824", "20120824"),  # GHG-only
    ("NASA-GISS","GISS-E2-R",   "r1i1p128",  "20121213", "20121213"),  # land-use-only
]


def _cache_path(model, member, basin):
    return OUT_DIR / f"{model}_{member}_{basin}.json"


def _cache_is_good(p: Path) -> bool:
    if not p.exists(): return False
    try:
        d = json.loads(p.read_text())
        return bool(d.get("Q")) and d.get("Q_mean") is not None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin", default="atlantic")
    ap.add_argument("--stride-yr", type=int, default=10)
    args = ap.parse_args()

    print("=" * 70)
    print(f" CEDA past1000 queue   basin={args.basin}   N_targets={len(TARGETS)}"
          f"   max_passes={MAX_PASSES}")
    print(f" inter-model pause     {INTER_MODEL_SLEEP_S} s")
    print("=" * 70)

    env = os.environ.copy()
    if "CEDA_TOKEN" not in env:
        tok_path = Path.home() / ".ceda_token"
        if tok_path.exists():
            env["CEDA_TOKEN"] = tok_path.read_text().strip()
        else:
            print("ERROR: no CEDA token; aborting.")
            sys.exit(2)

    for pass_idx in range(1, MAX_PASSES + 1):
        remaining = [(inst, model, mem, t_v, z_v)
                     for (inst, model, mem, t_v, z_v) in TARGETS
                     if not _cache_is_good(_cache_path(model, mem,
                                                         args.basin))]
        if not remaining:
            print("\nall targets cached; queue complete.")
            break
        print(f"\n=== pass {pass_idx}/{MAX_PASSES}  remaining "
              f"{len(remaining)}/{len(TARGETS)} ===")
        for i, (inst, model, mem, tos_v, zos_v) in enumerate(remaining):
            target_id = f"{model}_{mem}_{args.basin}"
            cache = _cache_path(model, mem, args.basin)
            print(f"\n[pass {pass_idx} {i + 1}/{len(remaining)}]  "
                  f"{target_id}")
            if _cache_is_good(cache):
                print("  cached & nonempty -> skip")
                continue
            if cache.exists():
                print(f"  removing previous empty cache: {cache}")
                cache.unlink()
            cmd = [
                sys.executable, "-u", str(INGEST_SCRIPT),
                "--institute", inst, "--model", model, "--member", mem,
                "--tos-version", tos_v, "--zos-version", zos_v,
                "--basin", args.basin,
                "--stride-yr", str(args.stride_yr),
            ]
            t0 = time.time()
            try:
                subprocess.run(cmd, env=env, check=False)
            except KeyboardInterrupt:
                print("interrupted by user; stopping queue.")
                return
            except Exception as e:
                print(f"  ingest subprocess died: "
                      f"{type(e).__name__}: {e}")
            print(f"  {target_id} done in {time.time() - t0:.0f}s")
            if i + 1 < len(remaining):
                print(f"  sleeping {INTER_MODEL_SLEEP_S}s to let "
                      f"CEDA throttle reset ...")
                time.sleep(INTER_MODEL_SLEEP_S)
        # Between passes, take a longer break so any whole-server
        # outage has time to recover.
        if pass_idx < MAX_PASSES:
            print(f"\n=== end of pass {pass_idx}; sleeping 300s "
                  f"before next pass ===")
            time.sleep(300)

    print("\n=== queue complete ===")
    for inst, model, mem, _, _ in TARGETS:
        cache = _cache_path(model, mem, args.basin)
        ok = _cache_is_good(cache)
        print(f"  {'OK' if ok else 'MISS'}  {model}_{mem}")


if __name__ == "__main__":
    main()
