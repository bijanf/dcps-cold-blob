"""Parse a running grand-ensemble log to build a partial JSON in the same
schema that `holocene_q_grand_ensemble.py` writes at the end.

Reads logs/grand_<model>_<exp>_<basin>.log (parsed from the log lines
``<exp_short> <start>-<end>  Q=±X  n=Y  (...)``) and emits a
``partial_<model>_<exp>_<basin>.json`` that the plot script can read
just like a finished JSON.  Useful while the long-running ensemble
is still streaming.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT


HOL_DIR = CACHE_DIR / "holocene_exit"

RE_OPEN = re.compile(
    r"open (?P<model>\S+) (?P<exp>\S+) member=(?P<member>\S+)$"
)
# matches '    1pct 1850-1879  Q=+0.064  n=1041  (28s)'
RE_Q = re.compile(
    r"^\s*(?P<exp_short>\S+)\s+(?P<start>\d+)-(?P<end>\d+)\s+"
    r"Q=(?P<Q>[-+]?\d+\.\d+)\s+n=(?P<n>\d+)"
)


def parse_log(log_path: Path):
    """Return (members, year_starts, per_member_Q) where per_member_Q is a
    list-of-list aligned with members and year_starts."""
    members = []
    per_member = []     # list[dict[start_year -> Q]]
    current_member = None
    cur_dict = None
    for line in log_path.read_text().splitlines():
        m_open = RE_OPEN.search(line)
        if m_open:
            current_member = m_open.group("member")
            cur_dict = {}
            members.append(current_member)
            per_member.append(cur_dict)
            continue
        m_q = RE_Q.search(line)
        if m_q and cur_dict is not None:
            s = int(m_q.group("start"))
            cur_dict[s] = float(m_q.group("Q"))
    # all year_starts seen
    all_starts = sorted({s for d in per_member for s in d})
    arr = np.full((len(members), len(all_starts)), np.nan, dtype=float)
    for i, d in enumerate(per_member):
        for j, s in enumerate(all_starts):
            if s in d:
                arr[i, j] = d[s]
    return members, all_starts, arr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--basin", default="atlantic")
    ap.add_argument("--window-years", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    members, starts, arr = parse_log(args.log)
    if not members:
        raise SystemExit(f"no members found in {args.log}")
    centres = [s + args.window_years // 2 for s in starts]
    mean_Q = np.nanmean(arr, axis=0)
    median_Q = np.nanmedian(arr, axis=0)
    p10 = np.nanpercentile(arr, 10, axis=0)
    p90 = np.nanpercentile(arr, 90, axis=0)

    out = dict(
        model=args.model, experiment=args.experiment, basin=args.basin,
        n_members_used=int(len(members)),
        members=members,
        window_years=args.window_years,
        year_centres=[int(c) for c in centres],
        mean_Q=[float(x) for x in mean_Q],
        median_Q=[float(x) for x in median_Q],
        p10_Q=[float(x) for x in p10],
        p90_Q=[float(x) for x in p90],
        per_member_Q=[[float(x) for x in row] for row in arr],
        partial=True,
    )
    out_path = args.out or (
        HOL_DIR / f"grand_{args.model}_{args.experiment}_{args.basin}.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"  members: {len(members)}; year centres: {len(centres)} "
          f"({centres[0]}-{centres[-1]})")
    print(f"  ensemble median (first/last): {median_Q[0]:+.3f} / "
          f"{median_Q[-1]:+.3f}")


if __name__ == "__main__":
    main()
