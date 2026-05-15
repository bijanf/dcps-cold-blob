"""Aggregate all P6 verdict JSONs into a single summary CSV + markdown."""
from __future__ import annotations

import json
from pathlib import Path
import sys

OUT_DIR = Path("/p/projects/poem/fallah/dlesym_p6_out")
CSV_OUT = Path("/p/projects/poem/fallah/p6_bundle/summary.csv")
MD_OUT = Path("/p/projects/poem/fallah/p6_bundle/summary.md")
FIELDS = ("tag", "Q_X", "p_perm", "tau_fit", "tau_obs_NA",
          "chi2_nu", "n_cells", "drift_K_per_year", "verdict")


def parse_tag(filename: str) -> str:
    # p6_verdict_<tag>.json → <tag>
    stem = Path(filename).stem
    if stem.startswith("p6_verdict_"):
        return stem[len("p6_verdict_") :]
    return stem


def main() -> int:
    rows = []
    for p in sorted(OUT_DIR.glob("p6_verdict_*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
            continue
        tag = parse_tag(p.name)
        row = {k: d.get(k) for k in FIELDS}
        row["tag"] = tag
        rows.append(row)

    if not rows:
        print("no verdicts found", file=sys.stderr)
        return 1

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w") as fh:
        fh.write(",".join(FIELDS) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r[k])
                              for k in FIELDS) + "\n")
    print(f"wrote {CSV_OUT} ({len(rows)} rows)")

    # Markdown summary
    lines = ["# P6 verdict ensemble summary\n",
             f"{len(rows)} realizations.\n",
             "| tag | Q_X | p_perm | tau_fit | chi2_nu | drift | verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(v, p):
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{p}f}"
            except (TypeError, ValueError):
                return str(v)
        lines.append(
            f"| {r['tag']} | {fmt(r.get('Q_X'),3)} | {fmt(r.get('p_perm'),3)} "
            f"| {fmt(r.get('tau_fit'),1)} | {fmt(r.get('chi2_nu'),2)} "
            f"| {fmt(r.get('drift_K_per_year'),4)} | {r.get('verdict', '?')} |"
        )

    # Ensemble stats by class (ic_*, a*o*, basin_*)
    import statistics
    def num(values):
        return [float(v) for v in values if v is not None]

    def cohort(prefix, label):
        sub = [r for r in rows if r["tag"].startswith(prefix)]
        if not sub:
            return None
        qx = num(r.get("Q_X") for r in sub)
        tau = num(r.get("tau_fit") for r in sub)
        chi = num(r.get("chi2_nu") for r in sub)
        out_lines = [f"\n## {label} cohort (n={len(sub)})\n"]
        if qx:
            out_lines.append(
                f"- Q_X: median={statistics.median(qx):.3f}, "
                f"mean={statistics.mean(qx):.3f}, "
                f"range=[{min(qx):.3f}, {max(qx):.3f}]"
            )
        if tau:
            out_lines.append(
                f"- tau_fit: median={statistics.median(tau):.1f}, "
                f"mean={statistics.mean(tau):.1f}, "
                f"range=[{min(tau):.1f}, {max(tau):.1f}]"
            )
        if chi:
            out_lines.append(
                f"- chi2_nu: median={statistics.median(chi):.2f}, "
                f"mean={statistics.mean(chi):.2f}, "
                f"range=[{min(chi):.2f}, {max(chi):.2f}]"
            )
        verdicts = [r.get("verdict") for r in sub]
        from collections import Counter
        c = Counter(verdicts)
        out_lines.append(f"- verdicts: {dict(c)}")
        return "\n".join(out_lines)

    for prefix, label in [("ic_", "IC ensemble (Tier 1A)"),
                          ("a", "DLESyM member ensemble (Tier 2D)"),
                          ("basin_", "Multi-basin (Tier 2E)")]:
        s = cohort(prefix, label)
        if s:
            lines.append(s)

    MD_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
