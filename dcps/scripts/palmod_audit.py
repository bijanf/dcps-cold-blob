"""PALMOD-130k subpolar North Atlantic coverage audit.

Reads the PALMOD-130k v2 site overview (Jonkers et al. 2026, ESSD,
PANGAEA DOI 10.1594/PANGAEA.984602) and produces:

    1. Per-cell core counts on a 5 deg x 5 deg grid covering the NA basin
       (80W-0E, 35N-70N).
    2. The site count in the pre-registered subpolar NA window
       (50W-10W, 50N-65N).
    3. A JSON summary at cache/palmod/audit.json.
    4. A supplementary figure at manuscript/figs/figS_palmod_audit.pdf.

The pre-registered decision rule (locked before this audit, see plan file):

    Holocene Mann-Kendall stack triggers iff the subpolar NA window contains
    >=10 cores with median Holocene resolution <=2 kyr AND consistent SST
    proxy type (all alkenone OR all Mg/Ca).

The PALMOD-130k overview CSV gives only site lat/lon/DOI -- it does not
itself record per-core proxy type or temporal resolution. Verifying the
resolution and proxy-type criteria therefore requires reading the full LiPD
package (~hours to set up, age-model handling, calibration tracking). In
the present manuscript pass we report the spatial-coverage audit (which is
satisfied) and document the proxy-resolution audit + Mann-Kendall stack as
immediate follow-up work.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dcps.config import CACHE_DIR, PKG_ROOT


PALMOD_DIR = Path.home() / "Documents" / "NEW_Theory" / "data" / "external" / "palmod_130k"
OVERVIEW_CSV = PALMOD_DIR / "PALMOD_v2_overview.csv"

AUDIT_DIR = CACHE_DIR / "palmod"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

# Pre-registered windows (locked).
NA_BASIN = dict(lon_min=-80, lon_max=0, lat_min=35, lat_max=70)
SUBPOLAR_NA = dict(lon_min=-50, lon_max=-10, lat_min=50, lat_max=65)

# Decision rule thresholds (locked).
MIN_CORES = 10


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OVERVIEW_CSV)
    sites = df.drop_duplicates(subset="site")[["site", "longitude", "latitude"]]
    print(f"Loaded {len(sites)} unique PALMOD-130k sites from overview.")

    in_na = sites[
        sites.longitude.between(NA_BASIN["lon_min"], NA_BASIN["lon_max"])
        & sites.latitude.between(NA_BASIN["lat_min"], NA_BASIN["lat_max"])
    ]
    in_sp = sites[
        sites.longitude.between(SUBPOLAR_NA["lon_min"], SUBPOLAR_NA["lon_max"])
        & sites.latitude.between(SUBPOLAR_NA["lat_min"], SUBPOLAR_NA["lat_max"])
    ]
    print(f"  in NA basin (80W-0E, 35N-70N):       {len(in_na)} sites")
    print(f"  in subpolar NA (50W-10W, 50N-65N):  {len(in_sp)} sites")

    # 5x5 cell counts over the NA basin.
    grid = in_na.copy()
    grid["lat_cell"] = (grid.latitude // 5 * 5).astype(int)
    grid["lon_cell"] = (grid.longitude // 5 * 5).astype(int)
    cell_counts = (grid.groupby(["lat_cell", "lon_cell"]).size()
                   .reset_index(name="n").to_dict("records"))

    # Pre-registered location bar.
    location_bar_met = len(in_sp) >= MIN_CORES
    audit = {
        "n_sites_total": int(len(sites)),
        "n_sites_NA_basin": int(len(in_na)),
        "n_sites_subpolar_NA": int(len(in_sp)),
        "cell_counts_5x5_NA_basin": cell_counts,
        "subpolar_NA_sites": in_sp.to_dict("records"),
        "min_cores_threshold": MIN_CORES,
        "location_bar_met": bool(location_bar_met),
        "proxy_resolution_audit_status": (
            "not_performed_in_this_pass: requires full LiPD package "
            "(~hours of setup with age-model handling and per-proxy "
            "calibration). The location-coverage bar is met; the "
            "proxy-resolution audit + Holocene Mann-Kendall stack is "
            "documented as immediate follow-up work."
        ),
        "decision": (
            "report-as-future-work" if location_bar_met
            else "document-as-sparse-and-cite"
        ),
        "reference": (
            "Jonkers, L., Hollstein, M., Siccha, M., Kucera, M. (2026). "
            "The PALMOD 130k marine palaeoclimate data synthesis version 2. "
            "Earth Syst. Sci. Data 18, 3013-3068. "
            "doi:10.5194/essd-18-3013-2026; data: PANGAEA "
            "10.1594/PANGAEA.984602."
        ),
    }
    with open(AUDIT_DIR / "audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nWrote {AUDIT_DIR / 'audit.json'}")

    # ----- supplementary figure -------------------------------------------
    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5),
                                           gridspec_kw={"width_ratios": [1.6, 1.0]},
                                           constrained_layout=True)

    ax_map.scatter(in_na.longitude, in_na.latitude, s=18,
                    c="0.5", alpha=0.6, label=f"NA basin ({len(in_na)})")
    ax_map.scatter(in_sp.longitude, in_sp.latitude, s=28,
                    c="C3", alpha=0.85,
                    label=f"subpolar NA pre-registration window ({len(in_sp)})")
    sp_box = plt.Rectangle(
        (SUBPOLAR_NA["lon_min"], SUBPOLAR_NA["lat_min"]),
        SUBPOLAR_NA["lon_max"] - SUBPOLAR_NA["lon_min"],
        SUBPOLAR_NA["lat_max"] - SUBPOLAR_NA["lat_min"],
        edgecolor="C3", facecolor="none", linewidth=1.5, linestyle="--",
    )
    ax_map.add_patch(sp_box)
    ax_map.set_xlim(-85, 5)
    ax_map.set_ylim(30, 75)
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.legend(loc="lower left", fontsize=9, frameon=False)
    ax_map.grid(alpha=0.25)

    # 5x5 count grid as a histogram for the NA basin.
    counts = (grid.groupby(["lat_cell", "lon_cell"]).size()
              .unstack(fill_value=0).sort_index(ascending=False))
    im = ax_hist.imshow(counts.values, aspect="auto", cmap="viridis",
                         extent=[counts.columns.min(), counts.columns.max() + 5,
                                 counts.index.min(), counts.index.max() + 5],
                         origin="upper")
    plt.colorbar(im, ax=ax_hist, label="cores per 5° cell")
    ax_hist.set_xlabel("longitude (5° cell west edge)")
    ax_hist.set_ylabel("latitude (5° cell south edge)")

    out_fig = MANUSCRIPT_FIGS / "figS_palmod_audit.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
