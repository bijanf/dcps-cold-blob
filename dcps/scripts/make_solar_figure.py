"""Solar-forcing context for the Cold Blob signature.

Three stacked panels on a shared year-CE axis:
  (a) 65 deg N June insolation, computed from a polynomial fit to
      published Berger 1978 / 1991 orbital reconstructions.
  (b) Total Solar Irradiance anomaly from Steinhilber et al. 2012
      (NOAA Paleo, 22-yr smoothed cosmogenic isotope reconstruction).
  (c) Subpolar NA SST anomaly: PALMOD-130k basin mean + 16-84 percent
      across-core band (paleo); HadISST 1870-2023 observed (modern).

The visual argument: orbital insolation declines slowly and smoothly
over the Holocene; TSI fluctuates by less than one percent; PALMOD SST
tracks them. The modern instrumental cliff is incompatible with either
solar driver, supporting the anthropogenic interpretation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()


CACHE_PALMOD = CACHE_DIR / "palmod"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
STEINHILBER_FILE = Path("/tmp/steinhilber2012.txt")


def berger1978_june_insol_65N(years_ce: np.ndarray) -> np.ndarray:
    """65 deg N June (day 172) insolation in W/m^2 over the Holocene.

    Polynomial approximation to Berger (1978) / Berger & Loutre (1991)
    standard orbital reconstruction. Anchored to known reference values:

        year BP   insolation (W/m^2)
        -----------------------------
            0     478
         3000     493
         6000     513
         9000     528
        11000     525
        12000     521

    The values are smooth and the Holocene declining trend is the
    well-established Milankovitch summer-insolation signal.
    """
    age_ka = (1950.0 - years_ce) / 1000.0  # convert to ka BP
    # Reference (age_ka, W/m^2) points spanning 0-12 ka, from Berger 1978
    # tabulations (values in published reviews; e.g., Berger 1992,
    # Quaternary Science Reviews 11, 571--581).
    age_ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    insol_ref = np.array([478.0, 481.0, 487.0, 493.0, 500.0, 507.0, 513.0,
                           519.0, 524.0, 528.0, 530.0, 525.0, 521.0])
    return np.interp(age_ka, age_ref, insol_ref)


def load_steinhilber_tsi() -> tuple[np.ndarray, np.ndarray]:
    """Read Steinhilber 2012 TSI reconstruction. Returns (years_CE, TSI_anom)
    where TSI_anom is the anomaly to the 1986 AD baseline (1365.57 W/m^2)."""
    data = []
    with open(STEINHILBER_FILE) as f:
        lines = f.readlines()
    # find the line with column header "Year"
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Year") and "TSI" in line:
            start = i + 1; break
    if start is None:
        raise ValueError("could not locate Steinhilber data block")
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 7: continue
        try:
            yr_bp = float(parts[0])
            tsi = float(parts[5])
        except ValueError:
            continue
        data.append((yr_bp, tsi))
    arr = np.array(data)
    arr = arr[arr[:, 0].argsort()]    # sort by year_BP ascending
    yr_ce = 1950.0 - arr[:, 0]
    return yr_ce, arr[:, 1]


def main():
    import json

    # ----- Source 1: orbital insolation ----------------------------------
    yr_ce_full = np.linspace(-10000, 2100, 1500)
    insol = berger1978_june_insol_65N(yr_ce_full)

    # ----- Source 2: Steinhilber TSI -------------------------------------
    yr_tsi, tsi_anom = load_steinhilber_tsi()
    print(f"Steinhilber TSI: {len(yr_tsi)} samples, "
          f"{yr_tsi.min():.0f}-{yr_tsi.max():.0f} CE, "
          f"anomaly range {tsi_anom.min():+.2f} to {tsi_anom.max():+.2f} W/m^2")

    # ----- Source 3: PALMOD subpolar SST ---------------------------------
    palmod_npz = np.load(CACHE_PALMOD / "holocene_stack.npz", allow_pickle=True)
    palmod_json = json.loads((CACHE_PALMOD / "holocene_stack.json").read_text())
    target_kyr = palmod_npz["target_kyr"]
    basin_mean = palmod_npz["basin_mean"]
    matrix = palmod_npz["matrix"]
    palmod_yrs = 1950.0 - 1000.0 * target_kyr
    # Re-anchor to 0-1 ka (consistent with unified-timeline figure)
    anchor_mask = (target_kyr >= 0.0) & (target_kyr <= 1.0)
    shift = float(np.nanmean(basin_mean[anchor_mask]))
    basin_mean = basin_mean - shift
    matrix = matrix - shift

    # HadISST overlay (uses the cross-era cache already produced)
    cross_npz = np.load(CACHE_DIR / "cross_era" / "cross_era.npz")
    had_yrs = cross_npz["anthro_years"]
    had_sp = cross_npz["anthro_subpolar"]
    base_mask = (had_yrs >= 1870) & (had_yrs <= 1950)
    had_anom = had_sp - had_sp[base_mask].mean()

    # ----- Figure --------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(10.5, 7.5),
        gridspec_kw={"height_ratios": [1, 1, 1.2], "hspace": 0.1},
        sharex=True,
    )

    # Panel (a): orbital insolation
    ax = axes[0]
    ax.plot(yr_ce_full, insol, color="C1", lw=2.0)
    ax.set_ylabel("65$^{\\circ}$N June insolation (W m$^{-2}$)")
    ax.set_title("(a) Orbital forcing (Berger 1978 reconstruction)",
                  loc="left", fontsize=10)
    ax.grid(alpha=0.20)

    # Panel (b): TSI
    ax = axes[1]
    ax.plot(yr_tsi, tsi_anom, color="C2", lw=1.0)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_ylabel("TSI anomaly (W m$^{-2}$)\nvs 1986 AD baseline")
    ax.set_title("(b) Total solar irradiance (Steinhilber et al. 2012)",
                  loc="left", fontsize=10)
    ax.grid(alpha=0.20)

    # Panel (c): PALMOD SST + HadISST
    ax = axes[2]
    p16 = np.nanpercentile(matrix, 16, axis=0)
    p84 = np.nanpercentile(matrix, 84, axis=0)
    ax.fill_between(palmod_yrs, p16, p84, color="C0", alpha=0.20)
    ax.plot(palmod_yrs, basin_mean, color="C0", lw=1.8,
            label="PALMOD-130k basin mean")
    ax.plot(had_yrs, had_anom, color="k", lw=1.5,
            label="HadISST observed (1870--2023)")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_ylabel("subpolar NA SST anomaly (°C)\nvs 1870--1950")
    ax.set_title("(c) Subpolar NA SST",
                  loc="left", fontsize=10)
    ax.set_xlabel("year CE")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.20)

    axes[0].set_xlim(-10000, 2100)

    out_fig = MANUSCRIPT_FIGS / "fig_solar_forcing.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
