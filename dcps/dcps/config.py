"""Paths and processing parameters for the DCPS Phase 1 pipeline."""

from __future__ import annotations

from pathlib import Path

# --- External data backbone (ARDP cache) ------------------------------------
ARDP_ROOT = Path.home() / "Documents" / "AMOC_renalysis" / "data"
ORAS5_DIR = ARDP_ROOT / "oras5"
RAPID_FILE = ARDP_ROOT / "external" / "rapid_moc_monthly.nc"

# --- This-package outputs ---------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PKG_ROOT / "cache"
FIGURES_DIR = PKG_ROOT / "figures"
PHASE1_OUTPUT = CACHE_DIR / "phase1_oras5_NA_2deg.nc"

# --- Analysis domain --------------------------------------------------------
# North Atlantic basin per the manuscript Methods.
LAT_MIN, LAT_MAX = 0.0, 75.0     # deg N
LON_MIN, LON_MAX = -80.0, 0.0    # deg E (i.e. 80W..0)

# --- Time window ------------------------------------------------------------
# ORAS5 SST/SSH cache is contiguous 1958-2014, then a gap, then 2022-2023.
# Phase 1 uses the contiguous segment so bandpass/Hilbert have no internal hole.
# ORAS5 AMOC and observation-constrained ocean state are unreliable pre-2000
# (Argo became operational ~2000-2003; pre-Argo ocean reanalysis is sparsely
# constrained, especially in the deep Atlantic). We restrict the DCPS analysis
# to the Argo-era observationally-constrained period.
TIME_START = "2000-01-01"
TIME_END = "2023-12-31"

# --- Coarse-grained target grid --------------------------------------------
GRID_DEG = 2.0    # 2 deg x 2 deg cells

# --- Anomaly preprocessing parameters --------------------------------------
BANDPASS_LO_YEARS = 1.0    # short-period corner
BANDPASS_HI_YEARS = 10.0   # long-period corner
BUTTER_ORDER = 4           # forward-backward Butterworth
EDGE_TRIM_MONTHS = 6       # to be applied after Hilbert in Phase 2

# Variables to extract from ORAS5
ORAS5_VARS = {
    "sst": "sosstsst",
    "ssh": "sossheig",
}
