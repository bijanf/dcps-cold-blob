"""Cross-prediction test of the Quiescence law on NVIDIA DLESyM v1 ERA5.

Registered as prediction P6 in ``PRE_REGISTRATION_P6.md`` at commit
``7960f9f83bd332177f24b7dbc5390aad02d16c74``.  Full execution is
pending an A100 allocation on the institutional cluster; this module
defines the contract that the future implementation must satisfy and
is importable without invoking any GPU-side code path.

The protocol is:

1. Autoregressively roll out DLESyM from an ERA5 initial condition
   at 2010-01-01 for ``years`` simulated years (>= 30 for the
   registered decision rule).
2. Regrid the HEALPix monthly SST output to the canonical 2-degree
   basin grid expected by ``multi_basin_quiescence.py``.
3. Run the unchanged Quiescence pipeline on the DLESyM SST.
4. Cross-predict against the observed GLORYS12 climatological EKE
   field and report ``Q_X`` with its spatial-block-permutation null
   plus the non-linear fit of ``(1 + tau * EKE) ** -0.5``.
"""

from __future__ import annotations

from typing import Any

# Reused entry points from the unchanged P1-P5 pipeline.  Each import is
# referenced by the deferred implementation; keeping them at the top of
# the module documents the reuse contract and lets a static check confirm
# the entry points still exist on the registered commit.
from multi_basin_quiescence import (  # noqa: F401
    BASINS,
    basin_target_grid,
    instantaneous_phase,
    local_r_mean,
    preprocess_anomaly,
)

from dcps.spatial_stats import spatial_block_permutation  # noqa: F401


def rollout_dlesym(years: int, ic_date: str = "2010-01-01") -> Any:
    """Autoregressively roll out NVIDIA DLESyM v1 ERA5 for ``years`` years.

    Returns monthly SST on the native HEALPix grid as an ``xarray.Dataset``
    with dimensions ``(time, ncells)`` and a single data variable ``sst``.
    Implementation pending A100 allocation.
    """
    raise NotImplementedError(
        "P6 rollout pending DLESyM execution on institutional A100"
    )


def healpix_to_basin_grid(dlesym_sst: Any, basin: str = "NA") -> Any:
    """Regrid HEALPix DLESyM SST to the canonical 2-degree basin grid.

    Uses area-weighted box averaging via ``healpy.pixelfunc.ang2pix`` and
    cosine-latitude weighting.  The output NetCDF schema matches the
    ORAS5 loader output expected by ``multi_basin_quiescence.py``:
    dimensions ``(time, lat, rlon)``, coordinate names matching
    ``basin_target_grid(basin)``.  Implementation pending.
    """
    raise NotImplementedError(
        "P6 HEALPix-to-basin regrid pending A100 rollout output"
    )


def cross_prediction_test(
    dlesym_sst_basin: Any,
    glorys12_eke_clim: Any,
    basin: str = "NA",
) -> dict[str, float]:
    """Compute the P6 cross-product Quiescence Index and decision metrics.

    Returns a dictionary with keys ``Q_X``, ``p_perm``, ``tau_fit``,
    ``tau_obs_NA``, ``chi2_nu``, and ``verdict`` (one of ``"supported"``,
    ``"falsified"``, or ``"deferred"``).  Implementation pending.
    """
    raise NotImplementedError(
        "P6 cross-prediction test pending DLESyM rollout and regrid"
    )


def main() -> int:
    """Print the registered protocol and exit successfully without GPU."""
    print(
        "P6 protocol registered at commit "
        "7960f9f83bd332177f24b7dbc5390aad02d16c74; see PRE_REGISTRATION_P6.md."
    )
    print(
        "Pending: 30-yr DLESyM rollout on institutional A100, "
        "then HEALPix-to-2deg regrid, then unchanged Quiescence pipeline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
