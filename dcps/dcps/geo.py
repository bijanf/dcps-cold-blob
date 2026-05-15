"""Geographic helpers shared across the package.

A single canonical implementation of the great-circle distance lives here.
Previously this function was copy-pasted into ``spatial_stats``,
``order_parameter`` and ``scripts/multi_basin_quiescence``; that fanned-out
copy was a maintenance hazard flagged in the code review for the
manuscript revision.
"""
from __future__ import annotations

import numpy as np

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km.

    Accepts scalars or numpy arrays for each argument; broadcasts under
    standard numpy rules.  Inputs in degrees, output in km.
    """
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))
