"""End-to-end Phase 2 pipeline.

Inputs:  cache/phase1_oras5_NA_2deg.nc  (Phase 1 cache)
Outputs: cache/phase2_phases.nc         (per-node phase + amplitude, edge-trimmed)
         cache/phase2_R.nc              (global R(t), local r(x,t), surrogate band)

Default surrogate ensemble: 200 (a compromise between Methods' M=1000 target and
turnaround time; can be raised via --surrogates).
"""

from __future__ import annotations

import argparse
import time

import xarray as xr

from dcps.config import CACHE_DIR, PHASE1_OUTPUT
from dcps.order_parameter import global_R, global_R_pooled, local_r
from dcps.phase import analytic_signal, edge_trim
from dcps.surrogates import surrogate_R_envelope


PHASE2_PHASES = CACHE_DIR / "phase2_phases.nc"
PHASE2_R = CACHE_DIR / "phase2_R.nc"


def main(n_surrogates: int, local_radius_km: float) -> None:
    print("Phase 2 driver:")
    print(f"  input: {PHASE1_OUTPUT}")
    print(f"  surrogates: {n_surrogates}")
    print(f"  local-r radius: {local_radius_km} km")

    ds = xr.open_dataset(PHASE1_OUTPUT)

    # ---- 1. Hilbert per cell + edge trim --------------------------------------
    out_phases = {}
    for var in ("sst_anom", "ssh_anom"):
        tic = time.time()
        z, amp, phi = analytic_signal(ds[var])
        z = edge_trim(z); amp = edge_trim(amp); phi = edge_trim(phi)
        print(f"  [{var}] Hilbert + 6mo edge trim in {time.time() - tic:.1f}s "
              f"(time {phi.sizes['time']})")
        prefix = var.split("_")[0]                 # "sst" or "ssh"
        out_phases[f"{prefix}_amp"] = amp.rename(f"{prefix}_amp")
        out_phases[f"{prefix}_phase"] = phi.rename(f"{prefix}_phase")

    phases_ds = xr.Dataset(out_phases)
    phases_ds.attrs["title"] = "DCPS Phase 2: instantaneous amplitude + phase per node"
    phases_ds.attrs["edge_trim_months"] = 6
    phases_ds.attrs["source"] = str(PHASE1_OUTPUT)
    enc = {v: {"zlib": True, "complevel": 4} for v in phases_ds.data_vars}
    phases_ds.to_netcdf(PHASE2_PHASES, encoding=enc)
    print(f"  wrote {PHASE2_PHASES} ({PHASE2_PHASES.stat().st_size / 1e6:.1f} MB)")

    # ---- 2. Global R(t) for SST, SSH, pooled ----------------------------------
    R_sst = global_R(phases_ds.sst_phase).rename("R_sst")
    R_ssh = global_R(phases_ds.ssh_phase).rename("R_ssh")
    R_pooled = global_R_pooled(phases_ds.sst_phase, phases_ds.ssh_phase).rename("R_pooled")
    print(f"  global R: SST mean={float(R_sst.mean()):.3f}, SSH mean={float(R_ssh.mean()):.3f}, "
          f"pooled mean={float(R_pooled.mean()):.3f}")

    # ---- 3. Local r(x, t) for SST and SSH -------------------------------------
    tic = time.time()
    r_sst = local_r(phases_ds.sst_phase, radius_km=local_radius_km).rename("r_loc_sst")
    r_ssh = local_r(phases_ds.ssh_phase, radius_km=local_radius_km).rename("r_loc_ssh")
    print(f"  local r computed in {time.time() - tic:.1f}s")

    # ---- 4. Surrogate null on the bandpassed input ----------------------------
    # Note: the surrogate operates on the *bandpassed input* (sst_anom from Phase 1)
    # not the phase, so that we preserve each node's spectrum exactly.
    print(f"  surrogate null (M={n_surrogates}) for SST")
    tic = time.time()
    bp_sst_trim = edge_trim(ds.sst_anom)
    R_null_sst = surrogate_R_envelope(
        bp_sst_trim, n_surrogates=n_surrogates, seed=42,
        quantiles=(0.025, 0.5, 0.975),
    ).rename("R_null_sst")
    print(f"     in {time.time() - tic:.1f}s")

    # ---- 5. Save R cache ------------------------------------------------------
    R_ds = xr.merge([R_sst, R_ssh, R_pooled, r_sst, r_ssh, R_null_sst])
    R_ds.attrs["title"] = "DCPS Phase 2: global and local Kuramoto order parameters"
    R_ds.attrs["local_r_radius_km"] = float(local_radius_km)
    R_ds.attrs["surrogate_n"] = int(n_surrogates)
    R_ds.attrs["surrogate_method"] = "Schreiber phase-randomisation, FFT-based"
    enc = {v: {"zlib": True, "complevel": 4} for v in R_ds.data_vars}
    R_ds.to_netcdf(PHASE2_R, encoding=enc)
    print(f"  wrote {PHASE2_R} ({PHASE2_R.stat().st_size / 1e6:.1f} MB)")
    ds.close()
    phases_ds.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--surrogates", type=int, default=200,
                   help="Number of phase-randomised surrogates (Methods target: 1000)")
    p.add_argument("--radius-km", type=float, default=500.0,
                   help="Local-r window radius in km")
    args = p.parse_args()
    main(args.surrogates, args.radius_km)
