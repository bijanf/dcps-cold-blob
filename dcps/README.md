# dcps — Delay-Coupled Phase-Synchronization analysis of the North Atlantic

Phase 1 (data + preprocessing): subset ORAS5 reanalysis to the North Atlantic basin,
coarse-grain to 2°×2°, remove climatology, detrend, bandpass 1–10 yr.

Data backbone is the local ARDP cache at `~/Documents/AMOC_renalysis/data/`; this
package adds the phase-synchronization analysis layer.

## Layout

    dcps/
      io.py        ORAS5 + RAPID loaders
      regrid.py    ORCA025 -> 2x2 deg box-mean regrid
      anomaly.py   climatology + detrend + bandpass
      config.py    paths, parameters
    scripts/
      run_phase1.py       end-to-end pipeline
      validate_phase1.py  sanity-check figures
    cache/         (gitignored) pipeline outputs
    tests/

## Running

    pip install -e .
    python scripts/run_phase1.py          # ~minutes; writes cache/phase1_oras5_NA_2deg.nc
    python scripts/validate_phase1.py     # writes figures/phase1_validate.png
