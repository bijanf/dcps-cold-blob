# dcps-cold-blob

[![CI](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Reproducible analysis code for a pre-registered phase-synchronisation
study of the North Atlantic Cold Blob across the Holocene (PALMOD-130k
proxies), the instrumental era (HadISST 1870–2023), and CMIP6 future
projections through 2100.

This repository contains **only the analysis code, tests, and small
reference datasets** required to reproduce the pipelines. It does not
ship the manuscript, figure PDFs, or research notes.

---

## Layout

| Path | Contents |
|---|---|
| `dcps/dcps/` | Python package — phase fields, Kuramoto order parameter, winding-number topology, transfer-entropy (KSG) estimators, Nature-style figure helper |
| `dcps/scripts/` | End-to-end analysis pipelines (multi-basin Quiescence, multi-proxy unprecedentedness test, geostrophic-EKE mechanism test, CMIP6 emergent constraint, piControl detection-and-attribution, large-ensemble Q, past1000 forced-paleo, SSP scenario sweeps, fetch helpers) |
| `dcps/tests/` | pytest unit tests for the package |
| `quiescence_toolkit/` | Standalone Q-index toolkit: source modules, data fetchers (ERA5, HighResMIP), reproduction notebooks, JSON result tables |
| `data/external/` | Small reference datasets (Caesar 2021 multi-proxy `.xlsx`, Thornalley 2018 `.csv`, Steinhilber 2012 TSI, Moffa-Sánchez `.tab`). Bulk data (HadISST, PALMOD, ERA5, ORAS5, GLORYS12, …) is gitignored — fetch instructions below. |
| `data/external/p6/` | P6 cross-prediction verdict JSONs and summary tables |
| `.github/workflows/ci.yml` | GitHub Actions: pytest matrix (Python 3.11/3.12/3.13) + Ruff lint |

---

## Requirements

- **Python ≥ 3.11** (tested in CI on 3.11, 3.12, 3.13)
- **git** and **curl**
- System libraries for `cartopy` and `netCDF4`:
  - **Ubuntu / Debian**:
    ```bash
    sudo apt-get install -y libgeos-dev libproj-dev proj-data proj-bin \
                            libhdf5-dev libnetcdf-dev
    ```
  - **macOS** (Homebrew): `brew install geos proj hdf5 netcdf`
  - **conda** (cross-platform): `conda install -c conda-forge cartopy netcdf4 hdf5 proj geos`
- Optional credentials for the live data fetchers:
  - **CDS API** (Copernicus Climate Data Store) → `~/.cdsapirc`, for ERA5
  - **CEDA bearer token** → for the PMIP3 past1000 ingest
  - **Pangeo** (anonymous) for CMIP6 Zarr

---

## Quickstart

```bash
# 1. clone
git clone https://github.com/bijanf/dcps-cold-blob.git
cd dcps-cold-blob

# 2. (recommended) isolated env
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 3. install the package + dev extras (pulls in pytest + ruff)
pip install -e "./dcps[dev]"

# 4. smoke test: run the unit tests (~30 s)
pytest dcps/tests/ quiescence_toolkit/tests/ -q

# 5. fast end-to-end check that requires no external download
#    (synthetic data, ~1 s, exercises the heavy deps and a KSG TE call):
python -c "
from dcps import te_ksg, spatial_stats, regrid, geo, anomaly
import numpy as np
rng = np.random.default_rng(0)
y = rng.standard_normal(500)
x = 0.5*np.roll(y, 1) + 0.3*rng.standard_normal(500)
te = te_ksg.ksg_te(y, x, k=1, ell=1, k_nn=4)
print(f'KSG TE(y -> x) = {te:.3f}  (expect > 0)')
"
```

If `pytest` reports `23 passed` and the snippet prints a positive
KSG TE value, your install is healthy. Move on to fetching real data
below.

---

## Fetching external data

### Always free, no credentials

```bash
# HadISST 1870–2023 (Met Office) — used by the unprecedentedness pipelines
mkdir -p data/external/hadisst
curl -L https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz \
     -o data/external/hadisst/HadISST_sst.nc.gz
gunzip data/external/hadisst/HadISST_sst.nc.gz

# PALMOD-130k v2 (LiPDverse) — used by the Holocene-context pipelines
mkdir -p data/external/palmod_130k
curl -L https://lipdverse.org/PalMod/2_0_0/PalMod2_0_0.zip \
     -o data/external/palmod_130k/PalMod2_0_0.zip
unzip -d data/external/palmod_130k/lipd_unzipped \
     data/external/palmod_130k/PalMod2_0_0.zip
```

### Account required

```bash
# ERA5 monthly aggregates (NAO / AMO regression diagnostics) — needs ~/.cdsapirc
python dcps/scripts/fetch_era5.py

# PMIP3 past1000 forced-paleo via CEDA — needs CEDA bearer token in env
# (export CEDA_TOKEN=...)
python dcps/scripts/ceda_past1000_ingest.py

# CMIP6 piControl / historical / SSP via Pangeo (anonymous, no key needed)
python dcps/scripts/cmip6_picontrol_dna.py     # ~30 min
python dcps/scripts/cmip6_emergent_constraint.py
```

### Local-cache pointers

ORAS5, GLORYS12, ECCO, and RAPID are read from an external "ARDP"
cache. Point `dcps/dcps/config.py` (`ARDP_ROOT`) at wherever you keep
those files locally. The default expects
`~/Documents/AMOC_renalysis/data/`.

---

## Running the analyses

```bash
# Headline pipelines (need HadISST + PALMOD + ARDP cache)
python dcps/scripts/multi_basin_quiescence.py        # ~minutes
python dcps/scripts/winding_basin_test.py            # ~minutes
python dcps/scripts/cross_era_contrast.py            # ~seconds
python dcps/scripts/cold_blob_unprecedented.py       # ~seconds
python dcps/scripts/multi_proxy_unprecedented.py     # ~seconds
python dcps/scripts/eke_quiescence_test.py           # ~minutes
python dcps/scripts/cmip6_emergent_constraint.py     # ~seconds

# Revision-stage pipelines
python dcps/scripts/eke_quiescence_eddy_resolving.py # ~10 min  (GLORYS12 1/12° NA)
python dcps/scripts/nao_phase_regression.py          # ~5 min   (auto-downloads NAO/AMO)
python dcps/scripts/cmip6_picontrol_dna.py           # ~30 min  (Pangeo piControl)
python dcps/scripts/cold_blob_unprecedented_bootstrap.py  # ~seconds (B=1000)
python dcps/scripts/bandpass_sensitivity.py          # ~30 min  (3x3 sweep)
python dcps/scripts/pre_argo_diagnosis.py            # ~30 min  (6 basin runs)
python dcps/scripts/wavelet_phase_validation.py      # ~30 min  (Morlet cross-check)
```

Total cold-start cost (clone → headline pipelines done) is roughly:
- ~15 min: install + ~1 GB HadISST/PALMOD download
- ~10 min: ORAS5/GLORYS12 prerequisites (if you have them; otherwise ~hours via Copernicus Marine)
- ~1.5 h: all headline + revision pipelines on a recent laptop

### Figure output

The figure-generating scripts (`make_*`, `plot_*` under `dcps/scripts/`)
write PDFs into a sibling `manuscript/figs/` directory. The path
`manuscript/` at the repo root is gitignored — create it locally:

```bash
mkdir -p manuscript/figs
```

…and then any figure script will deposit its PDF there.

---

## Tests

```bash
pytest dcps/tests/ -v                    # 15 tests, ~10 s
pytest quiescence_toolkit/tests/ -v      #  8 tests, ~20 s
```

CI runs the same suite on Python 3.11, 3.12, and 3.13 on every push
and pull request (see badge at the top).

---

## Data dependencies

| Dataset | Use | Citation / URL |
|---|---|---|
| PALMOD-130k v2 | Holocene paleo SST | Jonkers et al. 2026, *Earth Syst. Sci. Data*; PANGAEA DOI 10.1594/PANGAEA.984602 |
| HadISST v1.1 | 1870–2023 SST | Rayner et al. 2003; metoffice.gov.uk/hadobs/hadisst |
| ORAS5 | 1958–2023 ocean reanalysis | Zuo et al. 2019; ECMWF Copernicus |
| GLORYS12V1 | 1993–2025 ocean reanalysis | Lellouche et al. 2021; Copernicus Marine |
| ECCO-V4r4 | 1992–2017 ocean reanalysis | Forget et al. 2015; JPL ECCO |
| CMIP6 (historical + ssp245 + ssp585 + piControl + past1000 + 1pctCO2 + abrupt-4xCO2) | model intercomparison | Eyring et al. 2016; Pangeo zarr catalogue |
| Caesar 2021 multi-proxy | gap-fill paleoclimate proxies | Caesar et al. 2021, *Nat. Geosci.*; github.com/ncahill89/AMOC-Analysis |
| Moffa-Sánchez | RAPID-region paleoceanographic records | Moffa-Sánchez et al. 2019; PANGAEA |
| RAPID-MOCHA 26.5°N | modern AMOC observations | McCarthy et al. 2015; rapid.ac.uk |
| ERA5 monthly | atmospheric forcing diagnostics | Hersbach et al. 2020; ECMWF Copernicus |

---

## License

Source code released under the **MIT License** (see `LICENSE`).

## Citation

If you use the analysis code in your work, please cite the repository
commit:

```bibtex
@misc{RostamiFallah_dcps_cold_blob,
  author = {Rostami, Masoud and Fallah, Bijan},
  title  = {{dcps-cold-blob}: phase-synchronisation analysis package
            for the {N}orth {A}tlantic {C}old {B}lob across 12{,}000 years},
  year   = {2026},
  url    = {https://github.com/bijanf/dcps-cold-blob},
}
```
