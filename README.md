# dcps-cold-blob

[![CI](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/formatter/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://docs.pytest.org/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](#requirements)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/bijanf/dcps-cold-blob/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/bijanf/dcps-cold-blob/main.svg)](https://github.com/bijanf/dcps-cold-blob/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/bijanf/dcps-cold-blob?include_prereleases&sort=semver&display_name=tag)](https://github.com/bijanf/dcps-cold-blob/releases)
[![GitHub issues](https://img.shields.io/github/issues/bijanf/dcps-cold-blob.svg)](https://github.com/bijanf/dcps-cold-blob/issues)
[![GitHub stars](https://img.shields.io/github/stars/bijanf/dcps-cold-blob.svg?style=social)](https://github.com/bijanf/dcps-cold-blob/stargazers)

> **Reproducible analysis pipelines for a phase-synchronisation study of
> the North Atlantic Cold Blob across the Holocene (PALMOD-130k proxies),
> the instrumental era (HadISST 1870–2023), and CMIP6 future projections
> through 2100.**

This repository is **code-only**: the Python package, the end-to-end
analysis scripts, the unit tests, and the small reference datasets
needed to reproduce every figure. Manuscript LaTeX, figure PDFs, and
research notes are kept out of the public history on purpose; this is
a working tool, not a paper.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Fetching external data](#fetching-external-data)
- [Running the analyses](#running-the-analyses)
- [Tests and continuous integration](#tests-and-continuous-integration)
- [Code quality](#code-quality)
- [Reproducibility](#reproducibility)
- [Data dependencies](#data-dependencies)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Highlights

- **End-to-end pipelines.** Single-command reproduction of the
  unprecedentedness test, Quiescence-Signature fit, CMIP6 emergent
  constraint, piControl detection-and-attribution, multi-basin
  large-ensemble Q, past1000 forced-paleo envelope, and the SSP
  scenario sweeps.
- **Pre-registered.** Basin definitions, sliding-window grid, exit
  thresholds, and bootstrap parameters are frozen before any
  detection run — see the `seal()` checkpoints in the
  `dcps/scripts/` headline scripts.
- **CI-tested on Python 3.11 / 3.12 / 3.13.** GitHub Actions runs
  the unit-test matrix and a Ruff lint gate on every push and pull
  request.
- **Pangeo-native CMIP6 access.** No ESGF nodes, no wget chains —
  zarr stores read straight out of `gcs://cmip6/`.
- **Small footprint.** Repo clones in under 5 MB; only small
  reference datasets are tracked, bulk binaries (HadISST, PALMOD,
  ERA5, ORAS5, GLORYS12, CMIP6) are fetched on demand.

---

## Architecture

```
                           ┌──────────────────────────────────────┐
                           │      dcps/  (Python package)         │
                           │                                      │
                           │   phase fields  (analytic signal,    │
                           │     Hilbert, bandpass)               │
                           │   Kuramoto order parameter R(t)      │
                           │   Quiescence Index   Q = ⟨R⟩_t       │
                           │   winding-number topology            │
                           │   KSG transfer entropy               │
                           │   Nature-style figure helper         │
                           └──────────────┬───────────────────────┘
                                          │   imported by ↓
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
        ▼                                 ▼                                 ▼
 dcps/scripts/                     dcps/tests/                     quiescence_toolkit/
 end-to-end pipelines              23 unit tests                   standalone Q-index +
   • unprecedentedness               (CI matrix)                   data fetchers + notebooks
   • emergent constraint
   • EKE → Q mechanism
   • forced-paleo envelope
   • SSP scenario sweep
        │
        ▼
 data/external/   ←  small (xlsx, csv, tab, txt, JSON)
                  +  large (.nc / .zarr / .h5) fetched via dcps/scripts/data_fetchers/
```

Every analysis script is **self-contained** — pass `--help` to see
its options, then run it. Outputs are written under
`dcps/cache/<analysis-name>/` (intermediate arrays, JSON summaries)
and `manuscript/figs/` (PDFs); both directories are gitignored.

---

## Requirements

- **Python ≥ 3.11** (CI matrix: 3.11, 3.12, 3.13)
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
  - **Pangeo** (anonymous) for CMIP6 zarr

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
from dcps import te_ksg
import numpy as np
rng = np.random.default_rng(0)
y = rng.standard_normal(500)
x = 0.5*np.roll(y, 1) + 0.3*rng.standard_normal(500)
te = te_ksg.ksg_te(y, x, k=1, ell=1, k_nn=4)
print(f'KSG TE(y -> x) = {te:.3f}  (expect > 0)')
"
```

If `pytest` reports **23 passed** and the snippet prints a positive
KSG TE value, your install is healthy.

---

## Fetching external data

### Always free, no credentials

```bash
# HadISST 1870-2023 (Met Office) - used by the unprecedentedness pipelines
mkdir -p data/external/hadisst
curl -L https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz \
     -o data/external/hadisst/HadISST_sst.nc.gz
gunzip data/external/hadisst/HadISST_sst.nc.gz

# PALMOD-130k v2 (LiPDverse) - used by the Holocene-context pipelines
mkdir -p data/external/palmod_130k
curl -L https://lipdverse.org/PalMod/2_0_0/PalMod2_0_0.zip \
     -o data/external/palmod_130k/PalMod2_0_0.zip
unzip -d data/external/palmod_130k/lipd_unzipped \
     data/external/palmod_130k/PalMod2_0_0.zip
```

### Account required

```bash
# ERA5 monthly aggregates (NAO / AMO regression diagnostics) - needs ~/.cdsapirc
python dcps/scripts/fetch_era5.py

# PMIP3 past1000 forced-paleo via CEDA - needs CEDA bearer token in env
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

# Mechanism / sensitivity
python dcps/scripts/eke_quiescence_eddy_resolving.py # ~10 min  (GLORYS12 1/12 deg NA)
python dcps/scripts/nao_phase_regression.py          # ~5 min   (auto-downloads NAO/AMO)
python dcps/scripts/cmip6_picontrol_dna.py           # ~30 min  (Pangeo piControl)
python dcps/scripts/cold_blob_unprecedented_bootstrap.py  # ~seconds (B=1000)
python dcps/scripts/bandpass_sensitivity.py          # ~30 min  (3x3 sweep)
python dcps/scripts/pre_argo_diagnosis.py            # ~30 min  (6 basin runs)
python dcps/scripts/wavelet_phase_validation.py      # ~30 min  (Morlet cross-check)
```

Cold-start cost (clone → headline pipelines done):

- ~15 min: install + ~1 GB HadISST/PALMOD download
- ~10 min: ORAS5/GLORYS12 prerequisites (if you have them locally; otherwise ~hours via Copernicus Marine)
- ~1.5 h: all headline + mechanism pipelines on a recent laptop

### Figure output

The figure scripts (`make_*`, `plot_*` under `dcps/scripts/`) write
PDFs into a sibling `manuscript/figs/` directory. The path
`manuscript/` at the repo root is gitignored - create it locally:

```bash
mkdir -p manuscript/figs
```

…and then any figure script will deposit its PDF there.

---

## Tests and continuous integration

```bash
pytest dcps/tests/ -v                    # 15 tests, ~10 s
pytest quiescence_toolkit/tests/ -v      #  8 tests, ~20 s
```

GitHub Actions runs the same suite on Python 3.11, 3.12, and 3.13
plus a Ruff lint gate on every push and pull request (see the **CI**
badge at the top). The full workflow lives at
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Code quality

- **Lint:** Ruff with project rules in `dcps/pyproject.toml`
  (`pip install -e "./dcps[dev]" && ruff check .`).
- **Format:** Ruff formatter — `ruff format .` for a one-shot pass.
- **Type hints:** Public APIs in `dcps/dcps/` are typed; scripts in
  `dcps/scripts/` favour readability and use hints where they help.
- **No bulk binaries in history.** `.gitignore` blocks
  `*.nc / *.zarr / *.h5 / *.grib` repository-wide so a scratch
  download can never end up in a commit.

---

## Reproducibility

- **Pre-registration.** Basin shapes, sliding-window grid, exit
  thresholds, and bootstrap parameters are sealed via the
  `seal(...)` checkpoint inside each headline script before any
  detection result is recorded. Re-running with the same input data
  reproduces every diagnostic bit-for-bit (the random streams use
  fixed seeds derived from `seal(...)`).
- **Deterministic caches.** Intermediate arrays land in
  `dcps/cache/<analysis-name>/` keyed by a content hash of the input
  configuration, so re-running a downstream figure script does not
  re-trigger an upstream compute.
- **Versioned dependencies.** `dcps/pyproject.toml` pins the minimum
  versions used in CI; reproducing exact behaviour at a given commit
  is `pip install -e "./dcps[dev]"` against that tagged tree.

---

## Data dependencies

| Dataset | Use | Citation / URL |
|---|---|---|
| PALMOD-130k v2 | Holocene paleo SST | Jonkers et al. 2026, *Earth Syst. Sci. Data*; PANGAEA DOI 10.1594/PANGAEA.984602 |
| HadISST v1.1 | 1870-2023 SST | Rayner et al. 2003; metoffice.gov.uk/hadobs/hadisst |
| ORAS5 | 1958-2023 ocean reanalysis | Zuo et al. 2019; ECMWF Copernicus |
| GLORYS12V1 | 1993-2025 ocean reanalysis | Lellouche et al. 2021; Copernicus Marine |
| ECCO-V4r4 | 1992-2017 ocean reanalysis | Forget et al. 2015; JPL ECCO |
| CMIP6 (historical + ssp245 + ssp585 + piControl + past1000 + 1pctCO2 + abrupt-4xCO2) | model intercomparison | Eyring et al. 2016; Pangeo zarr catalogue |
| Caesar 2021 multi-proxy | gap-fill paleoclimate proxies | Caesar et al. 2021, *Nat. Geosci.*; github.com/ncahill89/AMOC-Analysis |
| Moffa-Sanchez | RAPID-region paleoceanographic records | Moffa-Sanchez et al. 2019; PANGAEA |
| RAPID-MOCHA 26.5N | modern AMOC observations | McCarthy et al. 2015; rapid.ac.uk |
| ERA5 monthly | atmospheric forcing diagnostics | Hersbach et al. 2020; ECMWF Copernicus |

---

## Project layout

| Path | Contents |
|---|---|
| `dcps/dcps/` | Python package — phase fields, Kuramoto order parameter, winding-number topology, KSG transfer-entropy estimators, Nature-style figure helper |
| `dcps/scripts/` | End-to-end analysis pipelines + figure generators |
| `dcps/tests/` | pytest unit tests for the package |
| `quiescence_toolkit/` | Standalone Q-index toolkit: source modules, data fetchers (ERA5, HighResMIP), reproduction notebooks, JSON result tables |
| `data/external/` | Small reference datasets (Caesar 2021 multi-proxy `.xlsx`, Thornalley 2018 `.csv`, Steinhilber 2012 TSI, Moffa-Sanchez `.tab`). Bulk data is gitignored — see [Fetching external data](#fetching-external-data). |
| `data/external/p6/` | P6 cross-prediction verdict JSONs |
| `.github/workflows/ci.yml` | GitHub Actions: pytest matrix (Python 3.11/3.12/3.13) + Ruff lint |

---

## Contributing

Bug reports, reproductions on other platforms, and PRs that improve
the analysis pipelines or extend the data fetchers are welcome.

1. Open an issue describing the change.
2. Fork, branch from `main`, and make focused commits.
3. Run `ruff check . && ruff format --check . && pytest dcps/tests/ quiescence_toolkit/tests/ -q` locally.
4. Open a PR against `main`; CI must be green before review.

---

## License

Source code released under the **MIT License** (see [`LICENSE`](LICENSE)).

---

## Citation

If you use this analysis code in your work, please cite the repository
commit you used:

```bibtex
@misc{RostamiFallah_dcps_cold_blob,
  author = {Rostami, Masoud and Fallah, Bijan},
  title  = {{dcps-cold-blob}: phase-synchronisation analysis package
            for the {N}orth {A}tlantic {C}old {B}lob across 12{,}000 years},
  year   = {2026},
  url    = {https://github.com/bijanf/dcps-cold-blob},
}
```
