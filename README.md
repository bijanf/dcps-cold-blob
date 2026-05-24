# dcps-cold-blob

[![CI](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bijanf/dcps-cold-blob/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Reproducible analysis code for a pre-registered phase-synchronisation
study of the North Atlantic Cold Blob across the Holocene (PALMOD-130k
proxies), the instrumental era (HadISST 1870–2023), and CMIP6 future
projections through 2100.

This repository contains **only the analysis code, tests, and
reference datasets** required to reproduce the pipelines. It does not
ship the manuscript, figure PDFs, or research notes.

## What's here

| Path | Contents |
|---|---|
| `dcps/dcps/` | Python package — phase fields, Kuramoto order parameter, winding-number topology, transfer-entropy estimators, Nature-style figure helper |
| `dcps/scripts/` | End-to-end analysis pipelines (multi-basin Quiescence, multi-proxy unprecedentedness test, geostrophic-EKE mechanism test, CMIP6 emergent constraint, piControl detection-and-attribution, large-ensemble Q, past1000 forced-paleo, SSP scenario sweeps) |
| `dcps/tests/` | Unit tests for the package |
| `quiescence_toolkit/` | Standalone Q-index toolkit: source modules, data fetchers (ERA5, HighResMIP), reproduction notebooks, JSON result tables |
| `data/external/` | Small reference datasets (Caesar 2021 multi-proxy `.xlsx`, Thornalley 2018 `.csv`, Steinhilber 2012 TSI, Moffa-Sánchez `.tab`, ERA5 monthly aggregates). Bulk data is gitignored — see *Reproducing* below. |
| `data/external/p6/` | P6 cross-prediction verdict JSONs and summary tables |
| `resume_jobs.sh`, `run_dlesym_p6.sbatch` | SLURM helpers for the PIK A40 batch runs |

## Installation

```bash
cd dcps && pip install -e . && cd ..
```

## Fetching the bulk external data

```bash
mkdir -p data/external/hadisst
curl -L https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz \
     -o data/external/hadisst/HadISST_sst.nc.gz
gunzip data/external/hadisst/HadISST_sst.nc.gz

mkdir -p data/external/palmod_130k
curl -L https://lipdverse.org/PalMod/2_0_0/PalMod2_0_0.zip \
     -o data/external/palmod_130k/PalMod2_0_0.zip
unzip -d data/external/palmod_130k/lipd_unzipped \
     data/external/palmod_130k/PalMod2_0_0.zip
```

ORAS5 / GLORYS12 / ECCO / RAPID reanalyses are accessed via the local
ARDP cache; edit `dcps/dcps/config.py` to point at your own data
directory.

## Running the analyses

```bash
# Headline pipelines
python dcps/scripts/multi_basin_quiescence.py        # ~minutes
python dcps/scripts/winding_basin_test.py            # ~minutes
python dcps/scripts/cross_era_contrast.py            # ~seconds
python dcps/scripts/cold_blob_unprecedented.py       # ~seconds
python dcps/scripts/multi_proxy_unprecedented.py     # ~seconds
python dcps/scripts/eke_quiescence_test.py           # ~minutes
python dcps/scripts/cmip6_emergent_constraint.py     # ~seconds

# Revision-stage pipelines
python dcps/scripts/eke_quiescence_eddy_resolving.py # ~10 min (GLORYS12 1/12° NA)
python dcps/scripts/nao_phase_regression.py          # ~5 min  (downloads NAO/AMO)
python dcps/scripts/cmip6_picontrol_dna.py           # ~30 min (Pangeo piControl)
python dcps/scripts/cold_blob_unprecedented_bootstrap.py  # ~seconds (B=1000)
python dcps/scripts/bandpass_sensitivity.py          # ~30 min (3x3 sweep)
python dcps/scripts/pre_argo_diagnosis.py            # ~30 min (6 basin runs)
python dcps/scripts/wavelet_phase_validation.py      # ~30 min (Morlet cross-check)
```

Figure-generating scripts under `dcps/scripts/` (`make_*`, `plot_*`)
write their PDFs into a sibling `manuscript/figs/` directory. If
that directory does not exist in your checkout, create it or symlink
it to wherever you want figures collected — the scripts will
`mkdir -p` the parent on first run.

## Tests

```bash
pytest dcps/tests/
pytest quiescence_toolkit/tests/
```

## Data dependencies

| Dataset | Use | Citation / URL |
|---|---|---|
| PALMOD-130k v2 | Holocene paleo SST | Jonkers et al. 2026, *Earth Syst. Sci. Data*; PANGAEA DOI 10.1594/PANGAEA.984602 |
| HadISST v1.1 | 1870–2023 SST | Rayner et al. 2003; metoffice.gov.uk/hadobs/hadisst |
| ORAS5 | 1958–2023 ocean reanalysis | Zuo et al. 2019; ECMWF Copernicus |
| GLORYS12V1 | 1993–2025 ocean reanalysis | Lellouche et al. 2021; Copernicus Marine |
| ECCO-V4r4 | 1992–2017 ocean reanalysis | Forget et al. 2015; JPL ECCO |
| CMIP6 historical + ssp245 + ssp585 + piControl + past1000 + 1pctCO2 + abrupt-4xCO2 | model intercomparison | Eyring et al. 2016; Pangeo zarr catalogue |
| Caesar 2021 multi-proxy | gap-fill paleoclimate proxies | Caesar et al. 2021, *Nat. Geosci.*; github.com/ncahill89/AMOC-Analysis |
| Moffa-Sánchez | RAPID-region paleoceanographic records | Moffa-Sánchez et al. 2019; PANGAEA |
| RAPID-MOCHA 26.5°N | modern AMOC observations | McCarthy et al. 2015; rapid.ac.uk |
| ERA5 monthly | atmospheric forcing diagnostics | Hersbach et al. 2020; ECMWF Copernicus |

## License

Source code released under the **MIT License** (see `LICENSE`).

## Citation

If you use the analysis code in your work, please cite the repository
commit:

```
@misc{FallahRostami_dcps_cold_blob,
  author = {Fallah, Bijan and Rostami, Masoud},
  title  = {{dcps-cold-blob}: phase-synchronisation analysis package
            for the {N}orth {A}tlantic {C}old {B}lob across 12{,}000 years},
  year   = {2026},
  url    = {https://github.com/bijanf/dcps-cold-blob},
}
```
