# dcps-cold-blob

**The Anthropocene North Atlantic Cold Blob is unprecedented in 12,000 years.**

Pre-registered phase-synchronisation analysis of the North Atlantic
Cold Blob across the Holocene (PALMOD-130k proxies), the instrumental
era (HadISST 1870–2023), and CMIP6 future projections through 2100.

**Authors.** Bijan Fallah, Masoud Rostami.

**Status.** Manuscript drafted for *Nature Communications* submission;
private working copy.

---

## What's here

| Path | Contents |
|---|---|
| `manuscript/` | LaTeX source, 4 main figures, bibliography, Makefile |
| `dcps/dcps/` | Python package: phase fields, Kuramoto order parameter, winding-number topology, transfer entropy, Nature-style figure helper |
| `dcps/scripts/` | End-to-end analysis pipelines (multi-basin Quiescence, multi-proxy unprecedentedness test, geostrophic-EKE mechanism test, CMIP6 emergent constraint, etc.) |
| `dcps/tests/` | Unit tests for the package |
| `data/external/` | Small reference datasets (Caesar 2021 multi-proxy xlsx, Thornalley 2018 CSV, Steinhilber 2012 TSI). Bulk data (HadISST, PALMOD LiPD) is gitignored; see *Reproducing* below. |
| `CLAUDE.md`, `roadmap.md` | Working notes |

## Headline findings

1. **Multi-basin Quiescence pattern** — time-mean local Kuramoto
   coherence anti-correlates with the climatological sea-surface-height
   gradient across three independent ocean basins (NA, NP, ACC;
   Pearson ρ = −0.32, −0.35, −0.46, all p < 10⁻²⁶).
2. **Direct mechanistic evidence** — geostrophic eddy kinetic energy
   from ORAS5 SSH anti-correlates with local coherence in two of
   three basins (ρ = −0.34, −0.45 in NP and ACC).
3. **Unprecedented at every resolved timescale** — the modern
   subpolar–subtropical SST contrast trend rate is outside the
   Holocene Sen-slope envelope at |z| = 3.6 (500-yr), 5.2 (1000-yr),
   11.9 (2000-yr), 37.6 (5000-yr) sliding windows in PALMOD-130k.
   Three of four high-resolution Caesar 2021 multi-proxy records
   confirm at decadal-to-centennial scales (|z| up to 22.1).
4. **CMIP6 emergent-constraint subset** — the three models matching
   the observed historical Cold-Blob trend project less aggressive
   widening than the full ensemble; the observed rate exceeds any
   CMIP6 subset projection through 2100. The unprecedented claim is
   anchored in observations and proxies, not in CMIP6.

## Reproducing

```bash
# 1. Install the dcps package
cd dcps && pip install -e . && cd ..

# 2. Fetch external bulk data (see Data Availability section of the
#    manuscript for citation; URLs below current at time of writing).
mkdir -p data/external/hadisst
curl -L https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz \
     -o data/external/hadisst/HadISST_sst.nc.gz
gunzip data/external/hadisst/HadISST_sst.nc.gz

mkdir -p data/external/palmod_130k
curl -L https://lipdverse.org/PalMod/2_0_0/PalMod2_0_0.zip \
     -o data/external/palmod_130k/PalMod2_0_0.zip
unzip -d data/external/palmod_130k/lipd_unzipped \
     data/external/palmod_130k/PalMod2_0_0.zip

# 3. Configure the ORAS5 / GLORYS12 / ECCO / RAPID data backbone path
#    (ARDP cache at ~/Documents/AMOC_renalysis/data/ by default;
#    edit dcps/dcps/config.py to point elsewhere).

# 4. Run the analysis pipelines
python dcps/scripts/multi_basin_quiescence.py        # ~minutes
python dcps/scripts/winding_basin_test.py            # ~minutes
python dcps/scripts/cross_era_contrast.py            # ~seconds
python dcps/scripts/cold_blob_unprecedented.py       # ~seconds
python dcps/scripts/multi_proxy_unprecedented.py     # ~seconds
python dcps/scripts/eke_quiescence_test.py           # ~minutes
python dcps/scripts/cmip6_emergent_constraint.py     # ~seconds
python dcps/scripts/make_envelope_figure.py          # ~seconds
python dcps/scripts/make_wow_figure.py               # ~seconds

# 5. Build the manuscript
cd manuscript && make all
```

## Data dependencies

| Dataset | Use | Citation / URL |
|---|---|---|
| PALMOD-130k v2 | Holocene paleo SST | Jonkers et al. 2026, Earth Syst. Sci. Data; PANGAEA DOI 10.1594/PANGAEA.984602 |
| HadISST v1.1 | 1870–2023 SST | Rayner et al. 2003; metoffice.gov.uk/hadobs/hadisst |
| ORAS5 | 1958–2023 ocean reanalysis | Zuo et al. 2019; ECMWF Copernicus |
| GLORYS12V1 | 1993–2025 ocean reanalysis | Lellouche et al. 2021; Copernicus Marine |
| ECCO-V4r4 | 1992–2017 ocean reanalysis | Forget et al. 2015; JPL ECCO |
| CMIP6 historical + ssp245 + ssp585 | future projections | Eyring et al. 2016; Pangeo zarr catalogue |
| Caesar 2021 multi-proxy | gap-fill paleoclimate proxies | Caesar et al. 2021, Nat. Geosci.; github.com/ncahill89/AMOC-Analysis |
| RAPID-MOCHA 26.5°N | modern AMOC observations | McCarthy et al. 2015; rapid.ac.uk |

## License

Source code released under the **MIT License** (see `LICENSE`).
Manuscript text and figures: all rights reserved until publication.

## Citation

If you use the analysis code or pipelines in your work, please cite
the manuscript when it is published. Until then, please cite the
repository commit:

```
@misc{FallahRostami_DCPSColdBlob,
  author = {Fallah, Bijan and Rostami, Masoud},
  title  = {{DCPS}-Cold-Blob: pre-registered phase-synchronisation analysis
            of the {N}orth {A}tlantic {C}old {B}lob across 12,000 years},
  year   = {2026},
  note   = {Private working repository},
}
```
