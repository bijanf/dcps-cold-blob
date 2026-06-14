#!/bin/bash
# SLURM submission template for the P6 cross-prediction test.
# See PRE_REGISTRATION_P6.md for the sealed protocol and OPERATIONS.md for
# the full runbook.
#
# Usage:
#   1. Replace the <VERIFY ON CLUSTER> tokens below with your values.
#   2. Smoke test:    STAGE=smoke sbatch dcps/scripts/run_dlesym_p6.sbatch
#   3. Full 30-yr:    STAGE=full  sbatch dcps/scripts/run_dlesym_p6.sbatch
#   4. Reprocess only (after rollout exists):
#                     STAGE=postprocess sbatch dcps/scripts/run_dlesym_p6.sbatch
#
# Ensemble parameterisation (set on the sbatch command line):
#   IC_DATE=1990-01-01 OUT_TAG=ic_1990 sbatch ...        # IC ensemble
#   ATMOS_M=1 OCEAN_M=0 OUT_TAG=a1o0 sbatch ...           # member ensemble
#   BASIN=pacific OUT_TAG=basin_pacific sbatch ...        # multi-basin
#
# Verified configuration (PIK 2026-05-15, group `poem`, A40 GPUs):
#   account=poem  partition=gpu  gres=gpu:1  qos=gpumedium
#   anaconda/2025  cuda/12.6.0
#   30-yr rollout = ~16 min on a single A40 (drift 0.0007 K/yr)
#
#SBATCH --job-name=dlesym-p6
#SBATCH --account=<PROJECT>                # <VERIFY ON CLUSTER>: from `sshare -U $USER`
#SBATCH --partition=<PARTITION>            # <VERIFY ON CLUSTER>: from `sinfo -O partition,gres | grep -i gpu`
#SBATCH --gres=gpu:<GPU_TYPE>:1            # <VERIFY ON CLUSTER>: e.g. gpu:1, gpu:a100:1, gpu:l40s:1
#SBATCH --qos=<QOS>                        # <VERIFY ON CLUSTER>: optional, e.g. gpumedium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/dlesym-p6-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Stage selector
#   smoke        — 1 simulated year, ~1 h wall (pipeline shakedown)
#   full         — 30 simulated years, ~12 h wall (the registered run)
#   postprocess  — skip rollout, re-run analysis on an existing SST NetCDF
# NOTE: pass --time on the sbatch command line if your scheduler rejects
# in-script #SBATCH override; bash variable expansion inside #SBATCH lines
# does not work on most SLURM installs.
# ---------------------------------------------------------------------------
STAGE="${STAGE:-smoke}"
case "${STAGE}" in
  smoke)       MODE=smoke;            YEARS=1;  WALL="01:00:00"; SST_BASE=smoke_1yr_sst ;;
  full)        MODE=full;             YEARS=30; WALL="12:00:00"; SST_BASE=p6_30yr_sst ;;
  postprocess) MODE=postprocess-only; YEARS=0;  WALL="00:30:00"; SST_BASE=p6_30yr_sst ;;
  *)           echo "STAGE must be 'smoke' | 'full' | 'postprocess', got '${STAGE}'"; exit 2 ;;
esac

# Ensemble parameterisation. All optional; defaults reproduce the primary run.
IC_DATE="${IC_DATE:-2010-01-01}"
OUT_TAG="${OUT_TAG:-}"           # e.g. ic_1990, a0o0, basin_pacific
ATMOS_M="${ATMOS_M:-0}"
OCEAN_M="${OCEAN_M:-0}"
BASIN="${BASIN:-atlantic}"

if [ -n "${OUT_TAG}" ]; then
  OUT_NC="${SST_BASE}_${OUT_TAG}.nc"
  VERDICT_NAME="p6_verdict_${OUT_TAG}.json"
else
  OUT_NC="${SST_BASE}.nc"
  VERDICT_NAME="p6_verdict_${STAGE}.json"
fi
echo "[P6] stage=${STAGE} mode=${MODE} years=${YEARS} wall=${WALL} out=${OUT_NC}"
echo "[P6] ic=${IC_DATE} basin=${BASIN} atmos_m=${ATMOS_M} ocean_m=${OCEAN_M} tag=${OUT_TAG:-<default>}"

# ---------------------------------------------------------------------------
# Environment. Replace placeholders with your paths.
# ---------------------------------------------------------------------------
PROJECT="<PROJECT>"                                                       # <VERIFY>
ENV_PATH="/p/projects/${PROJECT}/${USER}/envs/dlesym"                     # <VERIFY>
CHECKPOINT_PATH="/p/projects/${PROJECT}/${USER}/checkpoints/dlesym-v1-era5"  # <VERIFY>
OUTPUT_DIR="/p/projects/${PROJECT}/${USER}/dlesym_p6_out"                 # <VERIFY>
# Pre-fetched ERA5 IC subset (built per-date on login via make_era5_ic.py).
# Some clusters block outbound internet from GPU nodes → ARCO unreachable;
# in that case ship a local file. Otherwise leave the path absent and the
# script falls back to earth2studio.data.ARCO.
ERA5_MIRROR="/p/projects/${PROJECT}/${USER}/era5_ic_${IC_DATE}.nc"
# Basin-specific EKE climatology (atlantic / pacific / southern).
if [ "${BASIN}" = "atlantic" ]; then
  EKE_CLIM_PATH="/p/projects/${PROJECT}/${USER}/cache/glorys12_eke_clim_atlantic_2deg.nc"
else
  EKE_CLIM_PATH="/p/projects/${PROJECT}/${USER}/cache/glorys12_eke_clim_${BASIN}_2deg.nc"
fi

mkdir -p "${OUTPUT_DIR}" logs

# Module load -- names are best-effort; run `module avail` once to verify.
# PIK 2026-05 worked with anaconda/2025 + cuda/12.6.0; older installs may
# need anaconda/2024.10 + cuda/12.4.
module purge
module load anaconda/2025 2>/dev/null || module load anaconda 2>/dev/null || true
module load cuda/12.6.0  2>/dev/null || module load cuda     2>/dev/null || true

source activate "${ENV_PATH}"

# Disable user-site (~/.local) which can hold a stale numpy/cftime that
# shadows the env on shared filesystems.
export PYTHONNOUSERSITE=1
export HF_HOME="/p/projects/${PROJECT}/${USER}/hf-cache"
export TMPDIR="/p/tmp/${USER}"
mkdir -p "${HF_HOME}" "${TMPDIR}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# EKE climatology is only required for the cross-prediction step (Phase B).
# For STAGE=smoke or STAGE=full we still run Phase A (rollout + drift check)
# even when EKE is not yet staged; Phase B can be re-run later with
# STAGE=postprocess.
EKE_FLAG=()
if [ -f "${EKE_CLIM_PATH}" ]; then
  EKE_FLAG=(--eke-clim "${EKE_CLIM_PATH}")
  echo "[P6] EKE climatology present: ${EKE_CLIM_PATH}"
else
  echo "[P6] EKE climatology not yet at ${EKE_CLIM_PATH}; Phase B skipped."
  echo "[P6] Re-run with STAGE=postprocess once EKE is staged."
fi

ERA5_FLAG=()
if [ -n "${ERA5_MIRROR}" ] && [ -e "${ERA5_MIRROR}" ]; then
  ERA5_FLAG=(--era5-mirror "${ERA5_MIRROR}")
fi

python dcps/scripts/quiescence_dlesym.py \
    --mode "${MODE}" \
    --years "${YEARS}" \
    --ic "${IC_DATE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --out "${OUTPUT_DIR}/${OUT_NC}" \
    "${ERA5_FLAG[@]}" \
    --basin "${BASIN}" \
    --atmos-member "${ATMOS_M}" \
    --ocean-member "${OCEAN_M}" \
    "${EKE_FLAG[@]}" \
    --verdict-json "${OUTPUT_DIR}/${VERDICT_NAME}"

echo "[P6] stage=${STAGE} done; verdict at ${OUTPUT_DIR}/${VERDICT_NAME}"
