#!/bin/bash
# SLURM submission template for the P6 cross-prediction test on PIK Cluster.
# See PRE_REGISTRATION_P6.md for the sealed protocol and OPERATIONS.md for
# the full PIK runbook.
#
# Usage on PIK:
#   1. Replace the six <VERIFY ON CLUSTER> tokens below with your values.
#   2. First run a smoke test:   STAGE=smoke sbatch dcps/scripts/run_dlesym_p6.sbatch
#   3. Then the full 30-yr run:  STAGE=full  sbatch dcps/scripts/run_dlesym_p6.sbatch
#
#SBATCH --job-name=dlesym-p6
#SBATCH --account=<PROJECT>                # <VERIFY ON CLUSTER>: from `sshare -U $USER`
#SBATCH --partition=<PARTITION>            # <VERIFY ON CLUSTER>: from `sinfo -O partition,gres | grep -i gpu`
#SBATCH --gres=gpu:<GPU_TYPE>:1            # <VERIFY ON CLUSTER>: e.g. gpu:a100:1 or gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/dlesym-p6-%j.out

set -euo pipefail

# ---------------------------------------------------------------------------
# Stage selector: smoke (1 yr, 1 h wall) or full (30 yr, 12 h wall).
# Set via:  STAGE=smoke sbatch …    or    STAGE=full sbatch …
# ---------------------------------------------------------------------------
STAGE="${STAGE:-smoke}"
case "${STAGE}" in
  smoke) YEARS=1;  WALL="01:00:00"; OUT_NC="smoke_1yr_sst.nc" ;;
  full)  YEARS=30; WALL="12:00:00"; OUT_NC="p6_30yr_sst.nc" ;;
  *)     echo "STAGE must be 'smoke' or 'full', got '${STAGE}'"; exit 2 ;;
esac
echo "[P6] stage=${STAGE}  years=${YEARS}  wall=${WALL}  out=${OUT_NC}"

# ---------------------------------------------------------------------------
# Environment.  Replace placeholders with your PIK paths.
# ---------------------------------------------------------------------------
PROJECT="<PROJECT>"                                              # <VERIFY>
ENV_PATH="/p/projects/${PROJECT}/${USER}/envs/dlesym"            # <VERIFY>
CHECKPOINT_PATH="/p/projects/${PROJECT}/${USER}/checkpoints/dlesym-v1-era5"  # <VERIFY>
OUTPUT_DIR="/p/projects/${PROJECT}/${USER}/dlesym_p6_out"        # <VERIFY>
ERA5_MIRROR="/p/projects/climate_data_center/ERA5"               # <VERIFY: adjust to actual path, or unset>
EKE_CLIM_PATH="/p/projects/${PROJECT}/${USER}/cache/glorys12_eke_clim_atlantic_2deg.nc"  # <VERIFY>

mkdir -p "${OUTPUT_DIR}" logs

# Module load -- names are best-effort; run `module avail` once to verify.
module purge
module load anaconda/2024.10 2>/dev/null || module load anaconda 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true

source activate "${ENV_PATH}"

export HF_HOME="/p/projects/${PROJECT}/${USER}/hf-cache"
export TMPDIR="/p/tmp/${USER}"
mkdir -p "${HF_HOME}" "${TMPDIR}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

python dcps/scripts/quiescence_dlesym.py \
    --mode "${STAGE}" \
    --years "${YEARS}" \
    --ic 2010-01-01 \
    --checkpoint "${CHECKPOINT_PATH}" \
    --out "${OUTPUT_DIR}/${OUT_NC}" \
    --era5-mirror "${ERA5_MIRROR}" \
    --basin atlantic \
    --eke-clim "${EKE_CLIM_PATH}" \
    --verdict-json "${OUTPUT_DIR}/p6_verdict_${STAGE}.json"

echo "[P6] stage=${STAGE} done; verdict at ${OUTPUT_DIR}/p6_verdict_${STAGE}.json"
