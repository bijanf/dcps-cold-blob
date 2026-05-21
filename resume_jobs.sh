#!/usr/bin/env bash
# Resume the autonomous Holocene-Q analysis pipeline.
# All scripts are resume-safe: cached results are skipped on re-run.
#
# Usage:
#   bash resume_jobs.sh
#
# Prints what's currently cached and re-launches the workers in the
# background.  Logs go to logs/.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== current cache state ==="
echo "  bulk JSONs:           $(ls dcps/cache/holocene_exit/bulk/*.json 2>/dev/null | wc -l) of 32 models"
echo "  hunter Q results:     $(ls dcps/cache/holocene_exit/auto/*.json 2>/dev/null | wc -l)"
echo "  hunter cooldown:      $(ls dcps/cache/holocene_exit/auto/*.failed 2>/dev/null | wc -l)"
echo "  grand-ens JSONs:      $(ls dcps/cache/holocene_exit/grand_*.json 2>/dev/null | wc -l)"
echo "  grand-ens member ckpt: $(find dcps/cache/holocene_exit/members -name '*.json' 2>/dev/null | wc -l)"
echo "  epoch EKE maps NA:    $(ls dcps/cache/eke_maps/*.nc 2>/dev/null | wc -l) of 29"
echo "  EKE time series NA:   $(ls dcps/cache/eke_timeseries/*.json 2>/dev/null | wc -l) of 29"
echo
mkdir -p logs

echo "=== launching jobs ==="
# 1. Hunter (NA) — resumes from cached per-target JSONs
nohup python -u dcps/scripts/autonomous_holocene_hunt.py \
  --basin atlantic --max-passes 100 --include-dkrz --include-ceda --include-ipsl \
  > logs/autonomous_hunt.log 2>&1 &
disown $!
echo "  hunter NA           PID=$!"

# 2. Figure refresh
nohup python -u dcps/scripts/autonomous_figure_refresh.py --interval-s 600 \
  > logs/autonomous_refresh.log 2>&1 &
disown $!
echo "  figure refresh      PID=$!"

# 3. Grand ensembles (resume-safe per-member checkpointing now in place)
nohup python -u dcps/scripts/holocene_q_grand_ensemble.py \
  --model CanESM5 --experiment "historical+ssp585" --basin atlantic \
  --n-members 50 --year-start 1850 --year-end 2099 --stride 10 \
  > logs/grand_canesm5_50.log 2>&1 &
disown $!
echo "  CanESM5 50-member   PID=$!"

nohup python -u dcps/scripts/holocene_q_grand_ensemble.py \
  --model MPI-ESM1-2-LR --experiment "historical+ssp585" --basin atlantic \
  --n-members 10 --year-start 1850 --year-end 2099 --stride 10 \
  > logs/grand_mpi_lr.log 2>&1 &
disown $!
echo "  MPI-ESM1-2-LR 10    PID=$!"

nohup python -u dcps/scripts/holocene_q_grand_ensemble.py \
  --model UKESM1-0-LL --experiment "historical+ssp585" --basin atlantic \
  --n-members 5 --year-start 1850 --year-end 2099 --stride 10 \
  > logs/grand_ukesm.log 2>&1 &
disown $!
echo "  UKESM1-0-LL 5       PID=$!"

# 4. Epoch EKE maps (resumes from cached NetCDFs)
nohup python -u dcps/scripts/compute_epoch_eke_maps.py --basin atlantic \
  > logs/epoch_eke_atlantic.log 2>&1 &
disown $!
echo "  epoch EKE maps NA   PID=$!"

# 5. EKE time series (resumes from cached JSONs)
nohup python -u dcps/scripts/compute_eke_timeseries.py --basin atlantic \
  > logs/eke_timeseries.log 2>&1 &
disown $!
echo "  EKE time series NA  PID=$!"

# 6. CEDA past1000 queue (5 PMIP3 models + 8 GISS-E2-R forcing variants).
# Requires ~/.ceda_token (CEDA Access Token).  Resume-safe: per-model
# JSONs are checked first.
if [ -f "$HOME/.ceda_token" ]; then
    nohup python -u dcps/scripts/ceda_past1000_queue.py \
      --basin atlantic --stride-yr 10 \
      > logs/ceda_past1000_queue.log 2>&1 &
    disown $!
    echo "  CEDA past1000 queue PID=$!"
else
    echo "  CEDA past1000 queue SKIPPED (no ~/.ceda_token)"
fi

# 7. Narrow-epoch EKE diff maps (1850-1900 / 2030-2060 / 2070-2099).
nohup python -u dcps/scripts/compute_eke_epoch_diff.py --basin atlantic \
  > logs/eke_epoch_diff.log 2>&1 &
disown $!
echo "  EKE epoch diff NA   PID=$!"

sleep 2
echo
echo "=== alive jobs ==="
ps -eo pid,etime,cmd | grep -E "python.*(holocene|autonomous|epoch_eke|eke_time)" | grep -v grep | awk '{print "  "$1, $5}'
echo
echo "To monitor:  tail -f logs/<jobname>.log"
echo "To re-render figures from cache anytime:  python dcps/scripts/plot_holocene_status.py"
