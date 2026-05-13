#!/bin/bash
set -euo pipefail
source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh
conda activate qdit
nvidia-smi --query-gpu=name --format=csv,noheader

REF=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/imagenet256_ref/VIRTUAL_imagenet256_labeled.npz
EV=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/Q-DiT/models/evaluations/evaluator.py

for SL in 48 128; do
  SMP=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg${SL}/samples.npz
  OUT=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg${SL}/openai_eval.txt
  echo "=========================================="
  echo "uniform_avg${SL}  ref=$REF  smp=$SMP"
  echo "=========================================="
  python -u "$EV" "$REF" "$SMP" 2>&1 | tee "$OUT"
done
