#!/bin/bash
set -euo pipefail
source /home/zhkangqi/miniconda3/etc/profile.d/conda.sh
conda activate tfeval

NV=$(python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== node $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

REF=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/imagenet256_ref/VIRTUAL_imagenet256_labeled.npz
EV=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/Q-DiT/models/evaluations/evaluator.py

for SL in 48 128; do
  SMP=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg${SL}/samples.npz
  OUT=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_bitrev/uniform_avg${SL}/openai_eval.txt
  echo "=========================================="
  echo "uniform_avg${SL}  $(date)"
  echo "=========================================="
  python -u "$EV" "$REF" "$SMP" 2>&1 | tee "$OUT"
done
echo "=== all done $(date) ==="
