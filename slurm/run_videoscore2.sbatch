#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
set -Eeuo pipefail
: "${BATCH_DIR:?Set BATCH_DIR}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${VIDEOSCORE2_PYTHON:?Set VIDEOSCORE2_PYTHON}"
: "${VIDEOSCORE2_MODEL_DIR:?Set VIDEOSCORE2_MODEL_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
test -f "$SOURCE_ROOT/VideoScore2/vs2_inference.py"
test -f "$VIDEOSCORE2_MODEL_DIR/config.json"
for shard in 00001 00002 00003 00004; do
    test -s "$VIDEOSCORE2_MODEL_DIR/model-${shard}-of-00004.safetensors"
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/data/zzhu126/cache/huggingface}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$VIDEOSCORE2_PYTHON" -B "$SCRIPT_DIR/run_videoscore2.py" \
  --selection "$BATCH_DIR/selection.json" --index "$SLURM_ARRAY_TASK_ID" \
  --official-script "$SOURCE_ROOT/VideoScore2/vs2_inference.py" \
  --model-dir "$VIDEOSCORE2_MODEL_DIR" --output-root "$BATCH_DIR/videoscore2" --seed 42
