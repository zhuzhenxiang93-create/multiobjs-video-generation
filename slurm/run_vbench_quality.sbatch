#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
set -Eeuo pipefail
: "${BATCH_DIR:?Set BATCH_DIR}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${VBENCH_PYTHON:?Set VBENCH_PYTHON}"
DIMS=(subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality)
DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
REPO="$SOURCE_ROOT/VBench"
mkdir -p "$BATCH_DIR/vbench/$DIM"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/data/zzhu126/cache/vbench}"
export HF_HOME="${HF_HOME:-/data/zzhu126/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data/zzhu126/cache/torch}"
export TMPDIR="${TMPDIR:-/data/zzhu126/tmp}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
exec "$VBENCH_PYTHON" -B evaluate.py --videos_path "$BATCH_DIR/videos" \
  --dimension "$DIM" --mode custom_input \
  --prompt_file "$BATCH_DIR/vbench_prompt_map.json" \
  --output_path "$BATCH_DIR/vbench/$DIM" --load_ckpt_from_local True
