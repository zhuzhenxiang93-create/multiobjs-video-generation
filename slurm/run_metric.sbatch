#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -Eeuo pipefail
: "${BATCH_DIR:?Set BATCH_DIR}"
: "${METRIC:?Set METRIC}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
case "$METRIC" in
  countvid)
    : "${COUNTVID_PYTHON:?Set COUNTVID_PYTHON}"
    exec "$COUNTVID_PYTHON" -B "$SCRIPT_DIR/run_countvid.py" --batch "$BATCH_DIR" --index "$SLURM_ARRAY_TASK_ID" --countvid-repo "$SOURCE_ROOT/CountVid"
    ;;
  numina)
    : "${NUMINA_PYTHON:?Set NUMINA_PYTHON}"
    : "${GDINO_CONFIG:?Set GDINO_CONFIG}"
    : "${GDINO_WEIGHTS:?Set GDINO_WEIGHTS}"
    export PYTHONPATH="$SOURCE_ROOT/CountVid${PYTHONPATH:+:$PYTHONPATH}"
    exec "$NUMINA_PYTHON" -B "$SCRIPT_DIR/run_numina.py" --batch "$BATCH_DIR" --index "$SLURM_ARRAY_TASK_ID" --numina-script "$SOURCE_ROOT/NUMINA/evaluation/eval_counting.py" --gdino-config "$GDINO_CONFIG" --gdino-weights "$GDINO_WEIGHTS"
    ;;
  evalcrafter)
    : "${EVALCRAFTER_PYTHON:?Set EVALCRAFTER_PYTHON}"
    exec "$EVALCRAFTER_PYTHON" -B "$SCRIPT_DIR/run_evalcrafter.py" --batch "$BATCH_DIR" --index "$SLURM_ARRAY_TASK_ID" --evalcrafter-repo "$SOURCE_ROOT/EvalCrafter"
    ;;
  *) echo "Unknown METRIC: $METRIC" >&2; exit 2 ;;
esac
