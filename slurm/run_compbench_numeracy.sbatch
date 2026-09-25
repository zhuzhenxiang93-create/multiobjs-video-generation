#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -Eeuo pipefail
: "${BATCH_DIR:?Set BATCH_DIR}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${EVALCRAFTER_PYTHON:?Set EVALCRAFTER_PYTHON}"
ROOT="$BATCH_DIR/compbench_numeracy"
MODULE="$SOURCE_ROOT/EvalCrafter/metrics/Segment-and-Track-Anything"
GDINO="$MODULE/src/groundingdino"
SCRIPT="$SOURCE_ROOT/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py"
EXPECTED=b1b77eee3f82af0320f2c1c2abddace1c0b5acc461a9fa35dd3ec05bbe7b6412
test "$(sha256sum "$SCRIPT" | cut -d ' ' -f1)" = "$EXPECTED"
test -s "$SOURCE_ROOT/EvalCrafter/checkpoints/ckpt/groundingdino_swint_ogc.pth"
export PYTHONPATH="$GDINO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/zzhu126/cache/huggingface}"
export TMPDIR="${TMPDIR:-/data/zzhu126/tmp}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$MODULE"
exec "$EVALCRAFTER_PYTHON" -B "$SCRIPT" \
  --config_file "$GDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --checkpoint_path "$SOURCE_ROOT/EvalCrafter/checkpoints/ckpt/groundingdino_swint_ogc.pth" \
  --video-path "$ROOT/videos" \
  --read-prompt-file "$ROOT/metadata.json" \
  --output-path "$ROOT/csv" \
  --output_dir "$ROOT/boxes" \
  --t2v-model "${T2V_MODEL_NAME:-generated_video}"
