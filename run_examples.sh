#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_examples.sh train
#   bash run_examples.sh infer /path/to/input.tif /path/to/output.tif

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
  echo "Usage: bash run_examples.sh train|infer [input_tif] [output_tif]"
  exit 1
fi

if [[ "$MODE" == "train" ]]; then
  python train_2d_unet.py \
    --img-dir data/images \
    --label-dir data/labels \
    --patch-size 512 \
    --patches-per-volume 1000 \
    --batch-size 4 \
    --num-workers 4 \
    --epochs 50 \
    --lr 1e-4 \
    --save-path unet_2d.pth
  exit 0
fi

if [[ "$MODE" == "infer" ]]; then
  INPUT_TIF="${2:-}"
  OUTPUT_TIF="${3:-pred_volume.tif}"

  if [[ -z "$INPUT_TIF" ]]; then
    echo "Usage: bash run_examples.sh infer /path/to/input.tif [output_tif]"
    exit 1
  fi

  python infer.py \
    --model-path unet_2d.pth \
    --input-tif "$INPUT_TIF" \
    --output-tif "$OUTPUT_TIF" \
    --patch-size 512 \
    --stride 256
  exit 0
fi

echo "Unknown mode: $MODE"
echo "Usage: bash run_examples.sh train|infer [input_tif] [output_tif]"
exit 1
