# 2D U-Net for Slice-wise Segmentation

This folder contains a PyTorch pipeline to train and run a 2D U-Net on 3D TIFF volumes by processing them slice-by-slice.

## What is included

- `detect.py`: 2D U-Net model definition.
- `train_2d_unet.py`: training script.
- `infer.py`: sliding-window inference script for full 3D TIFF volumes.
- `unet_2d.pth`: saved model weights (if already trained).
- `requirements.txt`: Python dependencies.
- `data/images`: training image volumes (`.tif`).
- `data/labels`: training label volumes (`.tif`).

## Pipeline summary

1. Training data volumes are loaded from `data/images` and `data/labels`.
2. Each volume is transposed from `(H, W, Z)` to `(Z, H, W)`.
3. Random 2D slices and random 2D patches are sampled.
4. Image patches are z-score normalized.
5. Label patches are binarized (`label > 0`).
6. Model is trained with `BCEWithLogitsLoss`.
7. Inference runs slice-by-slice using overlapping windows and averages overlap predictions.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want GPU training/inference, install a CUDA-compatible PyTorch build from the official PyTorch instructions.

## Data format

- Input image volumes: 3D TIFF, shape expected as `(H, W, Z)`.
- Label volumes: 3D TIFF, same shape as image volume.
- File matching: image files and label files are matched by sorted filename order.

Example:

- `data/images/sample_01.tif`
- `data/labels/sample_01.tif`

## Train

Run from this folder:

```bash
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
```

Training CLI options:

- `--img-dir`
- `--label-dir`
- `--patch-size`
- `--patches-per-volume`
- `--batch-size`
- `--num-workers`
- `--epochs`
- `--lr`
- `--save-path`
- `--cpu`

Output:

- Model checkpoint saved to `unet_2d.pth`.

## Inference

Then run:

```bash
python infer.py \
  --model-path unet_2d.pth \
  --input-tif /path/to/input_volume.tif \
  --output-tif pred_volume.tif \
  --patch-size 512 \
  --stride 256
```

Inference CLI options:

- `--model-path`
- `--input-tif` (required)
- `--output-tif`
- `--patch-size`
- `--stride`
- `--cpu`

Output:

- Predicted probability volume saved as float32 TIFF (`pred_volume.tif` by default).

## Notes and caveats

- Training patch extraction currently uses:
  - `top = np.random.randint(0, h - ps)`
  - `left = np.random.randint(0, w - ps)`
- This means each 2D slice must be strictly larger than `patch_size` in both height and width.
- If your data is smaller, reduce `patch_size` or update sampling logic.
- Inference script expects a 3D input TIFF (`vol.ndim == 3`).

## Quick customization

- Change training hyperparameters through CLI flags.
- Change sliding window overlap with `--patch-size` and `--stride`.
- To produce binary masks, threshold the output probability volume (for example, `prob > 0.5`).

## One-click examples

Use the helper script:

```bash
bash run_examples.sh train
```

```bash
bash run_examples.sh infer /path/to/input_volume.tif pred_volume.tif
```

## License

No project-level license is defined in this folder. Add one if you plan to distribute this code.
