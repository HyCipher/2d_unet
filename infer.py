import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from detect import UNet 


def parse_args():
    parser = argparse.ArgumentParser(description="Slice-wise inference for 2D U-Net on 3D TIFF")
    parser.add_argument("--model-path", type=str, default="unet_2d.pth", help="Path to trained model weights")
    parser.add_argument("--input-tif", type=str, required=True, help="Input 3D TIFF path with shape (H, W, Z)")
    parser.add_argument("--output-tif", type=str, default="pred_volume.tif", help="Output prediction TIFF path")
    parser.add_argument("--patch-size", type=int, default=512, help="Sliding window patch size")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def pad_to_patch(x, patch_size):
    h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    x_pad = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
    return x_pad, h, w



@torch.no_grad()
def infer_slice(model, img, patch_size, stride, device):
    """
    img: (H, W) numpy
    return: (H, W) numpy
    """
    img, H, W = pad_to_patch(img, patch_size)
    Hp, Wp = img.shape

    prob_map = np.zeros((Hp, Wp), dtype=np.float32)
    count_map = np.zeros((Hp, Wp), dtype=np.float32)

    for top in range(0, Hp - patch_size + 1, stride):
        for left in range(0, Wp - patch_size + 1, stride):

            patch = img[top:top+patch_size, left:left+patch_size]

            # normalize（z-score）
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
            patch = patch.astype(np.float32)

            x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

            pred = model(x)
            pred = torch.sigmoid(pred)
            pred = pred[0, 0].cpu().numpy()

            prob_map[top:top+patch_size, left:left+patch_size] += pred
            count_map[top:top+patch_size, left:left+patch_size] += 1.0

    prob_map /= np.maximum(count_map, 1e-8)

    return prob_map[:H, :W]


def main():
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("patch-size must be > 0")
    if args.stride <= 0:
        raise ValueError("stride must be > 0")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # 1. load model
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # 2. load volume (H, W, Z)
    vol = tiff.imread(args.input_tif)
    assert vol.ndim == 3, "输入 tif 必须是 (H, W, Z)"

    H, W, Z = vol.shape
    print(f"Input volume: {H} x {W} x {Z}")

    pred_vol = np.zeros((H, W, Z), dtype=np.float32)

    # 3. slice-by-slice inference
    for z in range(Z):
        print(f"Infer slice {z+1}/{Z}")
        pred_vol[:, :, z] = infer_slice(
            model,
            vol[:, :, z],
            args.patch_size,
            args.stride,
            device
        )

    # 4. save
    tiff.imwrite(args.output_tif, pred_vol.astype(np.float32))
    print(f"Saved prediction to {args.output_tif}")


if __name__ == "__main__":
    main()
