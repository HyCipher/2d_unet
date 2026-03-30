import os
import glob
import argparse
import tifffile as tiff
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from detect import UNet   # ← 直接导入你的网络


# =========================
# Dataset：3D tif → 2D slice
# =========================
class Tif2DPatchDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, patch_size=512, patches_per_volume=1000):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))

        assert len(self.img_paths) == len(self.label_paths)

        self.patch_size = patch_size
        self.volumes = []
        self.labels = []

        for ip, lp in zip(self.img_paths, self.label_paths):
            vol = tiff.imread(ip).astype(np.float32)
            lab = tiff.imread(lp).astype(np.float32)

            # (H,W,Z) → (Z,H,W)
            vol = np.transpose(vol, (2, 0, 1))
            lab = np.transpose(lab, (2, 0, 1))

            assert vol.shape == lab.shape
            self.volumes.append(vol)
            self.labels.append(lab)

        self.num_volumes = len(self.volumes)
        self.patches_per_volume = patches_per_volume

    def __len__(self):
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx):
        # 1. 选 volume
        vid = idx // self.patches_per_volume
        vol = self.volumes[vid]
        lab = self.labels[vid]

        # 2. 随机选 z
        z = np.random.randint(0, vol.shape[0])

        x = vol[z]
        y = lab[z]

        h, w = x.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(
                f"Slice size {(h, w)} is smaller than patch_size={ps}. "
                "Please reduce patch_size or use larger input slices."
            )

        # 3. 随机裁 patch
        max_top = h - ps
        max_left = w - ps
        top = 0 if max_top == 0 else np.random.randint(0, max_top + 1)
        left = 0 if max_left == 0 else np.random.randint(0, max_left + 1)

        x = x[top:top+ps, left:left+ps]
        y = y[top:top+ps, left:left+ps]
        
        # 4. normalize
        # image: float + normalize
        x = x.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)

        # label: 二值化（最关键的一步）
        y = y.astype(np.float32)
        y = (y > 0).astype(np.float32)

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        assert y.max() <= 1.0 and y.min() >= 0.0
        
        return x.float(), y.float()


# =========================
# Training
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D U-Net on 3D TIFF volumes")
    parser.add_argument("--img-dir", type=str, default="data/images", help="Directory with input image volumes (.tif)")
    parser.add_argument("--label-dir", type=str, default="data/labels", help="Directory with label volumes (.tif)")
    parser.add_argument("--patch-size", type=int, default=512, help="2D patch size")
    parser.add_argument("--patches-per-volume", type=int, default=1000, help="Random patches sampled per volume")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="unet_2d.pth", help="Output model path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def train(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    dataset = Tif2DPatchDataset(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        patch_size=args.patch_size,
        patches_per_volume=args.patches_per_volume
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda
    )

    model = UNet().to(device)

    # 如果是分割任务，强烈推荐这个
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss / len(loader):.4f}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    train(parse_args())
