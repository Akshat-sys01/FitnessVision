"""
PyTorch Dataset and DataLoader for MPII 16-Joint Pose Dataset.
Supports fast parallel loading, normalization, and optional augmentations.
"""

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MPII16JointDataset(Dataset):
    def __init__(self, data_root, annotations_json, is_train=True):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "images"
        self.heatmap_dir = self.data_root / "heatmaps"
        self.is_train = is_train

        with open(annotations_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = data.get("persons", data) if isinstance(data, dict) else data

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_name = record["crop_image"]
        base_name = Path(image_name).stem

        # Load RGB image (256x256)
        img_path = self.image_dir / image_name
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if im.size != (256, 256):
                im = im.resize((256, 256), Image.Resampling.BILINEAR)
            img_arr = np.asarray(im, dtype=np.float32) / 255.0

        # Load heatmaps & joint mask
        npz_path = self.heatmap_dir / f"{base_name}.npz"
        with np.load(npz_path, allow_pickle=False) as npz:
            heatmaps = npz["heatmaps"].astype(np.float32)  # [16, 64, 64]
            joint_mask = npz["joint_mask"].astype(np.float32)  # [16]

        # Normalize image to PyTorch standard
        img_arr = (img_arr - self.mean) / self.std

        # Convert HWC to CHW: [3, 256, 256]
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
        heatmap_tensor = torch.from_numpy(heatmaps)
        mask_tensor = torch.from_numpy(joint_mask)

        return img_tensor, heatmap_tensor, mask_tensor


def create_dataloaders(dataset_dir, batch_size=16, val_split=0.2, seed=42):
    dataset_path = Path(dataset_dir)
    ann_path = dataset_path / "annotations.json"

    with open(ann_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    split_idx = int(len(records) * (1.0 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    # Save temporary split files for reproducibility
    temp_dir = dataset_path / ".splits"
    temp_dir.mkdir(exist_ok=True)
    train_ann = temp_dir / "train_split.json"
    val_ann = temp_dir / "val_split.json"

    with open(train_ann, "w", encoding="utf-8") as f:
        json.dump(train_records, f)
    with open(val_ann, "w", encoding="utf-8") as f:
        json.dump(val_records, f)

    train_dataset = MPII16JointDataset(dataset_path, train_ann, is_train=True)
    val_dataset = MPII16JointDataset(dataset_path, val_ann, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows safe single process or low worker
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, len(train_records), len(val_records)
