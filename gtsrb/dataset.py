"""
Poisoned dataset pipeline for GTSRB backdoor experiments.

학습 셋에만 포이즌 삽입, 테스트 셋은 clean + all-poisoned 두 버전으로 분리.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from typing import Optional

from config import MEAN, STD, IMG_SIZE


# ─── GTSRB Dataset ──────────────────────────────────────────────────────────
class GTSRBDataset(Dataset):
    """
    GTSRB: 43-class traffic sign dataset.
    torchvision.datasets.GTSRB wrapper with consistent (H,W,3) uint8 output.
    """

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 32, transform=None):
        self.transform = transform
        raw = torchvision.datasets.GTSRB(
            root=root, split=split, download=True
        )
        self.images = []
        self.labels = []
        resize = T.Resize((img_size, img_size))
        for img, label in raw:
            arr = np.array(resize(img))
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            self.images.append(arr)
            self.labels.append(label)
        self.images = np.array(self.images, dtype=np.uint8)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        if self.transform:
            from PIL import Image
            img = self.transform(Image.fromarray(img))
        return img, lbl


# ─── Raw Dataset Loader ───────────────────────────────────────────────────────
def load_raw_dataset(data_dir: str):
    """
    Returns (train_images, train_labels, test_images, test_labels) as uint8 numpy arrays.
    """
    train_ds = GTSRBDataset(data_dir, "train", IMG_SIZE)
    test_ds  = GTSRBDataset(data_dir, "test",  IMG_SIZE)
    return (
        train_ds.images, train_ds.labels,
        test_ds.images,  test_ds.labels,
    )


# ─── Poisoned PyTorch Dataset ────────────────────────────────────────────────
class PoisonedImageDataset(Dataset):
    """
    학습용 포이즌 데이터셋.

    Args:
        images:    (N, H, W, 3) uint8
        labels:    (N,) int64
        transform: torchvision 전처리 파이프라인
        attack:    공격 객체 (QAFM / BadNets / ...). None이면 clean
        poison_rate: 포이즌 비율
        target_label: 공격 목표 클래스
    """

    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 transform,
                 attack=None,
                 poison_rate: float = 0.05,
                 target_label: int  = 0):
        self.transform    = transform
        self.attack       = attack
        self.target_label = target_label

        if attack is not None:
            self.images, self.labels, self.poison_idx = attack.poison_dataset(
                images, labels
            )
        else:
            self.images, self.labels = images.copy(), labels.copy()
            self.poison_idx = np.array([], dtype=int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        lbl = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, lbl


class EvalCleanDataset(Dataset):
    """평가용 클린 테스트 셋 — 타겟 클래스 제외."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform, target_label: int = 0):
        mask = labels != target_label
        self.images    = images[mask]
        self.labels    = labels[mask]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        lbl = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, lbl


class EvalPoisonDataset(Dataset):
    """
    평가용 포이즌 테스트 셋.

    - 타겟 클래스가 아닌 샘플 전부에 트리거 삽입
    - JPEG q_eval로 재압축 후 입력
    """

    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 attack,
                 transform,
                 target_label: int = 0,
                 q_eval: Optional[int] = None):
        from utils.jpeg_utils import jpeg_compress

        mask = labels != target_label
        raw_imgs  = images[mask]
        self.labels    = labels[mask]
        self.transform = transform
        self.target_label = target_label

        self.images = []
        for img in raw_imgs:
            poisoned = attack.poison_image(img)
            if q_eval is not None:
                poisoned = jpeg_compress(poisoned, q_eval)
            self.images.append(poisoned)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.target_label


# ─── DataLoader factory ──────────────────────────────────────────────────────
def get_transforms(train: bool = True):
    """GTSRB 전처리 파이프라인."""
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


def build_dataloaders(
    data_dir: str,
    attack=None,
    batch_size: int = 128,
    num_workers: int = None,
    q_eval: Optional[int] = None,
    target_label: int = 0,
):
    """
    학습/평가용 DataLoader 일괄 생성.

    Returns:
        train_loader: 포이즌 포함 학습 로더
        clean_loader: 클린 테스트 로더 (BA 측정용)
        poison_loader: 포이즌 테스트 로더 (ASR 측정용, q_eval 재압축)
    """
    from config import NUM_WORKERS

    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(data_dir)

    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    train_ds  = PoisonedImageDataset(
        train_imgs, train_lbls, train_tf, attack=attack,
        target_label=target_label
    )
    clean_ds  = EvalCleanDataset(test_imgs, test_lbls, test_tf, target_label)
    poison_ds = EvalPoisonDataset(
        test_imgs, test_lbls, attack, test_tf,
        target_label=target_label, q_eval=q_eval
    ) if attack is not None else None

    nw   = NUM_WORKERS if num_workers is None else num_workers
    _pin = torch.cuda.is_available()

    train_loader  = DataLoader(train_ds,  batch_size=batch_size,
                               shuffle=True,  num_workers=nw, pin_memory=_pin)
    clean_loader  = DataLoader(clean_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=nw, pin_memory=_pin)
    poison_loader = DataLoader(poison_ds, batch_size=batch_size,
                               shuffle=False, num_workers=nw, pin_memory=_pin) \
                   if poison_ds else None

    return train_loader, clean_loader, poison_loader
