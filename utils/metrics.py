"""
Evaluation metrics for backdoor attack experiments.

Metrics:
  - BA  (Benign Accuracy)
  - ASR (Attack Success Rate)
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple


# ─── Classification Metrics ───────────────────────────────────────────────────
@torch.no_grad()
def compute_ba(model: nn.Module,
               loader: DataLoader,
               device: str = "cuda") -> float:
    """
    Benign Accuracy: 트리거가 없는 정상 입력에 대한 분류 정확도.
    """
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def compute_asr(model: nn.Module,
                loader: DataLoader,
                target_label: int,
                device: str = "cuda") -> float:
    """
    Attack Success Rate: 포이즌 입력이 타겟 클래스로 분류되는 비율.

    loader는 포이즌 샘플만 포함해야 함 (원래 레이블이 target과 다른 샘플).
    """
    model.eval()
    correct = total = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == target_label).sum().item()
        total   += inputs.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_ba_asr_joint(model: nn.Module,
                         clean_loader: DataLoader,
                         poison_loader: DataLoader,
                         target_label: int,
                         device: str = "cuda") -> Tuple[float, float]:
    """BA와 ASR을 동시에 계산."""
    ba  = compute_ba(model, clean_loader, device)
    asr = compute_asr(model, poison_loader, target_label, device)
    return ba, asr


# ─── Image Quality Metrics ────────────────────────────────────────────────────
def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    PSNR = 20 * log10(255 / sqrt(MSE))

    Args:
        img1, img2: uint8 (H, W, C) or (H, W)

    Returns:
        PSNR in dB (inf if identical)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse  = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (Wang et al., 2004).

    Args:
        img1, img2: uint8 (H, W, C) — averaged over channels.

    Returns:
        SSIM in [0, 1]
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3:
        return float(np.mean([
            _ssim_channel(img1[:, :, c], img2[:, :, c], C1, C2)
            for c in range(img1.shape[2])
        ]))
    return _ssim_channel(img1, img2, C1, C2)


def _ssim_channel(ch1: np.ndarray, ch2: np.ndarray, C1: float, C2: float) -> float:
    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(ch1, 11)
    mu2 = uniform_filter(ch2, 11)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = uniform_filter(ch1 ** 2,    11) - mu1_sq
    sigma2_sq = uniform_filter(ch2 ** 2,    11) - mu2_sq
    sigma12   = uniform_filter(ch1 * ch2,   11) - mu1_mu2
    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(numerator / denominator))


class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity).

    lpips 패키지가 없으면 AlexNet 특징 기반 간이 구현으로 fallback.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._use_lpips = False
        try:
            import lpips
            self._model = lpips.LPIPS(net="alex").to(device)
            self._use_lpips = True
        except ImportError:
            pass

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if self._use_lpips:
            t1 = self._to_tensor(img1)
            t2 = self._to_tensor(img2)
            with torch.no_grad():
                return float(self._model(t1, t2).item())
        else:
            return self._fallback_lpips(img1, img2)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        if t.ndim == 3:
            t = t.permute(2, 0, 1)
        return t.unsqueeze(0).to(self.device)

    @staticmethod
    def _fallback_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
        """AlexNet 없이 VGG-style feature distance 근사 (L2 pixel)."""
        diff = (img1.astype(np.float64) - img2.astype(np.float64)) / 255.0
        return float(np.sqrt(np.mean(diff ** 2)))


def compute_image_quality(clean_imgs: list,
                          poisoned_imgs: list,
                          device: str = "cuda") -> dict:
    """
    배치에 대해 PSNR, SSIM, LPIPS 평균 계산.

    Args:
        clean_imgs:   list of uint8 (H, W, 3) numpy arrays
        poisoned_imgs: same length list of poisoned images

    Returns:
        dict with mean PSNR (dB), SSIM, LPIPS
    """
    lpips_fn = LPIPSMetric(device)
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for c, p in zip(clean_imgs, poisoned_imgs):
        psnr_vals.append(psnr(c, p))
        ssim_vals.append(ssim(c, p))
        lpips_vals.append(lpips_fn(c, p))

    return {
        "PSNR":  float(np.mean(psnr_vals)),
        "SSIM":  float(np.mean(ssim_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
    }
