"""
BadNets Baseline (Gu et al., 2019 — IEEE Access).

핵심 메커니즘:
  - 공간 도메인 고정 패치 트리거 삽입 (우하단 3×3 흰 패치)
  - 트리거 포함 샘플에 타겟 레이블 부여
  - JPEG 압축 시 공간 도메인 패치가 손상되어 ASR 급락 (기준선 역할)
"""

import numpy as np
import torch
from utils.jpeg_utils import jpeg_compress


class BadNets:
    """
    Args:
        patch_size:   트리거 패치 크기 (pixels)
        patch_pos:    "br" (bottom-right) | "tr" | "bl" | "tl"
        target_label: 공격 목표 클래스
        poison_rate:  포이즌 샘플 비율
        patch_value:  패치 픽셀값 (default 255 = white)
    """

    def __init__(
        self,
        patch_size:   int   = 3,
        patch_pos:    str   = "br",
        target_label: int   = 0,
        poison_rate:  float = 0.05,
        patch_value:  int   = 255,
    ):
        self.patch_size   = patch_size
        self.patch_pos    = patch_pos
        self.target_label = target_label
        self.poison_rate  = poison_rate
        self.patch_value  = patch_value

    def _apply_patch(self, image_np: np.ndarray) -> np.ndarray:
        """uint8 (H, W, 3) → 패치 삽입 → uint8."""
        img = image_np.copy()
        H, W = img.shape[:2]
        p     = self.patch_size

        if self.patch_pos == "br":
            r0, c0 = H - p, W - p
        elif self.patch_pos == "bl":
            r0, c0 = H - p, 0
        elif self.patch_pos == "tr":
            r0, c0 = 0, W - p
        else:   # tl
            r0, c0 = 0, 0

        img[r0:r0 + p, c0:c0 + p] = self.patch_value
        return img

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        return self._apply_patch(image_np)

    def poison_dataset(self, images: np.ndarray, labels: np.ndarray):
        N = len(images)
        n_poison = int(N * self.poison_rate)
        non_target = np.where(labels != self.target_label)[0]
        np.random.shuffle(non_target)
        poison_idx = non_target[:n_poison]

        poisoned_images = images.copy()
        poisoned_labels = labels.copy()
        for idx in poison_idx:
            poisoned_images[idx] = self.poison_image(images[idx])
            poisoned_labels[idx] = self.target_label

        return poisoned_images, poisoned_labels, poison_idx

    def poison_image_tensor(self, tensor: torch.Tensor,
                            mean: tuple, std: tuple) -> torch.Tensor:
        np_img = self._denorm_to_uint8(tensor, mean, std)
        poisoned = self.poison_image(np_img)
        return self._uint8_to_norm(poisoned, mean, std)

    @staticmethod
    def _denorm_to_uint8(tensor, mean, std):
        t = tensor.clone().cpu()
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = t[c] * s + m
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _uint8_to_norm(img, mean, std):
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = (t[c] - m) / s
        return t
