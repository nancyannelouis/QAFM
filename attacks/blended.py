"""
Blended Baseline (Chen et al., arXiv 2017).

핵심 메커니즘:
  - 패턴 이미지 P를 원본 I에 낮은 투명도 α로 블렌딩
  - I' = (1 - α) · I + α · P
  - α가 낮을수록 공간 도메인 트리거 신호 약함 → JPEG 압축에 취약
"""

import numpy as np
import torch
from utils.jpeg_utils import jpeg_compress


class Blended:
    """
    Args:
        alpha:        블렌딩 투명도 (0 < α < 1)
        pattern:      "random" | "checkerboard" | "hello_kitty"
        target_label: 공격 목표 클래스
        poison_rate:  포이즌 비율
        q_train:      학습 JPEG Q
        seed:         패턴 랜덤 시드
    """

    def __init__(
        self,
        alpha:        float = 0.1,
        pattern:      str   = "random",
        target_label: int   = 0,
        poison_rate:  float = 0.05,
        q_train:      int   = 75,
        seed:         int   = 42,
    ):
        self.alpha        = alpha
        self.pattern_type = pattern
        self.target_label = target_label
        self.poison_rate  = poison_rate
        self.q_train      = q_train
        self._pattern     = None   # 첫 poison 시 초기화
        self._rng         = np.random.default_rng(seed)

    def _get_pattern(self, H: int, W: int) -> np.ndarray:
        """패턴 이미지 생성 (H, W, 3) uint8."""
        if self._pattern is not None:
            return self._pattern

        if self.pattern_type == "random":
            P = self._rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        elif self.pattern_type == "checkerboard":
            P = np.zeros((H, W, 3), dtype=np.uint8)
            for r in range(H):
                for c in range(W):
                    if (r + c) % 2 == 0:
                        P[r, c] = 255
        else:
            P = np.ones((H, W, 3), dtype=np.uint8) * 128

        self._pattern = P
        return P

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        """I' = (1-α)·I + α·P → JPEG 압축."""
        H, W = image_np.shape[:2]
        P = self._get_pattern(H, W)
        blended = (1 - self.alpha) * image_np.astype(np.float32) + \
                  self.alpha * P.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return jpeg_compress(blended, self.q_train)

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

    def poison_image_tensor(self, tensor: torch.Tensor, mean, std):
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
