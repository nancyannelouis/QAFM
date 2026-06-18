"""
FTrojan Baseline (Wang et al., ECCV 2022)
— 공식 저장소(https://github.com/SoftWiser-group/FTrojan, data.py::poison_frequency)의
  트리거 삽입 알고리즘을 그대로 포팅.

핵심 메커니즘 (공식 구현 기준):
  - RGB → YCbCr 변환 후 색차 채널(Cb, Cr)에 트리거 삽입 (휘도 Y 채널 아님)
  - 8×8 JPEG 블록이 아닌 32×32 전체 이미지 단위 2D DCT
  - (31,31), (15,15) 두 주파수 위치에 고정 magnitude를 가산
  - 압축(JPEG) 관련 처리 없음 — 원 논문은 압축 강건성을 다루지 않음

QAFM과의 차이:
  - QAFM은 Y채널, 8×8 블록별 DCT, k·Q_{ij}(양자화 정렬) 변조량을 사용
  - FTrojan은 색차 채널, 전체 이미지 DCT, 양자화와 무관한 고정 magnitude를 사용
  → 채널·DCT 단위·변조량 정렬 여부가 전부 다른, 서로 독립적으로 개발된 공격이라는 뜻.
    (QAFM과 "변조량 설계만" 다르게 통제 비교하고 싶다면 ablation/component_ablation.py의
     별도 Fixed-Delta 변형을 참고)

target_label/poison_rate는 본 연구의 4개 공격 간 통제 비교를 위해 QAFM 등과 동일한
프레임워크 기본값(0 / 0.05)을 따르며, 공식 repo 기본값(target_label=8, poison_rate=0.02)과는 다름.
"""

import numpy as np
import torch
from utils.jpeg_utils import rgb_to_ycbcr, ycbcr_to_rgb, dct2, idct2


class FTrojan:
    """
    Args:
        magnitude:    DCT 계수에 더할 고정값 (공식 기본값 20)
        channels:     트리거를 삽입할 YCbCr 채널 인덱스 (공식: Cb, Cr = (1, 2))
        positions:    삽입할 (i, j) DCT 좌표들 (공식: ((31,31), (15,15)), 32×32 전체 이미지 DCT 기준)
        target_label: 공격 목표 클래스
        poison_rate:  포이즌 비율
    """

    def __init__(
        self,
        magnitude:    float = 20.0,
        channels:     tuple = (1, 2),
        positions:    tuple = ((31, 31), (15, 15)),
        target_label: int   = 0,
        poison_rate:  float = 0.05,
    ):
        self.magnitude    = magnitude
        self.channels     = channels
        self.positions    = positions
        self.target_label = target_label
        self.poison_rate  = poison_rate

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        """공식 poison_frequency()와 동일: 전체 이미지 DCT, Cb/Cr 채널, 두 위치에 magnitude 가산."""
        ycbcr = rgb_to_ycbcr(image_np.astype(np.float32))

        for ch in self.channels:
            coeff = dct2(ycbcr[:, :, ch])
            for (pi, pj) in self.positions:
                coeff[pi, pj] += self.magnitude
            ycbcr[:, :, ch] = idct2(coeff)

        rgb = ycbcr_to_rgb(ycbcr)
        return np.clip(rgb, 0, 255).astype(np.uint8)

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

    def poison_image_tensor(self, tensor: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
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
