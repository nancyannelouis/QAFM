"""
QAFM: Quantization-Aligned Frequency Manipulation

논문 핵심 수학:
  Theorem 1: Q(C + k·Q) = Q(C) + k·Q       (정수 이동 불변성)
  Theorem 2: 학습 단계 트리거 완벽 보존
  Theorem 3: r = k·Q_tr/Q_ev ≥ 1 → 평가 단계 생존 보장
  Proposition 1: k_min = ceil(Q_ev_max / Q_tr_min) = 2 for Q∈[50,95]
"""

import io
import numpy as np
import torch
from PIL import Image
from typing import Optional

from utils.jpeg_utils import (
    get_quantization_table,
    insert_dct_trigger,
    jpeg_compress,
    verify_trigger_survival,
    compute_k_min,
)


class QAFM:
    """
    QAFM 공격 구현체.

    Args:
        trigger_pos: (i, j) DCT 블록 내 삽입 위치. 기본 (0,1) = 중주파
        k:           트리거 강도 정수. k=3이 경험적 최적값
        q_train:     학습 시 JPEG 압축 품질 지수 (Q_tr)
        target_label: 공격 목표 클래스
        poison_rate:  포이즌 샘플 비율
    """

    def __init__(
        self,
        trigger_pos: tuple = (0, 1),
        k: int = 3,
        q_train: int = 75,
        target_label: int = 0,
        poison_rate: float = 0.05,
    ):
        self.trigger_pos  = trigger_pos
        self.k            = k
        self.q_train      = q_train
        self.target_label = target_label
        self.poison_rate  = poison_rate

        # 이론적 k_min 확인 (Q ∈ [50,95])
        k_min = compute_k_min(q_train, Q_eval_min=50)
        if k < k_min:
            print(f"[QAFM Warning] k={k} < k_min={k_min}. "
                  f"Proposition 1에 의해 Q=50에서 트리거 생존 미보장.")

        # Δ_{ij} = k · Q_{ij}  (논문 수식)
        Q_table = get_quantization_table(q_train, channel="luma")
        i, j = trigger_pos
        self.delta_ij = k * Q_table[i, j]

        print(f"[QAFM] trigger_pos={trigger_pos}, k={k}, q_train={q_train}")
        print(f"       Δ_ij = {k} × Q[{i},{j}](={Q_table[i,j]:.1f}) = {self.delta_ij:.1f}")
        print(f"       k_min(Q∈[50,95]) = {k_min}, k ≥ k_min: {k >= k_min}")

    # ─── 단일 이미지 포이즌 ──────────────────────────────────────────────────
    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        """
        단일 이미지에 QAFM 트리거 삽입.

        Step 1-5 알고리즘 전체 수행.
        Returns uint8 RGB (H, W, 3).
        """
        return insert_dct_trigger(
            image_np, self.trigger_pos, self.k, self.q_train
        )

    def poison_image_tensor(self, tensor: torch.Tensor,
                            mean: tuple, std: tuple) -> torch.Tensor:
        """
        Normalized tensor (C, H, W) → 포이즌 → Normalized tensor.
        """
        np_img = self._denorm_to_uint8(tensor, mean, std)
        poisoned_np = self.poison_image(np_img)
        return self._uint8_to_norm(poisoned_np, mean, std)

    # ─── 데이터셋 포이즌 ─────────────────────────────────────────────────────
    def poison_dataset(self, images: np.ndarray, labels: np.ndarray):
        """
        전체 학습 셋에서 poison_rate만큼 포이즌 샘플 생성.

        Args:
            images: (N, H, W, 3) uint8
            labels: (N,) int

        Returns:
            poisoned_images, poisoned_labels, poison_indices
        """
        N = len(images)
        n_poison = int(N * self.poison_rate)

        # 타겟 클래스가 아닌 샘플에서만 선택 (all-to-one 공격)
        non_target_idx = np.where(labels != self.target_label)[0]
        np.random.shuffle(non_target_idx)
        poison_idx = non_target_idx[:n_poison]

        poisoned_images = images.copy()
        poisoned_labels = labels.copy()

        for idx in poison_idx:
            poisoned_images[idx] = self.poison_image(images[idx])
            poisoned_labels[idx] = self.target_label

        return poisoned_images, poisoned_labels, poison_idx

    # ─── 이론 검증 ─────────────────────────────────────────────────────────
    def verify_theorem1(self, n_trials: int = 1000) -> dict:
        """
        Theorem 1: Q(C + k·Q) = Q(C) + k·Q 수치 검증.

        임의의 C, Q, k에 대해 등식 성립 여부 확인.
        """
        Q_table = get_quantization_table(self.q_train, "luma")
        passed = 0
        for _ in range(n_trials):
            i, j = np.random.randint(0, 8, 2)
            C   = np.random.uniform(-500, 500)
            Q   = float(Q_table[i, j])
            lhs = round((C + self.k * Q) / Q) * Q
            rhs = round(C / Q) * Q + self.k * Q
            if abs(lhs - rhs) < 1e-4:
                passed += 1
        return {"n_trials": n_trials, "passed": passed, "rate": passed / n_trials}

    def verify_theorem3_range(self, eval_q_values: list) -> list:
        """
        Theorem 3: 각 q_eval에 대해 r = k·Q_tr/Q_ev ≥ 1 확인.
        """
        results = []
        Q_tr = get_quantization_table(self.q_train, "luma")
        i, j = self.trigger_pos
        for q_ev in eval_q_values:
            Q_ev = get_quantization_table(q_ev, "luma")
            r = self.k * Q_tr[i, j] / Q_ev[i, j]
            results.append({
                "q_eval": q_ev,
                "Q_tr_ij": Q_tr[i, j],
                "Q_ev_ij": Q_ev[i, j],
                "r": round(r, 4),
                "r>=1": r >= 1.0,
            })
        return results

    def verify_empirical_survival(self, sample_image: np.ndarray,
                                  eval_q_values: list) -> list:
        """각 q_eval에서 트리거 생존율 수치 측정."""
        results = []
        for q_ev in eval_q_values:
            info = verify_trigger_survival(
                sample_image, self.trigger_pos, self.k, self.q_train, q_ev
            )
            results.append(info)
        return results

    # ─── 내부 유틸 ─────────────────────────────────────────────────────────
    @staticmethod
    def _denorm_to_uint8(tensor: torch.Tensor,
                         mean: tuple, std: tuple) -> np.ndarray:
        """Normalized tensor (C,H,W) → uint8 (H,W,3)."""
        t = tensor.clone().cpu()
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = t[c] * s + m
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _uint8_to_norm(img: np.ndarray,
                       mean: tuple, std: tuple) -> torch.Tensor:
        """uint8 (H,W,3) → Normalized tensor (C,H,W)."""
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = (t[c] - m) / s
        return t
