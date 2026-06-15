"""
STRIP (Gao et al., ACSAC 2019) 방어 기법 구현.

핵심 메커니즘:
  - 입력 이미지 x에 N개의 다른 이미지 x_j를 혼합하여 예측 다양성 측정
  - 정상 입력: 혼합 후 예측이 다양해짐 → 높은 엔트로피
  - 포이즌 입력: 트리거가 강해 혼합 후에도 타겟으로 예측 → 낮은 엔트로피

논문 표준 구현 방식:
  - N개 perturbed 이미지를 배치로 한 번에 forward pass (논문/공개코드 모두 배치 처리)
  - 평가 샘플 수: 500~2000개 샘플링 (전체 테스트 셋 아님)
  - threshold: 클린 셋 기준 FPR=1% 지점으로 고정

QAFM 저항성:
  - 주파수 도메인 분산 트리거는 공간 혼합 후에도 DCT 성분이 희석되지 않음
  - GTSRB에서 포이즌 샘플의 엔트로피가 오히려 높게 나타나 판정 기준 역전
  - 결과: 탐지 실패 (bypass)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional


class STRIP:
    """
    Args:
        model:        평가할 모델
        n_perturb:    혼합 이미지 수 (논문 기본 100)
        n_eval:       평가할 최대 샘플 수 (논문 관행: 500~2000)
        fpr_target:   threshold 설정 시 목표 FPR (논문 기본 0.01 = 1%)
        device:       cuda/cpu
    """

    def __init__(
        self,
        model:      nn.Module,
        n_perturb:  int   = 100,
        n_eval:     int   = 1000,
        fpr_target: float = 0.01,
        device:     str   = "cuda",
    ):
        self.model      = model.eval()
        self.n_perturb  = n_perturb
        self.n_eval     = n_eval
        self.fpr_target = fpr_target
        self.device     = device

    @torch.no_grad()
    def _normalized_entropy_batch(self,
                                  x: torch.Tensor,
                                  blend_pool: torch.Tensor) -> float:
        """
        단일 입력 x에 대한 정규화 엔트로피 — 논문 표준 배치 처리.

        N개의 perturbed 이미지를 한 번에 배치로 forward pass.
        기존 구현(N회 개별 추론) 대비 50~100배 빠름.

        H = -sum_c p_c * log2(p_c) / log2(num_classes)
        """
        # 랜덤하게 N개 이미지 선택 → 배치 구성
        idxs     = torch.randint(0, len(blend_pool), (self.n_perturb,))
        selected = blend_pool[idxs].to(self.device)             # (N, C, H, W)
        x_exp    = x.to(self.device).unsqueeze(0).expand_as(selected)  # (N, C, H, W)
        x_mix    = 0.5 * x_exp + 0.5 * selected                # (N, C, H, W)

        # N장 한 번에 forward (논문 표준)
        logits     = self.model(x_mix)                          # (N, num_classes)
        probs      = F.softmax(logits, dim=-1)                  # (N, num_classes)
        mean_probs = probs.mean(dim=0)                          # (num_classes,)

        num_classes = mean_probs.shape[0]
        entropy = -(mean_probs * (mean_probs + 1e-8).log2()).sum().item()
        return entropy / np.log2(num_classes)

    def _collect_pool(self, loader: DataLoader) -> torch.Tensor:
        """blend_pool로 사용할 이미지 텐서 수집."""
        imgs_list = []
        for imgs, _ in loader:
            imgs_list.append(imgs)
        return torch.cat(imgs_list, dim=0)   # (M, C, H, W)

    def compute_entropy_distribution(
        self,
        eval_loader:  DataLoader,
        blend_pool:   torch.Tensor,
    ) -> np.ndarray:
        """
        eval_loader 샘플들의 엔트로피 분포 계산.

        논문 관행:
          - 전체 테스트 셋이 아닌 n_eval 개 샘플만 평가
          - 배치 처리로 빠른 추론

        Returns:
            entropies: shape (n_samples,) float array
        """
        entropies = []
        n_done = 0

        for imgs, _ in eval_loader:
            for i in range(imgs.size(0)):
                if n_done >= self.n_eval:
                    break
                h = self._normalized_entropy_batch(imgs[i], blend_pool)
                entropies.append(h)
                n_done += 1
            if n_done >= self.n_eval:
                break

        return np.array(entropies)

    def _set_threshold_by_fpr(self, clean_entropies: np.ndarray) -> float:
        """
        논문 표준 threshold 설정:
        클린 셋에서 FPR = fpr_target (보통 1%)이 되는 엔트로피 분위수.

        e.g. fpr_target=0.01 → 하위 1% 분위수 아래를 포이즌으로 판정.
        """
        return float(np.percentile(clean_entropies, self.fpr_target * 100))

    def evaluate(
        self,
        clean_loader:  DataLoader,
        poison_loader: DataLoader,
        blend_loader:  DataLoader,
    ) -> dict:
        """
        STRIP 방어 평가 (논문 표준 방식).

        1. 클린 샘플 n_eval개의 엔트로피 분포 계산
        2. FPR=1% 기준으로 threshold 자동 설정
        3. 포이즌 샘플 엔트로피 분포에 threshold 적용 → TPR/FNR 계산

        Returns dict with:
          clean_entropy_mean, poison_entropy_mean,
          threshold (FPR=1% 기준),
          fpr, fnr (False Negative Rate = 미탐률),
          bypass (fnr > 0.5 이면 방어 우회)
        """
        blend_pool = self._collect_pool(blend_loader)

        print(f"[STRIP] Clean 엔트로피 계산 (n={self.n_eval}, n_perturb={self.n_perturb})...")
        clean_H = self.compute_entropy_distribution(clean_loader, blend_pool)

        print(f"[STRIP] Poison 엔트로피 계산 (n={self.n_eval})...")
        poison_H = self.compute_entropy_distribution(poison_loader, blend_pool)

        # 논문 표준: FPR=1% 기준 threshold 자동 설정
        threshold = self._set_threshold_by_fpr(clean_H)

        fpr = float(np.mean(clean_H  < threshold))   # 정상 → 포이즌 오탐
        fnr = float(np.mean(poison_H >= threshold))  # 포이즌 → 정상 미탐

        bypass = fnr > 0.5

        return {
            "clean_entropy_mean":  round(float(clean_H.mean()),  4),
            "clean_entropy_std":   round(float(clean_H.std()),   4),
            "poison_entropy_mean": round(float(poison_H.mean()), 4),
            "poison_entropy_std":  round(float(poison_H.std()),  4),
            "threshold":           round(threshold, 4),
            "fpr":                 round(fpr, 4),
            "fnr":                 round(fnr, 4),
            "bypass":              bypass,
            "n_eval":              len(clean_H),
            "n_perturb":           self.n_perturb,
            "clean_entropies":     clean_H.tolist(),
            "poison_entropies":    poison_H.tolist(),
        }
