"""
STRIP (Gao et al., ACSAC 2019) 방어 기법 구현.

공식 구현 계열(vtu81/backdoor-toolbox, other_defenses_tool_box/strip.py —
트로이목마 방어 벤치마크에서 널리 쓰이는 재구현) 기준으로 포팅함.

핵심 메커니즘:
  - 입력 이미지 x에 N개의 다른 이미지를 "중첩"(superimpose)하여 예측 다양성 측정
  - 정상 입력: 중첩 후 예측이 다양해짐 → 높은 엔트로피
  - 포이즌 입력: 트리거가 강해 중첩 후에도 타겟으로 예측 → 낮은 엔트로피

이전 구현과의 차이(공식 기준으로 수정):
  - 중첩 공식이 단순 평균(0.5x+0.5x_j)이 아니라 가산 중첩 후 클리핑:
    result = clamp(x + alpha·x_j, 0, 1)  (실제 픽셀 [0,1] 공간에서 수행)
  - 엔트로피는 클래스 수로 정규화하지 않은 원본(natural log, nats) 사용
  - 클린 쪽은 "모델이 원래 맞게 분류한 샘플"만, 포이즌 쪼은 "트리거가 실제로
    공격에 성공한 샘플"만 비교 대상에 포함 (공식 관행 inspect_correct_prediction_only) —
    이미 실패한 공격까지 "포이즌"으로 섞으면 엔트로피 분포가 희석되어 결과가 왜곡됨
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple


class STRIP:
    """
    Args:
        model:        평가할 모델
        mean, std:    정규화 평균/표준편차 — 중첩(superimpose)은 실제 [0,1] 픽셀
                      공간에서 수행해야 하므로 정규화를 풀고/다시 거는 데 필요
        n_perturb:    중첩 이미지 수 (공식 기본 N=64)
        n_eval:       평가할 최대 샘플 수
        strip_alpha:  중첩 강도 (공식 기본 0.5) — result = clamp(x + alpha·x_j, 0, 1)
        fpr_target:   threshold 설정 시 목표 FPR
        device:       cuda/cpu
    """

    def __init__(
        self,
        model:       nn.Module,
        mean:        tuple,
        std:         tuple,
        n_perturb:   int   = 64,
        n_eval:      int   = 1000,
        strip_alpha: float = 0.5,
        fpr_target:  float = 0.01,
        device:      str   = "cuda",
    ):
        self.model       = model.eval()
        self.mean        = torch.tensor(mean).view(3, 1, 1)
        self.std         = torch.tensor(std).view(3, 1, 1)
        self.n_perturb   = n_perturb
        self.n_eval      = n_eval
        self.strip_alpha = strip_alpha
        self.fpr_target  = fpr_target
        self.device      = device

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.cpu() * self.std + self.mean).clamp(0, 1)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def superimpose(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """공식 공식: clamp(denorm(x1) + alpha·denorm(x2), 0, 1) 후 다시 정규화."""
        result = (self._denorm(x1) + self.strip_alpha * self._denorm(x2)).clamp(0, 1)
        return self._norm(result)

    @torch.no_grad()
    def _entropy_one(self, x: torch.Tensor, blend_pool: torch.Tensor) -> float:
        """입력 1장 x에 대해 blend_pool에서 N장을 뽑아 중첩한 뒤 평균 엔트로피(nats)."""
        idxs     = torch.randint(0, len(blend_pool), (self.n_perturb,))
        selected = blend_pool[idxs]
        x_exp    = x.unsqueeze(0).expand_as(selected)
        x_mix    = self.superimpose(x_exp, selected).to(self.device)

        logits = self.model(x_mix)
        probs  = F.softmax(logits, dim=-1) + 1e-8
        entropy_per_sample = -(probs * probs.log()).sum(dim=-1)   # natural log, nats
        return float(entropy_per_sample.mean().item())

    def _collect_pool(self, loader: DataLoader) -> torch.Tensor:
        imgs_list = []
        for imgs, _ in loader:
            imgs_list.append(imgs)
        return torch.cat(imgs_list, dim=0)

    @torch.no_grad()
    def _predict(self, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        preds = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            preds.append(self.model(batch).argmax(dim=1).cpu())
        return torch.cat(preds)

    def _filter_by_success(
        self,
        clean_imgs: torch.Tensor, clean_lbls: torch.Tensor,
        poison_imgs: torch.Tensor, poison_lbls: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        공식 관행(inspect_correct_prediction_only): 클린은 원래 맞게 분류된 것만,
        포이즌은 실제로 공격이 성공한(타겟으로 분류된) 것만 비교에 포함.
        """
        clean_pred  = self._predict(clean_imgs)
        poison_pred = self._predict(poison_imgs)

        clean_correct  = clean_pred  == clean_lbls
        poison_success = poison_pred == poison_lbls

        return clean_imgs[clean_correct], poison_imgs[poison_success]

    def compute_entropy_distribution(self, images: torch.Tensor, blend_pool: torch.Tensor) -> np.ndarray:
        n = min(len(images), self.n_eval)
        return np.array([self._entropy_one(images[i], blend_pool) for i in range(n)])

    def _set_threshold_by_fpr(self, clean_entropies: np.ndarray) -> float:
        """클린 셋에서 FPR=fpr_target이 되는 엔트로피 분위수 (하위 분위수 = 의심 구간)."""
        return float(np.percentile(clean_entropies, self.fpr_target * 100))

    def evaluate(
        self,
        clean_loader:        DataLoader,
        poison_loader:       DataLoader,
        blend_loader:        DataLoader,
        filter_by_success:   bool = True,
    ) -> dict:
        """
        STRIP 방어 평가.

        1. (옵션) 클린=correctly-classified만, 포이즌=공격 성공한 것만 필터링
        2. 각각 n_eval개의 엔트로피 분포 계산
        3. FPR=fpr_target 기준으로 threshold 자동 설정 (클린 분포 기준)
        4. 포이즌 엔트로피 분포에 threshold 적용 → FNR 계산
        """
        blend_pool = self._collect_pool(blend_loader)

        clean_imgs, clean_lbls   = self._collect_with_labels(clean_loader)
        poison_imgs, poison_lbls = self._collect_with_labels(poison_loader)

        if filter_by_success:
            clean_imgs, poison_imgs = self._filter_by_success(
                clean_imgs, clean_lbls, poison_imgs, poison_lbls
            )

        print(f"[STRIP] Clean 엔트로피 계산 (n={min(len(clean_imgs), self.n_eval)}, n_perturb={self.n_perturb})...")
        clean_H = self.compute_entropy_distribution(clean_imgs, blend_pool)

        print(f"[STRIP] Poison 엔트로피 계산 (n={min(len(poison_imgs), self.n_eval)})...")
        poison_H = self.compute_entropy_distribution(poison_imgs, blend_pool)

        threshold = self._set_threshold_by_fpr(clean_H)

        fpr = float(np.mean(clean_H  < threshold))
        fnr = float(np.mean(poison_H >= threshold))
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
            "n_eval_clean":        len(clean_H),
            "n_eval_poison":       len(poison_H),
            "n_perturb":           self.n_perturb,
            "clean_entropies":     clean_H.tolist(),
            "poison_entropies":    poison_H.tolist(),
        }

    def _collect_with_labels(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs_list, lbls_list = [], []
        for imgs, lbls in loader:
            imgs_list.append(imgs)
            lbls_list.append(lbls)
        return torch.cat(imgs_list, dim=0), torch.cat(lbls_list, dim=0)
