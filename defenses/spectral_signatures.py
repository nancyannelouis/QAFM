"""
Spectral Signatures (Tran et al., NeurIPS 2018) 방어 기법 구현.

공식 구현(THUYimingLi/BackdoorBox, core/defenses/Spectral.py — 원저자
MadryLab/backdoor_data_poisoning 포팅)을 기준으로 포팅함.

핵심 메커니즘 (공식 알고리즘):
  1. "타겟 클래스로 레이블된" 학습 샘플만 모음 (진짜 타겟 클래스 샘플 + 포이즌되어
     타겟으로 relabel된 샘플이 섞여 있음 — 방어자는 어느 게 어느 건지 모름)
  2. 이 샘플들의 중간 특징(layer4, pooling 이전)을 추출
  3. 평균을 빼고 SVD 분해 → 최대 특이벡터에 대한 투영 점수 계산
  4. 상위 percentile(점수가 큰 쪽)을 포이즌 의심으로 제거

이전 구현의 버그: "클린 vs 포이즌"을 비교할 때 클린 쪽에 9개 비-타겟 클래스를
전부 섞어서 SVD를 돌렸음 — 그러면 최대 특이벡터가 "포이즌이냐 아니냐"가 아니라
"원래 어느 클래스냐"라는 훨씬 큰 자연 변동을 잡아버려 신호가 묻힘. 공식 알고리즘은
반드시 "타겟 클래스로 레이블된 샘플들 안에서만" 비교해야 함.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Optional


class SpectralSignatures:
    """
    Args:
        model:       분석할 모델 (poisoned_trainset으로 학습된 모델)
        layer_name:  특징 추출 레이어 이름 (공식 구현 기준 'layer4', pooling 이전)
        device:      cuda/cpu
        margin:      포이즌 비율 추정치에 곱하는 안전 마진 (공식 관행 1.5배)
    """

    def __init__(
        self,
        model:      nn.Module,
        layer_name: str   = "layer4",
        device:     str   = "cuda",
        margin:     float = 1.5,
    ):
        self.model      = model.eval()
        self.device     = device
        self.layer_name = layer_name
        self.margin     = margin

        self._features = []
        target_module = dict(self.model.named_modules()).get(layer_name)
        if target_module is None:
            raise ValueError(f"모델에 '{layer_name}' 레이어가 없음")
        self._hook = target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        self._features.append(out.detach().cpu().view(out.size(0), -1))

    def remove_hook(self):
        self._hook.remove()

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor, batch_size: int = 256) -> np.ndarray:
        """이미지 텐서 (N,C,H,W) → layer 특징 (N,D)."""
        self._features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)
            self.model(batch)
        return torch.cat(self._features, dim=0).numpy()

    def detect(
        self,
        poisoned_dataset,
        target_label: int,
        poison_idx:   np.ndarray,
    ) -> dict:
        """
        실제 포이즌된 학습 셋(poisoned_dataset)에서 target_label로 레이블된
        샘플만 추려 SVD 기반 이상치 탐지 (공식 알고리즘 그대로).

        Args:
            poisoned_dataset: (image_tensor, label) 반환하는 Dataset —
                              PoisonedImageDataset처럼 일부가 target_label로
                              relabel된 실제 학습 셋이어야 함
            target_label:     의심 타겟 클래스
            poison_idx:       poisoned_dataset 안에서 실제 포이즌된 샘플의
                              인덱스 (정답값, 탐지율 계산에만 사용 — 탐지
                              알고리즘 자체는 이 정보를 모른다고 가정)

        Returns:
            {n_target, n_poison_in_target, poison_detection_rate,
             clean_fp_rate, percentile, bypass}
        """
        # PoisonedImageDataset은 .labels를 numpy 배열로 직접 노출함 — 5만 장
        # 전체를 __getitem__(PIL 변환 포함)으로 훑지 않고 빠르게 레이블만 확인
        labels = np.asarray(poisoned_dataset.labels)
        target_global_idx = np.where(labels == target_label)[0]

        images = torch.stack([poisoned_dataset[i][0] for i in target_global_idx])
        features = self._extract_features(images)

        # target_global_idx 내에서 진짜 포이즌인 위치(로컬 인덱스)
        poison_set = set(poison_idx.tolist())
        is_poison_local = np.array([gi in poison_set for gi in target_global_idx])
        n_target  = len(target_global_idx)
        n_poison  = int(is_poison_local.sum())

        if n_target < 2 or n_poison == 0:
            return {
                "n_target": n_target, "n_poison_in_target": n_poison,
                "poison_detection_rate": 0.0, "clean_fp_rate": 0.0,
                "percentile": None, "bypass": True,
            }

        # ─── SVD 기반 이상치 점수 (공식 알고리즘) ────────────────────────────
        mean_feat = features.mean(axis=0, keepdims=True)
        centered  = features - mean_feat
        u, s, v   = np.linalg.svd(centered, full_matrices=False)
        top_v     = v[0:1]                                   # 최대 우특이벡터 (1, D)
        scores    = np.linalg.norm(centered @ top_v.T, axis=1)  # (N,)

        # 포이즌 비율을 안다고 가정(공식 관행): 추정 비율 × 안전마진만큼 제거
        true_poison_frac = n_poison / n_target
        removal_frac     = min(true_poison_frac * self.margin, 0.9)
        percentile       = 100 * (1 - removal_frac)
        threshold         = np.percentile(scores, percentile)
        flagged           = scores >= threshold

        poison_detection_rate = float(flagged[is_poison_local].mean())
        clean_fp_rate         = float(flagged[~is_poison_local].mean()) if (~is_poison_local).any() else 0.0

        # 포이즌 탐지율이 클린 오탐율보다 뚜렷이 높지 않으면 사실상 회피로 판정
        bypass = poison_detection_rate < 0.5

        return {
            "n_target":               n_target,
            "n_poison_in_target":     n_poison,
            "poison_detection_rate":  round(poison_detection_rate, 4),
            "clean_fp_rate":          round(clean_fp_rate, 4),
            "percentile":             round(percentile, 2),
            "bypass":                 bypass,
        }
