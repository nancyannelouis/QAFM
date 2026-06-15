"""
Spectral Signatures (Tran et al., NeurIPS 2018) 방어 기법 구현.

핵심 메커니즘:
  - 모델 중간 특징 추출 후 SVD 분해
  - 포이즌 샘플은 정상 샘플과 다른 스펙트럼 서명 보유
  - 최대 특이벡터에 대한 투영값(correlation) 분포로 탐지
  - 상위 epsilon 비율을 포이즌으로 제거

scikit-learn 수준으로 구현 가능 (공개 코드 기반).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional


class SpectralSignatures:
    """
    Args:
        model:     분석할 모델
        layer:     특징 추출 레이어 이름 (default: 마지막 fc 이전 avgpool)
        device:    cuda/cpu
        epsilon:   포이즌으로 의심 상위 비율 (논문 기본 0.05)
        n_svd:     SVD 주성분 수
    """

    def __init__(
        self,
        model:   nn.Module,
        layer:   Optional[str] = None,
        device:  str   = "cuda",
        epsilon: float = 0.05,
        n_svd:   int   = 1,
    ):
        self.model   = model.eval()
        self.device  = device
        self.epsilon = epsilon
        self.n_svd   = n_svd

        # hook으로 중간 특징 캡처
        self._features  = []
        self._hook      = None
        self._layer_name = layer or "avgpool"
        self._register_hook()

    def _register_hook(self):
        """모델의 avgpool (또는 지정 레이어) 이후 특징 추출 hook 등록."""
        def _hook_fn(module, input, output):
            self._features.append(
                output.detach().cpu().view(output.size(0), -1)
            )

        # ResNet18 구조: self.model.layer4 → avgpool → linear
        target_module = None
        for name, module in self.model.named_modules():
            if "avgpool" in name or name == self._layer_name:
                target_module = module
                break
        if target_module is None:
            # fallback: 마지막에서 두 번째 모듈
            modules = list(self.model.named_modules())
            target_module = modules[-2][1]

        self._hook = target_module.register_forward_hook(_hook_fn)

    def _extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        모든 샘플의 중간 특징 벡터와 레이블 추출.

        Returns:
            features: (N, D) float32
            labels:   (N,) int
        """
        self._features = []
        all_labels = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(self.device)
                self.model(imgs)
                all_labels.extend(lbls.numpy().tolist())

        features = torch.cat(self._features, dim=0).numpy()
        labels   = np.array(all_labels)
        return features, labels

    def remove_hook(self):
        if self._hook:
            self._hook.remove()

    def compute_svd_scores(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        SVD 분해 후 최대 특이벡터에 대한 투영값 계산.

        각 샘플 i의 스펙트럼 서명 점수:
            score_i = |u_1^T (f_i - f_mean)|^2

        Returns:
            scores: (N,) float array
        """
        mean_feat = features.mean(axis=0)
        centered  = features - mean_feat

        svd = TruncatedSVD(n_components=self.n_svd)
        svd.fit(centered)
        top_v = svd.components_[0]   # (D,) 최대 우 특이벡터

        scores = np.abs(centered @ top_v) ** 2
        return scores

    def detect(
        self,
        mix_loader:    DataLoader,
        target_class:  int,
    ) -> dict:
        """
        혼합 (클린 + 포이즌) 로더에서 포이즌 샘플 탐지.

        타겟 클래스로 레이블된 샘플만 분석 (all-to-one 공격 가정).

        Returns:
            {
              "n_total":          전체 샘플 수
              "n_flagged":        포이즌으로 탐지된 수
              "flag_rate":        탐지 비율
              "scores":           SVD 점수 배열
              "threshold":        탐지 임계값
              "bypass":           실제 포이즌 탐지에 실패하면 True
            }
        """
        features, labels = self._extract_features(mix_loader)

        # 타겟 클래스 샘플만 필터
        target_mask = labels == target_class
        target_feat = features[target_mask]

        if len(target_feat) < 2:
            return {"bypass": True, "n_total": 0, "n_flagged": 0}

        scores    = self.compute_svd_scores(target_feat)
        n_total   = len(scores)
        n_flagged = int(np.ceil(self.epsilon * n_total))
        threshold = np.sort(scores)[::-1][n_flagged - 1]

        flagged_mask = scores >= threshold

        return {
            "n_total":   n_total,
            "n_flagged": int(flagged_mask.sum()),
            "flag_rate": float(flagged_mask.mean()),
            "threshold": float(threshold),
            "scores":    scores.tolist(),
            "bypass":    False,   # 탐지 성공 여부는 외부에서 실제 레이블로 교차검증
        }

    def evaluate(
        self,
        clean_loader:  DataLoader,
        poison_loader: DataLoader,
        target_class:  int,
    ) -> dict:
        """
        클린/포이즌 분리 로더를 받아 탐지 정확도 측정.

        포이즌 로더의 샘플을 "타겟 클래스" 레이블로 혼합 후
        SVD 서명이 분리되는지 확인.
        """
        # 클린 특징
        clean_feat, _ = self._extract_features(clean_loader)
        # 포이즌 특징
        poison_feat, _ = self._extract_features(poison_loader)

        # 혼합 후 SVD 분석
        all_feat = np.concatenate([clean_feat, poison_feat], axis=0)
        n_clean  = len(clean_feat)
        n_poison = len(poison_feat)
        true_labels = np.array([0] * n_clean + [1] * n_poison)   # 0=clean, 1=poison

        scores = self.compute_svd_scores(all_feat)
        threshold_idx = int(np.ceil(self.epsilon * len(scores)))
        threshold_val = np.sort(scores)[::-1][max(threshold_idx - 1, 0)]
        flagged = scores >= threshold_val

        # 포이즌이 상위 epsilon 구간에 집중되면 탐지 성공
        poison_detection_rate = float(flagged[n_clean:].mean())
        clean_fp_rate         = float(flagged[:n_clean].mean())

        # QAFM: 트리거가 분산되어 있어 스펙트럼 서명이 분리되지 않음 → bypass
        bypass = poison_detection_rate < 0.5

        return {
            "n_clean":                 n_clean,
            "n_poison":                n_poison,
            "poison_detection_rate":   round(poison_detection_rate, 4),
            "clean_fp_rate":           round(clean_fp_rate, 4),
            "epsilon":                 self.epsilon,
            "bypass":                  bypass,
            "scores_clean_mean":       float(scores[:n_clean].mean()),
            "scores_poison_mean":      float(scores[n_clean:].mean()),
        }
