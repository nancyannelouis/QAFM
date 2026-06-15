"""
Neural Cleanse (Wang et al., IEEE S&P 2019) 방어 기법 구현.

핵심 메커니즘:
  - 각 클래스 y에 대해 "다른 클래스 → y로 분류되게 하는 최소 패턴" 역공학으로 최적화
  - 최소 L1 노름 패턴을 가진 클래스를 백도어 타겟으로 판정
  - Anomaly Index (MAD 기반) < 2.0이면 정상 → 탐지 실패

QAFM 저항성:
  - QAFM 트리거는 전체 DCT 블록에 분산 → 공간 도메인 패치 가정 위반
  - 모든 클래스의 L1 노름이 유사하게 나타나 Anomaly Index < 2.0 달성 예상
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Dict


class NeuralCleanse:
    """
    Args:
        model:        백도어 삽입 의심 모델
        num_classes:  클래스 수
        img_size:     이미지 크기 (32 for CIFAR/GTSRB)
        device:       CUDA device
        lr:           최적화 학습률
        steps:        역공학 최적화 스텝 수
        lam:          L1 정규화 계수 (트리거 최소화 항)
        init_cost:    초기 cost (adaptive cost 조절용)
    """

    def __init__(
        self,
        model:       nn.Module,
        num_classes: int,
        img_size:    int   = 32,
        device:      str   = "cuda",
        lr:          float = 0.01,
        steps:       int   = 1000,
        lam:         float = 1e-3,
        init_cost:   float = 1e-3,
    ):
        self.model       = model.eval()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.device      = device
        self.lr          = lr
        self.steps       = steps
        self.lam         = lam
        self.init_cost   = init_cost

    def _reverse_trigger(self, loader: DataLoader, target_class: int) -> tuple:
        """
        특정 target_class로 유도하는 최소 트리거 (mask, pattern) 역공학.

        최적화 목적함수:
          min_{m, p} CE(f(x*(1-m) + p*m), y_t) + λ·||m||_1

        Returns:
            (mask, pattern) as float tensors (1, H, W), (3, H, W) in [0,1]
            l1_norm: scalar float
        """
        H = W = self.img_size
        mask    = torch.zeros(1, H, W, requires_grad=True,  device=self.device)
        pattern = torch.rand(3, H, W,  requires_grad=True,  device=self.device)

        optimizer = optim.Adam([mask, pattern], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        target_t  = torch.tensor([target_class], device=self.device)

        # 배치 수집 (최대 512샘플)
        batch_imgs, _ = next(iter(loader))
        batch_imgs = batch_imgs[:512].to(self.device)
        N = batch_imgs.size(0)
        target_full = target_t.expand(N)

        for _ in range(self.steps):
            optimizer.zero_grad()
            m = torch.sigmoid(mask)
            p = torch.sigmoid(pattern)
            # 트리거 적용: x' = x·(1-m) + p·m
            m_exp = m.unsqueeze(0).expand(N, -1, -1, -1)
            x_adv = batch_imgs * (1 - m_exp) + p.unsqueeze(0) * m_exp
            logits = self.model(x_adv)
            loss = criterion(logits, target_full) + self.lam * m.abs().mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            m_final = torch.sigmoid(mask).squeeze()
            p_final = torch.sigmoid(pattern)
        l1_norm = float(m_final.abs().sum().item())
        return m_final.cpu().numpy(), p_final.cpu().numpy(), l1_norm

    def run(self, loader: DataLoader) -> dict:
        """
        전체 클래스에 대해 역공학 수행 후 Anomaly Index 계산.

        Returns:
            {
              "l1_norms": [float×num_classes],
              "anomaly_index": [float×num_classes],
              "suspected_target": int or None,
              "max_ai": float,
              "bypass": bool   (True if all AI < 2.0)
            }
        """
        l1_norms = []
        print("[Neural Cleanse] Reversing triggers per class...")
        for cls in range(self.num_classes):
            _, _, l1 = self._reverse_trigger(loader, cls)
            l1_norms.append(l1)
            print(f"  Class {cls:3d}: L1={l1:.4f}")

        # Anomaly Index (Median Absolute Deviation)
        arr  = np.array(l1_norms)
        med  = np.median(arr)
        mad  = np.median(np.abs(arr - med))
        anomaly_idx = np.abs(arr - med) / (mad + 1e-8)

        suspected = int(np.argmin(arr))   # 가장 작은 L1 = 의심 클래스
        max_ai    = float(anomaly_idx.max())
        bypass    = bool(anomaly_idx[suspected] < 2.0)

        return {
            "l1_norms":       l1_norms,
            "anomaly_index":  anomaly_idx.tolist(),
            "suspected_target": suspected,
            "max_ai":         max_ai,
            "bypass":         bypass,
        }
