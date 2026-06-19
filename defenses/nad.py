"""
NAD: Neural Attention Distillation (Li et al., ICLR 2021) 방어 기법 구현.

공식 구현(THUYimingLi/BackdoorBox, core/defenses/NAD.py — 원저자 공식 코드
https://github.com/bboylyg/NAD 기반 포팅) 기준으로 포팅함.

핵심 메커니즘:
  1. 백도어 모델을 클린 데이터 일부로 살짝 파인튜닝해 "teacher" 모델을 만듦
     (이 teacher는 약하게라도 백도어가 옅어진 상태)
  2. teacher를 고정(freeze)하고, 원본(student) 모델을 다음 손실로 재학습:
       loss = CE(student(x), y) + Σ_l beta_l · AT(student_l, teacher_l)
     AT(attention transfer)는 지정 레이어들의 attention map 간 MSE.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class AttentionTransferLoss(nn.Module):
    """공식 구현과 동일한 attention map 정의(채널 축 L_p norm 합 후 정규화) 및 MSE 손실."""

    def __init__(self, power: float = 2.0):
        super().__init__()
        self.power = power

    def attention_map(self, fm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        am = torch.pow(torch.abs(fm), self.power)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        return am / (norm + eps)

    def forward(self, fm_s: torch.Tensor, fm_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))


class NAD:
    """
    Args:
        model:         백도어 모델 (in-place로 수정됨)
        target_layers: attention loss를 적용할 레이어 이름 목록 (공식 기본 ['layer2','layer3','layer4'])
        beta:          각 target_layer의 attention loss 가중치 (공식 기본 [500,500,500])
        power:         attention map 계산 시 거듭제곱 (공식 기본 2.0)
        device:        cuda/cpu
    """

    def __init__(self, model: nn.Module, target_layers=("layer2", "layer3", "layer4"),
                 beta=(500, 500, 500), power: float = 2.0, device: str = "cuda"):
        assert len(beta) == len(target_layers), "beta와 target_layers 길이가 같아야 함"
        self.model         = model
        self.target_layers = list(target_layers)
        self.beta           = list(beta)
        self.power           = power
        self.device           = device

    def repair(self, clean_loader: DataLoader,
               tune_epochs: int = 10, tune_lr: float = 0.01,
               epochs: int = 20, lr: float = 0.01,
               momentum: float = 0.9, weight_decay: float = 5e-4,
               lr_milestones=(2, 4, 6, 8), gamma: float = 0.1) -> None:
        """공식 알고리즘 그대로: teacher 파인튜닝 → student를 attention distillation으로 재학습."""
        criterion = nn.CrossEntropyLoss()

        # ── 1) teacher: 클린 데이터로 살짝 파인튜닝 ──────────────────────────
        print("  [NAD] teacher 모델 파인튜닝...")
        teacher = deepcopy(self.model).to(self.device)
        teacher.train()
        t_optim = optim.SGD(teacher.parameters(), lr=tune_lr, momentum=momentum, weight_decay=weight_decay)
        for _ in range(tune_epochs):
            for imgs, labels in clean_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                t_optim.zero_grad()
                loss = criterion(teacher(imgs), labels)
                loss.backward()
                t_optim.step()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # ── 2) student: CE + Attention Transfer loss로 재학습 ────────────────
        print("  [NAD] attention distillation으로 student 재학습...")
        self.model.to(self.device).train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        at_loss = AttentionTransferLoss(self.power)

        for epoch in range(epochs):
            if epoch in lr_milestones:
                lr *= gamma
                for g in optimizer.param_groups:
                    g["lr"] = lr

            for imgs, labels in clean_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                s_feats, t_feats = [], []
                hooks = []
                for name, module in self.model._modules.items():
                    if name in self.target_layers:
                        hooks.append(module.register_forward_hook(
                            lambda m, i, o: s_feats.append(o)))
                for name, module in teacher._modules.items():
                    if name in self.target_layers:
                        hooks.append(module.register_forward_hook(
                            lambda m, i, o: t_feats.append(o)))

                output = self.model(imgs)
                _ = teacher(imgs)
                for h in hooks:
                    h.remove()

                loss = criterion(output, labels)
                for i in range(len(self.beta)):
                    loss = loss + at_loss(s_feats[i], t_feats[i]) * self.beta[i]

                loss.backward()
                optimizer.step()

        self.model.eval()
