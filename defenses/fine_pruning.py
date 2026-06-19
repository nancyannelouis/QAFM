"""
Fine-Pruning (Liu, Dolan-Gavitt, Garg, RAID 2018) 방어 기법 구현.

공식 구현(THUYimingLi/BackdoorBox, core/defenses/Pruning.py + FineTuning.py)
기준으로 포팅함.

핵심 메커니즘:
  1. (Pruning) 클린 데이터에 거의 반응하지 않는("dormant") 채널을 가지치기
     - 지정한 레이어의 출력을 클린 데이터로 forward, 채널별 평균 활성값 계산
       (공식 구현과 동일: torch.mean(container, dim=[0,2,3]))
     - 활성값이 가장 낮은 prune_rate 비율의 채널을 0으로 마스킹
       (공식 구현과 동일한 MaskedLayer로 감쌈)
  2. (FineTuning) 가지치기로 손상된 벤치 정확도를 클린 데이터로 파인튜닝해 복구
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MaskedLayer(nn.Module):
    """공식 구현과 동일: base 레이어 출력에 채널 마스크를 곱함."""

    def __init__(self, base: nn.Module, mask: torch.Tensor):
        super().__init__()
        self.base = base
        self.mask = mask

    def forward(self, x):
        return self.base(x) * self.mask


class FinePruning:
    """
    Args:
        model:       백도어 모델 (가지치기/파인튜닝으로 in-place 수정됨)
        layer_name:  가지치기할 레이어 이름 (공식 기본값 'layer2')
        prune_rate:  가지치기 비율 (공식 기본값 0.2)
        device:      cuda/cpu
    """

    def __init__(self, model: nn.Module, layer_name: str = "layer2",
                 prune_rate: float = 0.2, device: str = "cuda"):
        self.model      = model
        self.layer_name = layer_name
        self.prune_rate = prune_rate
        self.device     = device

    @torch.no_grad()
    def prune(self, clean_loader: DataLoader, activation_fraction: float = 0.2) -> dict:
        """
        공식 알고리즘: clean_loader 중 activation_fraction 비율만 forward해
        layer_name의 채널별 평균 활성값을 구하고, 가장 낮은 prune_rate만큼을
        가지치기(0으로 마스킹)함.
        """
        self.model.eval()
        container = []

        def forward_hook(module, inp, out):
            container.append(out.detach())

        target_module = getattr(self.model, self.layer_name)
        hook = target_module.register_forward_hook(forward_hook)

        n_target = int(len(clean_loader.dataset) * activation_fraction)
        n_seen = 0
        for imgs, _ in clean_loader:
            self.model(imgs.to(self.device))
            n_seen += imgs.size(0)
            if n_seen >= n_target:
                break
        hook.remove()

        container  = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=[0, 2, 3])   # 채널별 평균 활성값 (공식과 동일)
        seq_sort   = torch.argsort(activation)
        num_channels = len(activation)
        n_pruned     = int(num_channels * self.prune_rate)

        mask = torch.ones(num_channels, device=self.device)
        mask[seq_sort[:n_pruned]] = 0
        mask = mask.reshape(1, -1, 1, 1)

        setattr(self.model, self.layer_name, MaskedLayer(target_module, mask))
        print(f"  [FinePruning] '{self.layer_name}' 채널 {num_channels}개 중 {n_pruned}개 가지치기 "
              f"(활성값 하위 {self.prune_rate*100:.0f}%, 활성 측정 샘플 {n_seen}장)")
        return {"n_channels": num_channels, "n_pruned": n_pruned}

    def fine_tune(self, clean_loader: DataLoader, epochs: int = 10, lr: float = 0.001,
                  momentum: float = 0.9, weight_decay: float = 5e-4) -> None:
        """공식 알고리즘: 가지치기 후 클린 데이터로 전체 모델 파인튜닝."""
        self.model.to(self.device).train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # 공식 구현의 LR 감소 스케줄: epoch 20에 0.1배 (epochs<=10 기본 설정에서는 발동 안 함)
            if epoch == 20:
                lr *= 0.1
                for g in optimizer.param_groups:
                    g["lr"] = lr
            for imgs, labels in clean_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(imgs), labels)
                loss.backward()
                optimizer.step()

        self.model.eval()
