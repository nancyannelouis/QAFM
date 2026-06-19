"""
ShrinkPad (Backdoor Attack in the Physical World, ICLR Workshop 2021) 방어 기법 구현.

공식 구현(THUYimingLi/BackdoorBox, core/defenses/ShrinkPad.py) 기준으로 포팅함.

핵심 메커니즘 (전처리 기반, 재학습 불필요):
  입력 이미지를 (size_map - pad) 크기로 줄인 뒤, size_map 크기로 무작위 위치에
  되돌려 패딩함 — 트리거의 정확한 픽셀 위치 의존성을 깨뜨리는 것이 목표.
  공식 구현과 동일하게 가능한 모든 (left,top,right,bottom) 패딩 배치 중
  하나를 무작위로 선택(RandomChoice).
"""

import random

import torch
import torchvision.transforms as T


def _all_pad_choices(pad_w: int, pad_h: int) -> list:
    """공식 RandomPad()와 동일: 가능한 모든 패딩 배치를 나열."""
    choices = []
    for i in range(pad_w + 1):
        for j in range(pad_h + 1):
            choices.append((i, j, pad_w - i, pad_h - j))
    return choices


class ShrinkPad:
    """
    Args:
        size_map: 원본 이미지 크기 (공식 기본값 32)
        pad:      패딩 크기 (공식 기본값 4)
    """

    def __init__(self, size_map: int = 32, pad: int = 4):
        self.size_map = size_map
        self.pad = pad
        self._pad_choices = _all_pad_choices(pad, pad)
        self._shrink = T.Resize((size_map - pad, size_map - pad))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """(C,H,W) 텐서 1장에 ShrinkPad 적용."""
        l, t, r, b = random.choice(self._pad_choices)
        pad_fn = T.Pad(padding=(l, t, r, b))
        return pad_fn(self._shrink(img))

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """(N,C,H,W) 배치에 샘플별로 독립적인 무작위 패딩을 적용."""
        return torch.stack([self(img) for img in batch])
