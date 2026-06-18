"""
Neural Cleanse (Wang et al., IEEE S&P 2019) 방어 기법 구현.

핵심 메커니즘:
  - 각 클래스 y에 대해 "다른 클래스 → y로 분류되게 하는 최소 패턴" 역공학으로 최적화
  - 최소 L1 노름 패턴을 가진 클래스를 백도어 타겟으로 판정
  - Anomaly Index (MAD 기반, ×1.4826 정규분포 보정상수) < 2.0이면 정상 → 탐지 실패

QAFM 저항성:
  - QAFM 트리거는 전체 DCT 블록에 분산 → 공간 도메인 패치 가정 위반
  - 모든 클래스의 L1 노름이 유사하게 나타나 Anomaly Index < 2.0 달성 예상

구현 충실성 (공식 저장소 bolunwang/backdoor 알고리즘 기준):
  - 고정 λ 1회 최적화가 아니라, ASR(공격성공률)이 목표치(99%)에 도달했는지에
    따라 cost를 적응적으로 증가/감소시키는 탐색을 사용함. 고정 λ를 쓰면 클래스마다
    결정 경계의 기하가 달라 "얼마나 쉽게 그 클래스로 끌려가는지"가 다르므로, L1
    노름 차이가 진짜 백도어 신호가 아니라 최적화 난이도 차이(노이즈)에 더 가까워짐.
    적응적 cost 탐색은 모든 클래스를 "동일한 ASR 기준에서 최소화된 패턴 크기"로
    맞춰서 비교 가능하게 만드는 것이 원 논문의 핵심 아이디어임.
  - 마스크 L1 정규화 항도 sum(|m|) 기준으로 통일 (이전 코드는 loss는 mean(|m|)으로
    페널티를 주고 정작 비교용 L1 노름은 sum(|m|)으로 계산하는 불일치가 있었음 —
    32×32 이미지에서 mean은 sum의 1/1024이므로 사실상 패널티가 거의 안 걸려
    모든 클래스의 마스크가 이미지 전체로 퍼지기 쉬웠고, 그 결과 클래스 간 L1
    차이가 의미 있는 신호가 아니라 최적화 노이즈가 됨).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class NeuralCleanse:
    """
    Args:
        model:        백도어 삽입 의심 모델
        num_classes:  클래스 수
        img_size:     이미지 크기 (32 for CIFAR/GTSRB)
        device:       CUDA device
        lr:           역공학 최적화 학습률
        steps:        역공학 최적화 최대 스텝 수
        init_cost:    L1 정규화 계수(cost)의 초기값 — 아래 적응 탐색으로 조절됨
        attack_succ_threshold: 이 ASR(공격성공률) 이상을 "트리거 성공"으로 간주
        patience:     cost 증가/감소를 결정하기 전 연속 관찰 스텝 수
        cost_multiplier: cost 증가 배율 (감소는 그 1.5배 지수로 더 빠르게, 원 논문 관행)
        n_samples:    역공학용 샘플 풀 크기. 기본값 max(500, num_classes*10)
    """

    def __init__(
        self,
        model:       nn.Module,
        num_classes: int,
        img_size:    int   = 32,
        device:      str   = "cuda",
        lr:          float = 0.1,
        steps:       int   = 1000,
        init_cost:   float = 1e-3,
        attack_succ_threshold: float = 0.99,
        patience:    int   = 5,
        cost_multiplier: float = 1.5,
        n_samples:   int   = None,
    ):
        self.model       = model.eval()
        self.num_classes = num_classes
        self.img_size    = img_size
        self.device      = device
        self.lr          = lr
        self.steps       = steps
        self.init_cost   = init_cost
        self.attack_succ_threshold = attack_succ_threshold
        self.patience    = patience
        self.cost_multiplier_up   = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        # 클래스 수가 많아질수록(예: CIFAR-100의 100클래스) 고정된 작은 배치 하나로는
        # 클래스당 대표 샘플이 거의 없어져(예: 128장 중 다수 클래스가 0~1장) 역공학
        # 결과가 샘플링 노이즈에 흔들리기 쉬움. 클래스 수에 비례해 풀 크기를 키워서
        # 클래스당 최소 ~10장 수준을 확보함.
        self.n_samples   = n_samples or max(500, num_classes * 10)

    def _collect_pool(self, loader: DataLoader) -> torch.Tensor:
        """역공학에 쓸 이미지 풀을 여러 배치에 걸쳐 self.n_samples까지 수집."""
        imgs_list = []
        n_collected = 0
        for imgs, _ in loader:
            imgs_list.append(imgs)
            n_collected += imgs.size(0)
            if n_collected >= self.n_samples:
                break
        return torch.cat(imgs_list, dim=0)[:self.n_samples]

    def _reverse_trigger(self, batch_imgs: torch.Tensor, target_class: int) -> tuple:
        """
        특정 target_class로 유도하는 최소 트리거 (mask, pattern) 역공학.

        최적화 목적함수:
          min_{m, p} CE(f(x*(1-m) + p*m), y_t) + cost·||m||_1
        cost는 ASR>=attack_succ_threshold 달성 여부에 따라 patience 스텝마다
        증가(마스크를 더 작게 압박)/감소(공격 성공을 우선)되는 적응 탐색을 따름
        (원 논문 Algorithm의 핵심 — 고정 λ 대신 클래스 간 비교 가능한 기준을 만듦).

        Returns:
            (mask, pattern, l1_norm) — ASR 목표를 만족하면서 L1이 가장 작았던 해
        """
        H = W = self.img_size
        mask    = torch.zeros(1, H, W, requires_grad=True,  device=self.device)
        pattern = torch.rand(3, H, W,  requires_grad=True,  device=self.device)

        optimizer = optim.Adam([mask, pattern], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        target_t  = torch.tensor([target_class], device=self.device)

        batch_imgs = batch_imgs.to(self.device)
        N = batch_imgs.size(0)
        target_full = target_t.expand(N)

        cost = self.init_cost
        up_counter, down_counter = 0, 0
        best_l1 = float("inf")
        best_mask, best_pattern = None, None

        for _ in range(self.steps):
            optimizer.zero_grad()
            m = torch.sigmoid(mask)
            p = torch.sigmoid(pattern)
            m_exp = m.unsqueeze(0).expand(N, -1, -1, -1)
            x_adv = batch_imgs * (1 - m_exp) + p.unsqueeze(0) * m_exp
            logits = self.model(x_adv)
            l1 = m.abs().sum()
            loss = criterion(logits, target_full) + cost * l1
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                asr     = (logits.argmax(1) == target_full).float().mean().item()
                cur_l1  = float(l1.item())

            # ASR 목표를 만족하는 후보 중 가장 작은 마스크를 최종 해로 채택
            if asr >= self.attack_succ_threshold and cur_l1 < best_l1:
                best_l1      = cur_l1
                best_mask    = m.detach().clone()
                best_pattern = p.detach().clone()

            # 적응적 cost 조절 (원 논문 알고리즘의 핵심)
            if asr >= self.attack_succ_threshold:
                up_counter += 1
                down_counter = 0
            else:
                down_counter += 1
                up_counter = 0

            if up_counter >= self.patience:
                cost *= self.cost_multiplier_up
                up_counter = 0
            elif down_counter >= self.patience:
                cost /= self.cost_multiplier_down
                down_counter = 0

        # ASR 목표를 한 번도 못 만족했으면 마지막 상태를 그대로 사용
        if best_mask is None:
            with torch.no_grad():
                best_mask    = torch.sigmoid(mask)
                best_pattern = torch.sigmoid(pattern)
            best_l1 = float(best_mask.abs().sum().item())

        return (best_mask.squeeze().cpu().numpy(),
                best_pattern.cpu().numpy(),
                best_l1)

    def run(self, loader: DataLoader) -> dict:
        """
        전체 클래스에 대해 역공학 수행 후 Anomaly Index 계산.

        Returns:
            {
              "l1_norms":         [float×num_classes],
              "anomaly_index":    [float×num_classes],
              "suspected_target": int,
              "max_ai":           float,
              "bypass":           bool   (True if 의심 클래스의 AI < 2.0)
            }
        """
        sample_pool = self._collect_pool(loader)
        print(f"[Neural Cleanse] 역공학 샘플 풀: {sample_pool.size(0)}장 "
              f"(클래스 {self.num_classes}개, 평균 {sample_pool.size(0)/self.num_classes:.1f}장/클래스)")

        l1_norms = []
        print("[Neural Cleanse] Reversing triggers per class (adaptive cost search)...")
        for cls in range(self.num_classes):
            _, _, l1 = self._reverse_trigger(sample_pool, cls)
            l1_norms.append(l1)
            print(f"  Class {cls:3d}: L1={l1:.4f}")

        # Anomaly Index (MAD 기반). 1.4826은 정규분포 가정 하의 일관성 보정 상수로
        # 원 논문 표기와 동일하게 사용.
        arr = np.array(l1_norms)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) * 1.4826
        anomaly_idx = np.abs(arr - med) / (mad + 1e-8)

        # 백도어는 "비정상적으로 작은" 패턴으로 universal trigger를 만들 수 있어야
        # 성립하므로, 의심 클래스는 항상 L1이 가장 작은 클래스로 판정 (원 논문과 동일,
        # 양방향 MAD가 아니라 작은 쪽 이상치만 의미 있음).
        suspected = int(np.argmin(arr))
        max_ai    = float(anomaly_idx[suspected])
        bypass    = bool(max_ai < 2.0)

        return {
            "l1_norms":         l1_norms,
            "anomaly_index":    anomaly_idx.tolist(),
            "suspected_target": suspected,
            "max_ai":           max_ai,
            "bypass":           bypass,
        }
