"""
FTrojan Baseline (Wang et al., ECVA 2022).

핵심 메커니즘:
  - DCT 주파수 도메인에 고정 델타 δ 추가 (양자화 테이블과 미정렬)
  - C' = C + δ  (δ ≠ k·Q_{ij} → r = δ/Q가 비정수)
  - Q=50 환경에서 r < 1이 되면 트리거 소멸 → ASR 15% 수준으로 급락

QAFM과의 직접 비교:
  - 동일 DCT 위치 (0,1) 사용
  - 차이는 오직 Δ 설계: 고정 δ vs k·Q_{ij}
"""

import numpy as np
import torch
from utils.jpeg_utils import (
    get_quantization_table, rgb_to_ycbcr, ycbcr_to_rgb,
    jpeg_compress, dct2, idct2, compute_r, r_range_classification,
)


class FTrojan:
    """
    Args:
        trigger_pos: DCT 블록 내 삽입 위치 (QAFM과 동일하게 (0,1) 사용)
        delta:       고정 트리거 변조량 (양자화 테이블과 무관)
        target_label: 공격 목표 클래스
        poison_rate:  포이즌 비율
        q_train:     학습 JPEG Q (FTrojan은 q_train=75 적용 후 압축)
    """

    def __init__(
        self,
        trigger_pos:  tuple = (0, 1),
        delta:        float = 20.0,
        target_label: int   = 0,
        poison_rate:  float = 0.05,
        q_train:      int   = 75,
    ):
        self.trigger_pos  = trigger_pos
        self.delta        = delta
        self.target_label = target_label
        self.poison_rate  = poison_rate
        self.q_train      = q_train

        # r = δ / Q_{ij} 분석 (FTrojan 취약성 재현)
        Q_table = get_quantization_table(q_train, "luma")
        i, j = trigger_pos
        Q_ij = Q_table[i, j]
        r = compute_r(delta, Q_ij)
        cls_info = r_range_classification(r)
        print(f"[FTrojan] δ={delta}, Q[{i},{j}]={Q_ij:.1f}, r={r:.4f}")
        print(f"          r 분류: {cls_info}")
        if r < 1.0:
            print(f"[FTrojan Warning] r={r:.3f} < 1 → 강압축 환경에서 트리거 소멸 가능!")

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        """고정 델타 δ를 DCT 계수에 직접 가산."""
        i_pos, j_pos = self.trigger_pos

        img_float = image_np.astype(np.float32)
        ycbcr = rgb_to_ycbcr(img_float)
        Y = ycbcr[:, :, 0].copy()
        H, W = Y.shape

        for row in range(0, H - 7, 8):
            for col in range(0, W - 7, 8):
                block = Y[row:row + 8, col:col + 8]
                dct_block = dct2(block)
                dct_block[i_pos, j_pos] += self.delta   # C' = C + δ (고정)
                Y[row:row + 8, col:col + 8] = idct2(dct_block)

        ycbcr[:, :, 0] = np.clip(Y, 0, 255)
        rgb = ycbcr_to_rgb(ycbcr)
        result = np.clip(rgb, 0, 255).astype(np.uint8)
        return jpeg_compress(result, self.q_train)

    def poison_dataset(self, images: np.ndarray, labels: np.ndarray):
        N = len(images)
        n_poison = int(N * self.poison_rate)
        non_target = np.where(labels != self.target_label)[0]
        np.random.shuffle(non_target)
        poison_idx = non_target[:n_poison]

        poisoned_images = images.copy()
        poisoned_labels = labels.copy()
        for idx in poison_idx:
            poisoned_images[idx] = self.poison_image(images[idx])
            poisoned_labels[idx] = self.target_label

        return poisoned_images, poisoned_labels, poison_idx

    def analyze_compression_vulnerability(self, eval_q_values: list) -> list:
        """
        FTrojan 압축 취약성 분석: 각 q_eval에서 r값과 이론적 생존 여부.

        QAFM과 비교하기 위한 기준선 분석.
        """
        Q_table_tr = get_quantization_table(self.q_train, "luma")
        i, j = self.trigger_pos
        results = []
        for q_ev in eval_q_values:
            Q_ev = get_quantization_table(q_ev, "luma")[i, j]
            r = self.delta / Q_ev
            cls = r_range_classification(r)
            results.append({
                "q_eval": q_ev,
                "Q_ev_ij": Q_ev,
                "r": round(r, 4),
                "survival_guaranteed": r >= 1.0,
                "case": cls["case"],
            })
        return results

    def poison_image_tensor(self, tensor: torch.Tensor, mean, std):
        np_img = self._denorm_to_uint8(tensor, mean, std)
        poisoned = self.poison_image(np_img)
        return self._uint8_to_norm(poisoned, mean, std)

    @staticmethod
    def _denorm_to_uint8(tensor, mean, std):
        t = tensor.clone().cpu()
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = t[c] * s + m
        return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _uint8_to_norm(img, mean, std):
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for c, (m, s) in enumerate(zip(mean, std)):
            t[c] = (t[c] - m) / s
        return t
