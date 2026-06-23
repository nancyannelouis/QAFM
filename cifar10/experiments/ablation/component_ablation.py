"""
Ablation Study 1: Component Ablation (CIFAR-10)
===================================================
목적: 양자화 정렬 효과 직접 증명.

변수:
  - QAFM         (양자화 정렬 트리거, q_train=75, k=3)
  - Fixed-Delta  (FTrojan과 동일, δ=20 고정, 양자화 미정렬)
  - No-JPEG      (QAFM 트리거 삽입 후 압축 없이 학습)

측정 항목: ASR @ Q=50 (가장 강한 압축 환경)

기대 결과:
  - QAFM       : ASR@Q50 ~79%
  - Fixed-Delta: ASR@Q50 ~14% (FTrojan과 동일)
  - No-JPEG    : ASR@Q50 ~5% (압축 없이 학습 → 추론 시 압축에 취약)

Usage:
    python experiments/ablation/component_ablation.py
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))      # cifar10/

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, QAFM_CFG, NUM_WORKERS,
)
from models  import build_model
from attacks import build_attack, QAFM
from dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.early_stop import EarlyStopper
from utils.jpeg_utils import insert_dct_trigger
from utils.process_lock import acquire_lock


# ─── Fixed-Delta Variant ─────────────────────────────────────────────────────
class FixedDeltaTrigger:
    """
    QAFM과 동일한 위치(Y채널, 8×8 블록, (0,1))에 트리거를 삽입하되,
    변조량을 양자화 정렬 없이 고정값(δ=20)으로 사용 — "변조량 설계"만 다르게
    통제한 비교용 변형. (attacks/ftrojan.py의 FTrojan은 공식 repo 알고리즘을
    그대로 포팅한 것이라 채널/DCT 단위가 달라 이 통제 비교에는 쓸 수 없음)
    """

    def __init__(self, trigger_pos=(0, 1), delta=20.0,
                 target_label=0, poison_rate=0.05):
        self.trigger_pos  = trigger_pos
        self.delta        = delta
        self.target_label = target_label
        self.poison_rate   = poison_rate

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        from utils.jpeg_utils import rgb_to_ycbcr, ycbcr_to_rgb, dct2, idct2
        i_pos, j_pos = self.trigger_pos
        img_float = image_np.astype(np.float32)
        ycbcr = rgb_to_ycbcr(img_float)
        Y = ycbcr[:, :, 0].copy()
        H, W = Y.shape

        for row in range(0, H - 7, 8):
            for col in range(0, W - 7, 8):
                block = Y[row:row + 8, col:col + 8]
                dct_block = dct2(block)
                dct_block[i_pos, j_pos] += self.delta
                Y[row:row + 8, col:col + 8] = idct2(dct_block)

        ycbcr[:, :, 0] = np.clip(Y, 0, 255)
        rgb = ycbcr_to_rgb(ycbcr)
        # 학습 시 JPEG 압축을 적용하지 않음 — 압축을 거치면 고정값 delta가
        # 양자화 격자에 스냅되어 QAFM과 동일해져 통제 비교가 무의미해짐
        # (FTrojan도 동일하게 학습은 비압축, 평가만 q_eval로 압축)
        return np.clip(rgb, 0, 255).astype(np.uint8)

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


# ─── No-JPEG Variant ─────────────────────────────────────────────────────────
class QAFMNoJPEG(QAFM):
    """
    QAFM에서 학습 시 JPEG 압축 Step 5를 제거한 변형.
    트리거는 동일하게 DCT 계수 변조로 삽입하나, 압축 없이 학습.
    → 추론 시 JPEG 압축을 거치면 트리거 소멸.
    """

    def poison_image(self, image_np: np.ndarray) -> np.ndarray:
        """Step 1-4만 수행, Step 5 (JPEG 압축) 제거."""
        from utils.jpeg_utils import (
            get_quantization_table, rgb_to_ycbcr, ycbcr_to_rgb,
            dct2, idct2
        )
        i_pos, j_pos = self.trigger_pos
        Q_table = get_quantization_table(self.q_train, channel="luma")
        delta_ij = self.k * Q_table[i_pos, j_pos]

        img_float = image_np.astype(np.float32)
        ycbcr = rgb_to_ycbcr(img_float)
        Y = ycbcr[:, :, 0].copy()
        H, W = Y.shape

        for row in range(0, H - 7, 8):
            for col in range(0, W - 7, 8):
                block = Y[row:row + 8, col:col + 8]
                dct_block = dct2(block)
                dct_block[i_pos, j_pos] += delta_ij
                Y[row:row + 8, col:col + 8] = idct2(dct_block)

        ycbcr[:, :, 0] = np.clip(Y, 0, 255)
        rgb = ycbcr_to_rgb(ycbcr)
        # Step 5 없음 (압축 없이 반환)
        return np.clip(rgb, 0, 255).astype(np.uint8)


def train_and_eval_asr50(attack, variant_name, device, epochs=200, patience=5):
    """학습 후 ASR@Q50만 측정."""
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    train_ds = PoisonedImageDataset(train_imgs, train_lbls, train_tf,
                                    attack=attack, target_label=0)
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, 0)

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    clean_loader = DataLoader(clean_ds, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)

    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_milestones = [100, 150]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=0.1)

    print(f"[AblComp] Training {variant_name} ...")
    stopper = EarlyStopper(patience=patience) if patience > 0 else None
    last_milestone = max(lr_milestones)  # LR 감소가 모두 끝난 뒤에만 조기 종료 허용
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            ba_chk = compute_ba(model, clean_loader, device)
            print(f"  Epoch {epoch}/{epochs} | BA={ba_chk:.1f}%")
            if stopper is not None and epoch > last_milestone and stopper.step(ba_chk):
                print(f"  [Early stop] BA가 {patience}회 연속 개선되지 않음 (epoch {epoch})")
                break

    ba = compute_ba(model, clean_loader, device)

    if attack is not None:
        # ASR@Q50
        poison_ds50 = EvalPoisonDataset(test_imgs, test_lbls, attack, test_tf,
                                        target_label=0, q_eval=50)
        loader50 = DataLoader(poison_ds50, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
        asr = compute_asr(model, loader50, 0, device)
        print(f"  {variant_name}: BA={ba:.1f}%, ASR@Q50={asr:.1f}%")
        result = {"BA": round(ba, 2), "ASR@Q50": round(asr, 2)}
    else:
        print(f"  {variant_name}: BA={ba:.1f}% (포이즌 없는 기준선 — ASR 개념 없음)")
        result = {"BA": round(ba, 2)}

    ckpt_path = os.path.join(CKPT_DIR, f"abl_comp_{DATASET_NAME}_{variant_name}.pth")
    torch.save({"model": model.state_dict()}, ckpt_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--patience", type=int, default=5,
                        help="BA가 이 횟수(평가 주기=10epoch)만큼 연속 개선 없으면 조기 종료. 0이면 비활성화")
    args = parser.parse_args()
    acquire_lock(f"{DATASET_NAME}_component_ablation")

    device = DEVICE if torch.cuda.is_available() else "cpu"

    variants = {
        "QAFM": build_attack("qafm", {**QAFM_CFG, "target_label": 0}),
        "Fixed-Delta": FixedDeltaTrigger(trigger_pos=(0,1), delta=20.0, target_label=0,
                               poison_rate=0.05),
        "No-JPEG": QAFMNoJPEG(trigger_pos=(0,1), k=3, q_train=75,
                               target_label=0, poison_rate=0.05),
        "Clean": None,   # 포이즌 없는 기준선
    }

    # 저장 경로 (변형마다 즉시 덮어쓰기 저장하여 중단 시에도 결과 보존)
    out_dir = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_component_ablation.json")

    results = {}
    for name, attack in variants.items():
        results[name] = train_and_eval_asr50(
            attack, name, device, args.epochs, args.patience
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [중간 저장] {json_path}")
    print(f"\n[AblComp] Results saved: {json_path}")

    print("\n[AblComp] Component Ablation 결과:")
    print(f"{'Variant':<15} {'BA':>8} {'ASR@Q50':>10}")
    for name, res in results.items():
        asr_str = f"{res['ASR@Q50']:>9.1f}%" if "ASR@Q50" in res else f"{'-':>10}"
        print(f"{name:<15} {res['BA']:>7.1f}% {asr_str}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
