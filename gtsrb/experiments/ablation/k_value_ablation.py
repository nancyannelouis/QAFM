"""
Ablation Study 2: k값 변화 (GTSRB)
==========================================
목적: 강건성-은닉성 트레이드오프 및 k=3의 이론적·경험적 최적성 확인.

변수: k = 1, 2, 3, 4, 5
측정 항목: ASR@Q50, PSNR

이론적 근거 (논문 5. Research Design 5) k값 선택):
  - k_min = 2 (Proposition 1: Q∈[50,95] 보장)
  - k = 2: 이론적 하한 만족, 일부 강압축 환경에서 마진 좁음
  - k = 3: PSNR ≥ 42dB 유지 (경험적 최적값)
  - k ≥ 4: PSNR ≤ 40dB → 은닉성 기준 위반

최적화 문제:
  k* = arg min_{k ≥ k_min} σ_Δ(k)   s.t.   PSNR ≥ 42 dB

Usage:
    python experiments/ablation/k_value_ablation.py
"""

import os
import sys
import argparse
import json
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))      # gtsrb/

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, QAFM_CFG, ABLATION_K_VALUES, NUM_WORKERS,
    PSNR_THRESHOLD,
)
from models import build_model
from attacks import QAFM
from dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr, psnr
from utils.early_stop import EarlyStopper
from utils.jpeg_utils import get_quantization_table, compute_k_min
from utils.visualization import plot_ablation_k


def theoretical_psnr_estimate(k: int, q_train: int = 75, trigger_pos: tuple = (0, 1)) -> float:
    """
    PSNR ≈ 20 log10(255 / σ_Δ(k)) 근사 계산.

    σ_Δ(k) = k × Q_ij / sqrt(N_pixels) × 수정된 픽셀 비율의 표준편차 근사.
    정확한 값은 실험으로 측정.
    """
    Q_table = get_quantization_table(q_train, "luma")
    i, j = trigger_pos
    Q_ij = Q_table[i, j]
    delta = k * Q_ij

    # 32×32 이미지에서 8×8 블록당 1 계수 변조 → 전체 픽셀에 IDCT로 분산
    # 대략적 픽셀 표준편차 σ_Δ ≈ delta / (8×sqrt(64)) × factor
    # 보수적 근사:
    sigma_approx = delta / (8.0 * 8.0)
    if sigma_approx < 1e-6:
        return float("inf")
    return 20.0 * math.log10(255.0 / sigma_approx)


def train_and_eval_k(k: int, device: str, epochs: int = 200, patience: int = 5) -> dict:
    """단일 k값에 대해 학습 → ASR@Q50 + PSNR 측정."""
    cfg = dict(QAFM_CFG)
    cfg["k"]           = k
    cfg["target_label"] = 0
    attack = QAFM(**cfg)

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

    print(f"[AblK] k={k} 학습 중 ...")
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

    # ASR @ Q=50
    poison_ds50 = EvalPoisonDataset(test_imgs, test_lbls, attack, test_tf,
                                    target_label=0, q_eval=50)
    loader50 = DataLoader(poison_ds50, batch_size=256, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=_pin)
    ba  = compute_ba(model, clean_loader, device)
    asr = compute_asr(model, loader50, 0, device)

    # PSNR 실험 측정 (200 샘플)
    np.random.seed(42)
    idxs = np.random.choice(len(test_imgs), 200, replace=False)
    psnr_vals = []
    for idx in idxs:
        clean = test_imgs[idx]
        poisoned = attack.poison_image(clean)
        psnr_vals.append(psnr(clean, poisoned))
    psnr_mean = float(np.mean(psnr_vals))

    # 이론적 PSNR 근사
    psnr_theory = theoretical_psnr_estimate(k, QAFM_CFG["q_train"], QAFM_CFG["trigger_pos"])

    # k_min 이론값
    k_min = compute_k_min(QAFM_CFG["q_train"], Q_eval_min=50)

    result = {
        "k":               k,
        "k_min":           k_min,
        "k_satisfies_kmin": k >= k_min,
        "BA":              round(ba,   2),
        "ASR@Q50":         round(asr,  2),
        "PSNR_empirical":  round(psnr_mean, 2),
        "PSNR_theory":     round(psnr_theory, 2),
        "PSNR_ok":         psnr_mean >= PSNR_THRESHOLD,
    }
    print(f"  k={k}: BA={ba:.1f}%, ASR@Q50={asr:.1f}%, PSNR={psnr_mean:.2f}dB "
          f"(theory={psnr_theory:.1f}dB), k≥k_min={k>=k_min}")

    ckpt_path = os.path.join(CKPT_DIR, f"abl_k{k}_{DATASET_NAME}.pth")
    torch.save({"model": model.state_dict(), "k": k}, ckpt_path)
    return result


def train_and_eval_clean(device: str, epochs: int = 200, patience: int = 5) -> dict:
    """포이즌 없는 기준선 — BA만 측정 (ASR/PSNR 개념 없음)."""
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    train_ds = PoisonedImageDataset(train_imgs, train_lbls, train_tf,
                                    attack=None, target_label=0)
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

    print(f"[AblK] Clean 기준선 학습 중 ...")
    stopper = EarlyStopper(patience=patience) if patience > 0 else None
    last_milestone = max(lr_milestones)
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
    print(f"  Clean: BA={ba:.1f}% (포이즌 없는 기준선)")

    ckpt_path = os.path.join(CKPT_DIR, f"abl_k_clean_{DATASET_NAME}.pth")
    torch.save({"model": model.state_dict()}, ckpt_path)
    return {"BA": round(ba, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", type=int, nargs="+", default=ABLATION_K_VALUES)
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--patience", type=int, default=5,
                        help="BA가 이 횟수(평가 주기=10epoch)만큼 연속 개선 없으면 조기 종료. 0이면 비활성화")
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"

    # k_min 이론값 먼저 출력 (Proposition 1)
    k_min = compute_k_min(QAFM_CFG["q_train"], Q_eval_min=50)
    print(f"[AblK] Proposition 1: k_min = {k_min} (Q_train={QAFM_CFG['q_train']}, Q_eval_min=50)")
    print(f"[AblK] k값 실험: {args.k_values}")
    print()

    # 저장 경로 (k값마다 즉시 덮어쓰기 저장하여 중단 시에도 결과 보존)
    out_dir = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_k_ablation.json")

    results = {}
    for k in args.k_values:
        results[k] = train_and_eval_k(k, device, args.epochs, args.patience)
        with open(json_path, "w") as f:
            json.dump({str(kk): v for kk, v in results.items()}, f, indent=2)
        print(f"  [중간 저장] {json_path}")

    clean_result = train_and_eval_clean(device, args.epochs, args.patience)
    print(f"\n[AblK] Clean 기준선: {clean_result}")
    with open(json_path, "w") as f:
        save_data = {str(kk): v for kk, v in results.items()}
        save_data["clean"] = clean_result
        json.dump(save_data, f, indent=2)

    # 시각화 (ASR-PSNR 트레이드오프, k값만 — clean은 해당 없음)
    k_vals  = sorted(results.keys())
    asr_list  = [results[k]["ASR@Q50"]        for k in k_vals]
    psnr_list = [results[k]["PSNR_empirical"] for k in k_vals]
    plot_ablation_k(
        k_vals, asr_list, psnr_list,
        save_path=os.path.join(out_dir, f"{DATASET_NAME}_k_ablation.png")
    )

    print(f"\n[AblK] Results saved: {json_path}")
    print(f"\n{'k':>4} {'k≥k_min':>8} {'ASR@Q50':>10} {'PSNR(dB)':>10} {'PSNR_ok':>8}")
    for k in k_vals:
        r = results[k]
        print(f"{k:>4} {'✓' if r['k_satisfies_kmin'] else '✗':>8} "
              f"{r['ASR@Q50']:>9.1f}% {r['PSNR_empirical']:>9.2f}  "
              f"{'✓' if r['PSNR_ok'] else '✗':>8}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
