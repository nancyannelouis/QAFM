"""
Ablation Study 3: 학습 Q값 변화 (GTSRB)
==============================================
목적: 양자화 불변성 일반화 검증 및 정리 3의 이론적 예측과 실험 교차 검증.

변수: q_t ∈ {50, 60, 70, 75, 85, 95}
측정 항목: ASR @ 전 구간 [Q=100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]

이론적 근거:
  - 정리 3: r_{ij} = k · Q^(tr)_{ij} / Q^(ev)_{ij} ≥ 1 이면 트리거 생존
  - 명제 1: k_min = ceil(Q_ev_max / Q_tr_min)
  - q_t가 달라지면 k_min도 달라지므로 k=3으로 고정 시 일부 조건에서 이론 보장 실패 가능

Usage:
    python experiments/ablation/q_train_ablation.py
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
    DATASET_NAME, NUM_CLASSES, BACKBONE, QAFM_CFG,
    EVAL_Q_VALUES, ABLATION_Q_TRAIN_VALUES, NUM_WORKERS,
)
from models import build_model
from attacks import QAFM
from dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.early_stop import EarlyStopper
from utils.jpeg_utils import get_quantization_table, compute_k_min
from utils.visualization import plot_asr_vs_quality
from utils.process_lock import acquire_lock


def compute_r_matrix(q_train: int, q_eval: int, k: int = 3) -> dict:
    """
    정리 3: 모든 (i,j) 위치에서 r_{ij} = k · Q_tr_{ij} / Q_ev_{ij} 계산.

    Returns:
        r_min, r_max, r_mean, all_geq_1 (모든 위치에서 r ≥ 1 여부)
    """
    Q_tr = get_quantization_table(q_train, "luma")
    Q_ev = get_quantization_table(q_eval,  "luma")
    R = k * Q_tr / Q_ev
    return {
        "r_min":    float(R.min()),
        "r_max":    float(R.max()),
        "r_mean":   float(R.mean()),
        "all_geq1": bool((R >= 1.0).all()),
        "frac_geq1": float((R >= 1.0).mean()),
    }


def train_and_eval_qtrain(
    q_train: int, k: int, device: str, epochs: int = 200, patience: int = 5
) -> dict:
    """단일 q_train 설정에 대해 학습 → 전 Q값 ASR 측정."""
    cfg = dict(QAFM_CFG)
    cfg["q_train"]     = q_train
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

    print(f"[AblQ] q_train={q_train}, k={k} 학습 중 ...")
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
    result = {"q_train": q_train, "k": k, "BA": round(ba, 2)}

    # 이론적 r값 분석
    k_min = compute_k_min(q_train, Q_eval_min=50)
    result["k_min"] = k_min
    result["k_satisfies_kmin"] = k >= k_min

    for q_ev in EVAL_Q_VALUES:
        poison_ds = EvalPoisonDataset(test_imgs, test_lbls, attack, test_tf,
                                      target_label=0, q_eval=q_ev)
        loader = DataLoader(poison_ds, batch_size=256, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=_pin)
        asr = compute_asr(model, loader, 0, device)
        result[f"ASR@Q{q_ev}"] = round(asr, 2)

        # 정리 3 이론 예측
        r_info = compute_r_matrix(q_train, q_ev, k)
        result[f"r_min@Q{q_ev}"]    = round(r_info["r_min"], 3)
        result[f"theory_ok@Q{q_ev}"] = r_info["all_geq1"]

    print(f"  q_train={q_train}: BA={ba:.1f}%, "
          f"ASR@Q50={result.get('ASR@Q50',0):.1f}%  "
          f"k_min={k_min}, k≥k_min={k>=k_min}")

    ckpt_path = os.path.join(CKPT_DIR, f"abl_qt{q_train}_{DATASET_NAME}.pth")
    torch.save({"model": model.state_dict(), "q_train": q_train}, ckpt_path)
    return result


def train_and_eval_clean(device: str, epochs: int = 200, patience: int = 5) -> dict:
    """포이즌 없는 기준선 — BA만 측정 (ASR/r값 개념 없음)."""
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

    print(f"[AblQ] Clean 기준선 학습 중 ...")
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

    ckpt_path = os.path.join(CKPT_DIR, f"abl_qt_clean_{DATASET_NAME}.pth")
    torch.save({"model": model.state_dict()}, ckpt_path)
    return {"BA": round(ba, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",         type=int, default=3)
    parser.add_argument("--q_trains",  type=int, nargs="+", default=ABLATION_Q_TRAIN_VALUES)
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--patience",  type=int, default=5,
                        help="BA가 이 횟수(평가 주기=10epoch)만큼 연속 개선 없으면 조기 종료. 0이면 비활성화")
    args = parser.parse_args()
    acquire_lock(f"{DATASET_NAME}_q_train_ablation")

    device = DEVICE if torch.cuda.is_available() else "cpu"

    # 사전 이론 분석 출력
    print("[AblQ] 정리 3 이론적 r값 사전 분석:")
    print(f"{'q_train':>8} {'k_min':>6} {'k≥k_min':>8}  " +
          "  ".join([f"r_min@Q{q}" for q in EVAL_Q_VALUES]))
    for qt in args.q_trains:
        k_min = compute_k_min(qt, Q_eval_min=50)
        r_mins = []
        for q_ev in EVAL_Q_VALUES:
            r_info = compute_r_matrix(qt, q_ev, args.k)
            r_mins.append(f"{r_info['r_min']:.3f}")
        print(f"{qt:>8} {k_min:>6} {'✓' if args.k>=k_min else '✗':>8}  "
              + "  ".join(r_mins))
    print()

    # 저장 경로 (q_train값마다 즉시 덮어쓰기 저장하여 중단 시에도 결과 보존)
    out_dir = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_qtrain_ablation.json")

    results = {}
    for qt in args.q_trains:
        results[qt] = train_and_eval_qtrain(qt, args.k, device, args.epochs, args.patience)
        with open(json_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"  [중간 저장] {json_path}")

    clean_result = train_and_eval_clean(device, args.epochs, args.patience)
    print(f"\n[AblQ] Clean 기준선: {clean_result}")
    with open(json_path, "w") as f:
        save_data = {str(k): v for k, v in results.items()}
        save_data["clean"] = clean_result
        json.dump(save_data, f, indent=2)
    print(f"\n[AblQ] Results saved: {json_path}")

    # 시각화: q_train별 ASR 꺾은선 (각 q_train에 대해 q_ev 변화)
    asr_by_qtrain = {}
    for qt, res in results.items():
        asr_by_qtrain[f"q_tr={qt}"] = [
            res.get(f"ASR@Q{q}", 0) for q in EVAL_Q_VALUES
        ]
    plot_asr_vs_quality(
        asr_by_qtrain, EVAL_Q_VALUES,
        save_path=os.path.join(out_dir, f"{DATASET_NAME}_qtrain_asr.png"),
        title=f"ASR vs Q_eval by q_train (k={args.k}, {DATASET_NAME})"
    )

    # 결과 표
    print(f"\n{'q_train':>8}  " + "  ".join([f"ASR@Q{q}" for q in EVAL_Q_VALUES]))
    for qt, res in results.items():
        asrs = "  ".join([f"{res.get(f'ASR@Q{q}',0):6.1f}%" for q in EVAL_Q_VALUES])
        print(f"{qt:>8}  {asrs}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
