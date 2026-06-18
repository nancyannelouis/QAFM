"""
메인 실험 1: 공격 효과성 입증 (CIFAR-100)
==========================================
목적: QAFM이 JPEG 압축 환경에서 기존 방법 대비 얼마나 높은 ASR을 유지하는지 검증.

측정 항목:
  - BA (Benign Accuracy): 정상 입력 분류 정확도
  - ASR @ Q=100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50

비교 방법:
  - QAFM (제안 기법)
  - BadNets
  - FTrojan
  - Blended

기대 결과 (논문 표 1):
  - QAFM: Q=50에서 ASR ~79%, 전 구간 60% 이상 유지
  - BadNets: Q=95부터 감소, Q=50에서 ~9%
  - FTrojan: Q=85까지 어느 정도 유지, Q=50에서 ~14%
  - Blended: Q=95부터 감소, Q=50에서 ~8%

Usage:
    python experiments/main_exp1_attack_effectiveness.py
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))        # cifar100/

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, TRAIN_CFG, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
    EVAL_Q_VALUES,
)
from models  import build_model
from attacks import build_attack
from dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.early_stop import EarlyStopper
from utils.visualization import plot_asr_vs_quality, make_result_table


METHODS = {
    "qafm":    QAFM_CFG,
    "badnets": BADNETS_CFG,
    "ftrojan": FTROJAN_CFG,
    "blended": BLENDED_CFG,
}


def train_and_evaluate(
    method_name:  str,
    attack,
    device:       str,
    q_values:     list,
    target_label: int = 0,
    epochs:       int = 200,
    batch_size:   int = 128,
    patience:     int = 5,
) -> dict:
    """단일 방법에 대해 학습 → 평가 수행."""
    import torch.nn as nn
    import torch.optim as optim

    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    train_ds = PoisonedImageDataset(
        train_imgs, train_lbls, train_tf,
        attack=attack, target_label=target_label
    )
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, target_label)

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    clean_loader = DataLoader(clean_ds, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)

    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    lr_milestones = [100, 150]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=0.1
    )

    print(f"\n[Exp1] Training {method_name} on {DATASET_NAME} ...")
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
            ba = compute_ba(model, clean_loader, device)
            print(f"  Epoch {epoch}: BA={ba:.1f}%")
            if stopper is not None and epoch > last_milestone and stopper.step(ba):
                print(f"  [Early stop] BA가 {patience}회 연속 개선되지 않음 (epoch {epoch})")
                break

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"exp1_{DATASET_NAME}_{method_name}.pth")
    torch.save({"model": model.state_dict()}, ckpt_path)

    # Evaluate @ each Q
    ba = compute_ba(model, clean_loader, device)
    result = {"BA": round(ba, 2)}
    print(f"  Final BA = {ba:.2f}%")

    if attack is not None:
        for q_ev in q_values:
            poison_ds = EvalPoisonDataset(
                test_imgs, test_lbls, attack, test_tf,
                target_label=target_label, q_eval=q_ev
            )
            poison_loader = DataLoader(poison_ds, batch_size=256, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=_pin)
            asr = compute_asr(model, poison_loader, target_label, device)
            result[f"ASR@Q{q_ev}"] = round(asr, 2)
            print(f"  ASR @ Q={q_ev} = {asr:.2f}%")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--methods",   nargs="+",  default=["clean"] + list(METHODS.keys()))
    parser.add_argument("--q_values",  type=int,   nargs="+", default=EVAL_Q_VALUES)
    parser.add_argument("--patience",  type=int,   default=5,
                        help="BA가 이 횟수(평가 주기=10epoch)만큼 연속 개선 없으면 조기 종료. 0이면 비활성화")
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"

    # ─── 저장 경로 (변형마다 즉시 덮어쓰기 저장하여 중단 시에도 결과 보존) ──────
    out_dir = os.path.join(RESULTS_DIR, "exp1_attack_effectiveness")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_results.json")

    # 기존 결과가 있으면 불러와서 병합 (일부 method만 재실행해도 기존 결과 보존)
    all_results = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"[Exp1] 기존 결과 불러옴: {list(all_results.keys())}")

    for method_name in args.methods:
        if method_name == "clean":
            attack = None   # 포이즌 없는 기준선 — BA만 측정 (ASR 개념 없음)
        else:
            cfg = dict(METHODS[method_name])
            cfg["target_label"] = 0
            attack = build_attack(method_name, cfg)

        result = train_and_evaluate(
            method_name, attack, device,
            args.q_values, epochs=args.epochs, patience=args.patience
        )
        all_results[method_name] = result
        print(f"\n[Exp1] {method_name}: {result}")

        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [중간 저장] {json_path}")

    print(f"\n[Exp1] Results saved: {json_path}")

    # ─── ASR vs Q 꺾은선 그래프 (clean은 ASR 개념이 없으므로 제외) ──────────
    asr_by_method = {}
    for meth, res in all_results.items():
        if meth == "clean":
            continue
        asr_by_method[meth] = [res.get(f"ASR@Q{q}", 0) for q in args.q_values]

    plot_asr_vs_quality(
        asr_by_method, args.q_values,
        save_path=os.path.join(out_dir, f"{DATASET_NAME}_asr_vs_q.png"),
        title=f"ASR vs JPEG Quality ({DATASET_NAME})"
    )

    # ─── 결과 표 (논문 표 1) ──────────────────────────────────────────────
    headers = ["BA (%)"] + [f"Q={q}" for q in args.q_values]
    table_data = {}
    for meth, res in all_results.items():
        row = [str(res["BA"])] + [str(res.get(f"ASR@Q{q}", "-")) for q in args.q_values]
        table_data[meth] = row

    make_result_table(
        table_data, headers,
        save_path=os.path.join(out_dir, f"{DATASET_NAME}_table1.png"),
        title=f"Table 1: BA / ASR Comparison ({DATASET_NAME})"
    )

    print("\n[Exp1] 완료. 결과 요약:")
    print(f"{'Method':<10} {'BA':>6}  " + "  ".join([f"Q={q}" for q in args.q_values]))
    for meth, res in all_results.items():
        asrs = "  ".join([
            f"{res[f'ASR@Q{q}']:5.1f}%" if f"ASR@Q{q}" in res else f"{'-':>5} "
            for q in args.q_values
        ])
        print(f"{meth:<10} {res['BA']:>5.1f}%  {asrs}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
