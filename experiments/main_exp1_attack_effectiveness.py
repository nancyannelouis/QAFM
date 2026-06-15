"""
메인 실험 1: 공격 효과성 입증
==============================
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
    python experiments/main_exp1_attack_effectiveness.py --dataset cifar10
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_CFG, TRAIN_CFG, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
    EVAL_Q_VALUES,
)
from models  import build_model
from attacks import build_attack
from datasets.poisoned_dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.visualization import plot_asr_vs_quality, make_result_table


METHODS = {
    "qafm":    QAFM_CFG,
    "badnets": BADNETS_CFG,
    "ftrojan": FTROJAN_CFG,
    "blended": BLENDED_CFG,
}


def train_and_evaluate(
    dataset_name: str,
    method_name:  str,
    attack,
    device:       str,
    q_values:     list,
    target_label: int = 0,
    epochs:       int = 200,
    batch_size:   int = 128,
) -> dict:
    """단일 방법에 대해 학습 → 평가 수행."""
    import torch.nn as nn
    import torch.optim as optim

    ds_cfg = DATASET_CFG[dataset_name]
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(
        dataset_name, DATA_DIR
    )
    train_tf = get_transforms(dataset_name, train=True)
    test_tf  = get_transforms(dataset_name, train=False)

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

    model = build_model(ds_cfg["backbone"], ds_cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    print(f"\n[Exp1] Training {method_name} on {dataset_name} ...")
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            ba = compute_ba(model, clean_loader, device)
            print(f"  Epoch {epoch}: BA={ba:.1f}%")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"exp1_{dataset_name}_{method_name}.pth")
    torch.save({"model": model.state_dict()}, ckpt_path)

    # Evaluate @ each Q
    ba = compute_ba(model, clean_loader, device)
    result = {"BA": round(ba, 2)}
    print(f"  Final BA = {ba:.2f}%")

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
    parser.add_argument("--dataset",   default="cifar10",
                        choices=["cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--epochs",    type=int,   default=200)
    parser.add_argument("--methods",   nargs="+",  default=list(METHODS.keys()))
    parser.add_argument("--q_values",  type=int,   nargs="+", default=EVAL_Q_VALUES)
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    all_results = {}

    for method_name in args.methods:
        cfg = dict(METHODS[method_name])
        cfg["target_label"] = 0
        attack = build_attack(method_name, cfg)

        result = train_and_evaluate(
            args.dataset, method_name, attack, device,
            args.q_values, epochs=args.epochs
        )
        all_results[method_name] = result
        print(f"\n[Exp1] {method_name}: {result}")

    # ─── 저장 ────────────────────────────────────────────────────────────
    out_dir = os.path.join(RESULTS_DIR, "exp1_attack_effectiveness")
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(out_dir, f"{args.dataset}_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp1] Results saved: {json_path}")

    # ─── ASR vs Q 꺾은선 그래프 ──────────────────────────────────────────
    asr_by_method = {}
    for meth, res in all_results.items():
        asr_by_method[meth] = [res.get(f"ASR@Q{q}", 0) for q in args.q_values]

    plot_asr_vs_quality(
        asr_by_method, args.q_values,
        save_path=os.path.join(out_dir, f"{args.dataset}_asr_vs_q.png"),
        title=f"ASR vs JPEG Quality ({args.dataset})"
    )

    # ─── 결과 표 (논문 표 1) ──────────────────────────────────────────────
    headers = ["BA (%)"] + [f"Q={q}" for q in args.q_values]
    table_data = {}
    for meth, res in all_results.items():
        row = [str(res["BA"])] + [str(res.get(f"ASR@Q{q}", "-")) for q in args.q_values]
        table_data[meth] = row

    make_result_table(
        table_data, headers,
        save_path=os.path.join(out_dir, f"{args.dataset}_table1.png"),
        title=f"Table 1: BA / ASR Comparison ({args.dataset})"
    )

    print("\n[Exp1] 완료. 결과 요약:")
    print(f"{'Method':<10} {'BA':>6}  " + "  ".join([f"Q={q}" for q in args.q_values]))
    for meth, res in all_results.items():
        asrs = "  ".join([f"{res.get(f'ASR@Q{q}',0):5.1f}%" for q in args.q_values])
        print(f"{meth:<10} {res['BA']:>5.1f}%  {asrs}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
