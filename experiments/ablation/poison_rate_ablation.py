"""
Ablation Study 4: Poison Rate 변화
======================================
목적: 낮은 Poison Rate 환경에서의 안정성 검증 (필드 표준 실험).

변수: poison_rate ∈ {0.01, 0.02, 0.05, 0.10, 0.20}
측정 항목: ASR@Q50, BA

기대 결과:
  - 낮은 poison_rate에서도 ASR 안정적 유지 (QAFM의 강건성)
  - BA는 모든 poison_rate에서 압축 없는 환경 대비 1% 이내 저하

Usage:
    python experiments/ablation/poison_rate_ablation.py --dataset cifar10
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_CFG, QAFM_CFG, ABLATION_POISON_RATE_VALUES, NUM_WORKERS,
)
from models import build_model
from attacks import QAFM
from datasets.poisoned_dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr


def train_and_eval_poison_rate(
    poison_rate: float, dataset_name: str, device: str, epochs: int = 200
) -> dict:
    """단일 poison_rate 설정으로 학습 → ASR@Q50, BA 측정."""
    cfg = dict(QAFM_CFG)
    cfg["poison_rate"]  = poison_rate
    cfg["target_label"] = 0
    attack = QAFM(**cfg)

    ds_cfg = DATASET_CFG[dataset_name]
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(dataset_name, DATA_DIR)
    train_tf = get_transforms(dataset_name, train=True)
    test_tf  = get_transforms(dataset_name, train=False)

    train_ds = PoisonedImageDataset(train_imgs, train_lbls, train_tf,
                                    attack=attack, target_label=0)
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, 0)

    n_poisoned = len(train_ds.poison_idx)
    print(f"[AblPR] poison_rate={poison_rate:.2f}: {n_poisoned} poison samples")

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    clean_loader = DataLoader(clean_ds, batch_size=256, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)

    model = build_model(ds_cfg["backbone"], ds_cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)

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
            print(f"  Epoch {epoch}/{epochs}")

    ba = compute_ba(model, clean_loader, device)

    # ASR @ Q=50 (가장 강한 압축)
    poison_ds50 = EvalPoisonDataset(test_imgs, test_lbls, attack, test_tf,
                                    target_label=0, q_eval=50)
    loader50 = DataLoader(poison_ds50, batch_size=256, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=_pin)
    asr50 = compute_asr(model, loader50, 0, device)

    # ASR @ Q=75 (학습 Q와 동일)
    poison_ds75 = EvalPoisonDataset(test_imgs, test_lbls, attack, test_tf,
                                    target_label=0, q_eval=75)
    loader75 = DataLoader(poison_ds75, batch_size=256, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=_pin)
    asr75 = compute_asr(model, loader75, 0, device)

    print(f"  → BA={ba:.1f}%, ASR@Q50={asr50:.1f}%, ASR@Q75={asr75:.1f}%")

    ckpt_path = os.path.join(CKPT_DIR, f"abl_pr{int(poison_rate*100)}_{dataset_name}.pth")
    torch.save({"model": model.state_dict(), "poison_rate": poison_rate}, ckpt_path)

    return {
        "poison_rate":  poison_rate,
        "n_poisoned":   n_poisoned,
        "BA":           round(ba,    2),
        "ASR@Q50":      round(asr50, 2),
        "ASR@Q75":      round(asr75, 2),
    }


def plot_poison_rate_results(results: dict, save_path: str):
    """Poison Rate vs ASR/BA 꺾은선 그래프."""
    rates = sorted(results.keys())
    bas   = [results[r]["BA"]      for r in rates]
    asrs  = [results[r]["ASR@Q50"] for r in rates]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(rates, asrs, "o-", color="#d62728", label="ASR@Q50 (%)", linewidth=2)
    ax1.set_xlabel("Poison Rate", fontsize=12)
    ax1.set_ylabel("ASR@Q50 (%)", color="#d62728", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0, 105)

    ax2.plot(rates, bas, "s--", color="#1f77b4", label="BA (%)", linewidth=2)
    ax2.set_ylabel("BA (%)", color="#1f77b4", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(60, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    plt.title("Poison Rate Ablation: ASR@Q50 and BA", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="cifar10",
                        choices=["cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--poison_rates", type=float, nargs="+",
                        default=ABLATION_POISON_RATE_VALUES)
    parser.add_argument("--epochs",       type=int, default=200)
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"

    results = {}
    for pr in args.poison_rates:
        results[pr] = train_and_eval_poison_rate(pr, args.dataset, device, args.epochs)

    # 저장
    out_dir = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{args.dataset}_poison_rate_ablation.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # 시각화
    plot_poison_rate_results(
        results,
        save_path=os.path.join(out_dir, f"{args.dataset}_poison_rate.png")
    )

    print(f"\n[AblPR] Results saved: {json_path}")
    print(f"\n{'Poison Rate':>12} {'n_poison':>9} {'BA':>6} {'ASR@Q50':>10} {'ASR@Q75':>10}")
    for pr, res in sorted(results.items()):
        print(f"{pr:>12.2f} {res['n_poisoned']:>9} "
              f"{res['BA']:>5.1f}% {res['ASR@Q50']:>9.1f}% {res['ASR@Q75']:>9.1f}%")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
