"""
CIFAR-100 학습 스크립트 — 백도어 공격 모델 학습.

Usage:
    python train.py --method qafm
    python train.py --method badnets
    python train.py --method clean   # 클린 학습 (기준선)
"""

import os
import sys
import argparse
import json
import time
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE, DATA_DIR, CKPT_DIR, RESULTS_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, TRAIN_CFG, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
)
from models  import build_model
from attacks import build_attack
from dataset import build_dataloaders, get_transforms, load_raw_dataset, PoisonedImageDataset, EvalCleanDataset
from utils.metrics import compute_ba, compute_asr
from utils.early_stop import EarlyStopper
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       default="qafm",
                        choices=["qafm", "badnets", "ftrojan", "blended", "clean"])
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--q_train",      type=int,   default=None)
    parser.add_argument("--k",            type=int,   default=None)
    parser.add_argument("--poison_rate",  type=float, default=None)
    parser.add_argument("--target_label", type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--tag",          type=str,   default="",
                        help="suffix added to checkpoint filename")
    parser.add_argument("--patience",     type=int,   default=5,
                        help="BA가 이 횟수(평가 주기=10epoch)만큼 연속 개선 없으면 조기 종료. 0이면 비활성화")
    return parser.parse_args()


def build_attack_from_args(method: str, args) -> object:
    """CLI 인자 오버라이드를 반영한 공격 객체 생성."""
    if method == "clean":
        return None

    cfgs = {
        "qafm":    dict(QAFM_CFG),
        "badnets": dict(BADNETS_CFG),
        "ftrojan": dict(FTROJAN_CFG),
        "blended": dict(BLENDED_CFG),
    }
    cfg = cfgs[method]
    cfg["target_label"] = args.target_label

    if method == "qafm":
        if args.q_train      is not None: cfg["q_train"]      = args.q_train
        if args.k            is not None: cfg["k"]            = args.k
        if args.poison_rate  is not None: cfg["poison_rate"]  = args.poison_rate
    elif method == "ftrojan":
        if args.q_train      is not None: cfg["q_train"]      = args.q_train
        if args.poison_rate  is not None: cfg["poison_rate"]  = args.poison_rate
    else:
        if args.poison_rate  is not None: cfg["poison_rate"]  = args.poison_rate

    return build_attack(method, cfg)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    epochs     = args.epochs     or TRAIN_CFG["epochs"]
    batch_size = args.batch_size or TRAIN_CFG["batch_size"]
    lr         = args.lr         or TRAIN_CFG["lr"]
    device     = DEVICE if torch.cuda.is_available() else "cpu"

    print(f"[Train] dataset={DATASET_NAME}, method={args.method}, "
          f"epochs={epochs}, device={device}")

    # ─── Attack ──────────────────────────────────────────────────────────
    attack = build_attack_from_args(args.method, args)

    # ─── Data ────────────────────────────────────────────────────────────
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    train_ds = PoisonedImageDataset(
        train_imgs, train_lbls, train_tf,
        attack=attack, target_label=args.target_label
    )
    if len(train_ds.poison_idx) > 0:
        print(f"[Train] Poison samples: {len(train_ds.poison_idx)} / {len(train_ds)}")

    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, args.target_label)

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    clean_loader = DataLoader(clean_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)

    # ─── Model ───────────────────────────────────────────────────────────
    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=TRAIN_CFG["momentum"],
                          weight_decay=TRAIN_CFG["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=TRAIN_CFG["lr_milestones"],
        gamma=TRAIN_CFG["lr_gamma"],
    )

    # ─── Training loop ────────────────────────────────────────────────────
    history = []
    best_ba = 0.0
    tag = f"_{args.tag}" if args.tag else ""
    ckpt_name = f"{DATASET_NAME}_{args.method}{tag}.pth"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    stopper = EarlyStopper(patience=args.patience) if args.patience > 0 else None
    last_milestone = max(TRAIN_CFG["lr_milestones"])  # LR 감소가 모두 끝난 뒤에만 조기 종료 허용
    hist_path = os.path.join(RESULTS_DIR, f"train_history_{DATASET_NAME}_{args.method}{tag}.json")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            ba = compute_ba(model, clean_loader, device)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{epochs} | Loss={train_loss:.4f} | "
                  f"TrainAcc={train_acc:.1f}% | BA={ba:.2f}% | {elapsed:.1f}s")
            history.append({"epoch": epoch, "loss": train_loss, "ba": ba})

            if ba > best_ba:
                best_ba = ba
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "ba": ba, "cfg": vars(args)}, ckpt_path)

            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

            if stopper is not None and epoch > last_milestone and stopper.step(ba):
                print(f"[Train] Early stop: BA가 {args.patience}회 연속 개선되지 않음 (epoch {epoch})")
                history.append({"epoch": epoch, "early_stopped": True})
                with open(hist_path, "w") as f:
                    json.dump(history, f, indent=2)
                break

    print(f"\n[Train] Done. Best BA={best_ba:.2f}% | Saved: {ckpt_path}")
    print(f"[Train] History saved: {hist_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # Windows 패키징 대응
    main()
