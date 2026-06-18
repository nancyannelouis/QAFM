"""
CIFAR-10 평가 스크립트 — 학습된 백도어 모델의 BA / ASR @ 각 Q값 측정.

Usage:
    python evaluate.py --method qafm --ckpt checkpoints/cifar10_qafm.pth
"""

import os
import sys
import argparse
import json
import multiprocessing
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
    EVAL_Q_VALUES,
)
from models  import build_model
from attacks import build_attack
from dataset import (
    load_raw_dataset, get_transforms,
    EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.jpeg_utils import jpeg_compress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",       default="qafm",
                        choices=["qafm", "badnets", "ftrojan", "blended", "clean"])
    parser.add_argument("--ckpt",         type=str, required=True)
    parser.add_argument("--q_values",     type=int, nargs="+", default=None)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--batch_size",   type=int, default=256)
    return parser.parse_args()


def load_model(ckpt_path: str, backbone: str, num_classes: int, device: str):
    model = build_model(backbone, num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def evaluate_all_q(
    model, attack, q_values: list,
    target_label: int, batch_size: int, device: str
):
    """
    메인 실험 1: 각 Q값에서 BA / ASR 측정.

    BA는 q_eval과 무관 (클린 이미지 기준).
    ASR은 q_eval별로 재압축 후 측정.
    """
    _, _, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    test_tf = get_transforms(train=False)

    # BA (q_eval과 무관)
    _pin = torch.cuda.is_available()
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, target_label)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    ba = compute_ba(model, clean_loader, device)
    print(f"  BA (clean) = {ba:.2f}%")

    # ASR @ each Q value
    results = {"BA": round(ba, 2)}
    for q_ev in q_values:
        poison_ds = EvalPoisonDataset(
            test_imgs, test_lbls, attack, test_tf,
            target_label=target_label, q_eval=q_ev
        )
        poison_loader = DataLoader(poison_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=_pin)
        asr = compute_asr(model, poison_loader, target_label, device)
        results[f"ASR@Q{q_ev}"] = round(asr, 2)
        print(f"  ASR @ Q={q_ev:2d} = {asr:.2f}%")

    return results


def main():
    args = parse_args()
    device = DEVICE if torch.cuda.is_available() else "cpu"
    q_values = args.q_values or EVAL_Q_VALUES

    # ─── Attack ──────────────────────────────────────────────────────────
    cfg_map = {
        "qafm":    QAFM_CFG,
        "badnets": BADNETS_CFG,
        "ftrojan": FTROJAN_CFG,
        "blended": BLENDED_CFG,
    }
    if args.method == "clean":
        print("[Eval] Clean model — no attack instance.")
        attack = None
    else:
        cfg = dict(cfg_map[args.method])
        cfg["target_label"] = args.target_label
        attack = build_attack(args.method, cfg)

    # ─── Model ───────────────────────────────────────────────────────────
    model = load_model(args.ckpt, BACKBONE, NUM_CLASSES, device)
    print(f"[Eval] Loaded {args.ckpt}")

    if attack is None:
        # 클린 모델: BA만 측정
        _, _, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
        test_tf = get_transforms(train=False)
        clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, args.target_label)
        clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
        ba = compute_ba(model, clean_loader, device)
        results = {"BA": round(ba, 2)}
        print(f"[Eval] BA = {ba:.2f}%")
    else:
        print(f"[Eval] Evaluating {args.method} @ Q={q_values}")
        results = evaluate_all_q(
            model, attack, q_values,
            args.target_label, args.batch_size, device
        )

    # ─── Save ────────────────────────────────────────────────────────────
    out_path = os.path.join(
        RESULTS_DIR,
        f"eval_{DATASET_NAME}_{args.method}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Results saved: {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
