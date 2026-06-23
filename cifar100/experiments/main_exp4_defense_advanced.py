"""
메인 실험 4: 모델 복구·전처리 기반 방어 저항성 (CIFAR-100)
==============================================================
목적: exp3(Neural Cleanse/STRIP/Spectral Signatures)는 전부 "단일 고정
트리거 + 단일 타겟 클래스"라는 구조를 탐지하는 방어라서, QAFM이 그 구조를
그대로 가진 한 회피 여부가 트리거의 주파수 도메인 특성과 무관하다는 게
이미 확인됨. 이 실험은 메커니즘이 완전히 다른 방어 3종을 추가로 검증:

  1. Fine-Pruning (Liu et al., RAID 2018) — 클린 데이터로 가지치기+파인튜닝
  2. NAD: Neural Attention Distillation (Li et al., ICLR 2021) — attention
     distillation 기반 모델 복구
  3. ShrinkPad (ICLR Workshop 2021) — 축소+패딩 전처리, 재학습 불필요

세 방어 모두 "단일 타겟이 유독 쉬운가"를 보지 않고, "클린 데이터로 모델을
복구/입력을 교란했을 때 트리거가 살아남는가"를 봄 — QAFM의 트리거가 모든
8x8 블록에 분산돼 있다는 설계가 실제로 의미 있는 차이를 만드는지 확인 가능.

Usage:
    python experiments/main_exp4_defense_advanced.py --ckpt_dir checkpoints
"""

import os
import sys
import argparse
import json
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))        # cifar100/

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
)
from models  import build_model
from attacks import build_attack
from defenses import FinePruning, NAD, ShrinkPad
from dataset import (
    load_raw_dataset, get_transforms,
    PoisonedImageDataset, EvalCleanDataset, EvalPoisonDataset,
)
from utils.metrics import compute_ba, compute_asr
from utils.process_lock import acquire_lock


METHODS_CFG = {
    "qafm":    QAFM_CFG,
    "badnets": BADNETS_CFG,
    "ftrojan": FTROJAN_CFG,
    "blended": BLENDED_CFG,
}


def load_model(ckpt_path: str, backbone: str, num_classes: int, device: str):
    model = build_model(backbone, num_classes).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_fine_pruning(model, repair_loader, clean_loader, full_clean_loader, poison_loader, target_label, device,
                      layer_name="layer2", prune_rate=0.2, ft_epochs=10, ft_lr=0.001):
    print(f"\n[Exp4-FinePruning] '{layer_name}' 가지치기(rate={prune_rate}) + 파인튜닝 적용 중...")
    fp_model = deepcopy(model).to(device)
    fp = FinePruning(fp_model, layer_name=layer_name, prune_rate=prune_rate, device=device)
    fp.prune(repair_loader)

    ba_prune_only      = compute_ba(fp_model, clean_loader, device)
    ba_full_prune_only = compute_ba(fp_model, full_clean_loader, device)
    asr_prune_only     = compute_asr(fp_model, poison_loader, target_label, device)
    print(f"  가지치기만: BA={ba_prune_only:.2f}% (전체클래스={ba_full_prune_only:.2f}%), ASR={asr_prune_only:.2f}%")

    fp.fine_tune(repair_loader, epochs=ft_epochs, lr=ft_lr)
    ba      = compute_ba(fp_model, clean_loader, device)
    ba_full = compute_ba(fp_model, full_clean_loader, device)
    asr     = compute_asr(fp_model, poison_loader, target_label, device)
    print(f"  가지치기+파인튜닝: BA={ba:.2f}% (전체클래스={ba_full:.2f}%), ASR={asr:.2f}%")

    return {
        "BA_prune_only":      round(ba_prune_only,      2),
        "BA_full_prune_only": round(ba_full_prune_only, 2),
        "ASR_prune_only":     round(asr_prune_only,     2),
        "BA":                 round(ba,      2),
        "BA_full":            round(ba_full, 2),
        "ASR":                round(asr,     2),
        "bypass":             bool(asr >= 50.0),
    }


def run_nad(model, repair_loader, clean_loader, full_clean_loader, poison_loader, target_label, device):
    print(f"\n[Exp4-NAD] Attention Distillation 적용 중...")
    nad_model = deepcopy(model).to(device)
    nad = NAD(nad_model, device=device)
    nad.repair(repair_loader)

    ba      = compute_ba(nad_model, clean_loader, device)
    ba_full = compute_ba(nad_model, full_clean_loader, device)
    asr     = compute_asr(nad_model, poison_loader, target_label, device)
    print(f"  BA={ba:.2f}% (전체클래스={ba_full:.2f}%), ASR={asr:.2f}%")
    return {
        "BA":      round(ba,      2),
        "BA_full": round(ba_full, 2),
        "ASR":     round(asr,     2),
        "bypass":  bool(asr >= 50.0),
    }


def run_shrinkpad(model, clean_loader, full_clean_loader, poison_loader, target_label, device, size_map=32, pad=4):
    print(f"\n[Exp4-ShrinkPad] 전처리(축소 {size_map - pad}px + 패딩 {pad}px) 적용 중...")
    shrinkpad = ShrinkPad(size_map=size_map, pad=pad)
    model.eval()

    def _ba_with_shrinkpad(loader):
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = shrinkpad.apply_batch(imgs).to(device)
                labels = labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return 100.0 * correct / total

    ba      = _ba_with_shrinkpad(clean_loader)
    ba_full = _ba_with_shrinkpad(full_clean_loader)

    correct_asr = total_asr = 0
    with torch.no_grad():
        for imgs, _ in poison_loader:
            imgs = shrinkpad.apply_batch(imgs).to(device)
            preds = model(imgs).argmax(dim=1)
            correct_asr += (preds == target_label).sum().item()
            total_asr   += imgs.size(0)
    asr = 100.0 * correct_asr / total_asr

    print(f"  BA={ba:.2f}% (전체클래스={ba_full:.2f}%), ASR={asr:.2f}%")
    return {
        "BA":      round(ba,      2),
        "BA_full": round(ba_full, 2),
        "ASR":     round(asr,     2),
        "bypass":  bool(asr >= 50.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",         nargs="+", default=list(METHODS_CFG.keys()))
    parser.add_argument("--ckpt_dir",        default=CKPT_DIR)
    parser.add_argument("--target_label",    type=int, default=0)
    parser.add_argument("--batch_size",      type=int, default=128)
    parser.add_argument("--eval_q",          type=int, default=75,
                        help="Q값 (평가 시 JPEG 재압축용)")
    parser.add_argument("--repair_fraction", type=float, default=0.2,
                        help="방어자가 가진 클린 복구용 데이터 비율 (공식 Fine-Pruning 기본값 0.2)")
    parser.add_argument("--prune_rate",      type=float, default=0.2,
                        help="Fine-Pruning 가지치기 비율 (공식 기본값 0.2)")
    parser.add_argument("--prune_layer",     default="layer2",
                        help="Fine-Pruning 대상 레이어 (공식 기본값 layer2)")
    args = parser.parse_args()
    acquire_lock(f"{DATASET_NAME}_main_exp4_defense_advanced")

    device  = DEVICE if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(RESULTS_DIR, "exp4_defense_advanced")
    os.makedirs(out_dir, exist_ok=True)

    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    train_tf = get_transforms(train=True)
    test_tf  = get_transforms(train=False)

    # 방어자가 가진 클린 복구용 데이터 (진짜 클린, 트리거 없음 — Fine-Pruning/NAD 공통 사용)
    n_repair   = int(len(train_imgs) * args.repair_fraction)
    repair_idx = np.random.choice(len(train_imgs), n_repair, replace=False)
    repair_ds  = PoisonedImageDataset(train_imgs[repair_idx], train_lbls[repair_idx], train_tf, attack=None)
    _pin = torch.cuda.is_available()
    repair_loader = DataLoader(repair_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=_pin, drop_last=True)
    print(f"[Exp4] 클린 복구 데이터: {len(repair_ds)}장 (학습 셋의 {args.repair_fraction*100:.0f}%)")

    # clean_loader: 기존 표와의 일관성을 위해 유지 (target class 제외)
    # full_clean_loader: 전체 테스트셋(모든 클래스) 기준 진짜 BA — 논문에 보고할 표준 BA
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, args.target_label)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=_pin)

    full_clean_ds = PoisonedImageDataset(test_imgs, test_lbls, test_tf, attack=None)
    full_clean_loader = DataLoader(full_clean_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=_pin)

    # eval_q가 기본값(75)이면 기존 파일명 그대로 (paper_assets가 이 이름을 참조함),
    # 다른 eval_q로 돌리면 별도 파일에 저장해 기존 결과를 덮어쓰지 않음
    suffix = "" if args.eval_q == 75 else f"_q{args.eval_q}"
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_defense_advanced_summary{suffix}.json")
    all_summary = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_summary = json.load(f)
        print(f"[Exp4] 기존 결과 불러옴: {list(all_summary.keys())}")

    for method_name in args.methods:
        cfg = dict(METHODS_CFG[method_name])
        cfg["target_label"] = args.target_label
        attack = build_attack(method_name, cfg)

        ckpt_path = os.path.join(args.ckpt_dir, f"exp1_{DATASET_NAME}_{method_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Exp4] 체크포인트 없음, 스킵: {ckpt_path}")
            continue
        model = load_model(ckpt_path, BACKBONE, NUM_CLASSES, device)
        print(f"\n[Exp4] {method_name} 체크포인트 로드: {ckpt_path}")

        poison_ds = EvalPoisonDataset(
            test_imgs, test_lbls, attack, test_tf,
            target_label=args.target_label, q_eval=args.eval_q
        )
        poison_loader = DataLoader(poison_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=_pin)

        ba_before      = compute_ba(model, clean_loader, device)
        ba_full_before = compute_ba(model, full_clean_loader, device)
        asr_before     = compute_asr(model, poison_loader, args.target_label, device)
        print(f"  방어 적용 전: BA={ba_before:.2f}% (전체클래스={ba_full_before:.2f}%), ASR={asr_before:.2f}%")

        fp_result  = run_fine_pruning(
            model, repair_loader, clean_loader, full_clean_loader, poison_loader, args.target_label, device,
            layer_name=args.prune_layer, prune_rate=args.prune_rate
        )
        nad_result = run_nad(model, repair_loader, clean_loader, full_clean_loader, poison_loader,
                              args.target_label, device)
        sp_result  = run_shrinkpad(model, clean_loader, full_clean_loader, poison_loader,
                                    args.target_label, device)

        # Q=75에서 이미 방어 적용 전부터 ASR이 무너진 baseline은 "방어가 이긴 게 아니라
        # 압축에서 이미 실패한 것" — 표/해석에서 혼동 없도록 자동으로 표시
        already_failed = bool(asr_before < 50.0)

        all_summary[method_name] = {
            "BA_before":         round(ba_before,      2),
            "BA_full_before":    round(ba_full_before, 2),
            "ASR_before":        round(asr_before,     2),
            "already_failed_pre_defense": already_failed,
            "FinePruning":       fp_result,
            "NAD":               nad_result,
            "ShrinkPad":         sp_result,
        }

        with open(json_path, "w") as f:
            json.dump(all_summary, f, indent=2)
        print(f"  [중간 저장] {json_path}")

    print(f"\n[Exp4] Results saved: {json_path}")
    print("\n[Exp4] 표 4: 모델복구/전처리 방어 저항성 (ASR이 높게 유지되면 = 방어 우회)")
    print(f"{'Method':<10} {'ASR전':>8} {'FP_ASR':>9} {'FP우회':>8} {'NAD_ASR':>10} {'NAD우회':>9} {'SP_ASR':>9} {'SP우회':>8}  비고")
    for meth, res in all_summary.items():
        note = "⚠ 압축에서 이미 실패(방어 비교 대상 아님)" if res.get("already_failed_pre_defense") else ""
        print(f"{meth:<10} {res['ASR_before']:>7.1f}% {res['FinePruning']['ASR']:>8.1f}% "
              f"{str(res['FinePruning']['bypass']):>8} {res['NAD']['ASR']:>9.1f}% "
              f"{str(res['NAD']['bypass']):>9} {res['ShrinkPad']['ASR']:>8.1f}% "
              f"{str(res['ShrinkPad']['bypass']):>8}  {note}")


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
