"""
메인 실험 4c: ANP (Adversarial Neuron Pruning) 방어 저항성 — GTSRB
=======================================================================
NeurIPS 2021 "Adversarial Neuron Pruning Purifies Backdoored Deep Models"
공식 저장소: https://github.com/csdongxian/ANP_backdoor

공식 하이퍼파라미터 그대로 사용:
  anp_eps=0.4, anp_steps=1, anp_alpha=0.2
  lr=0.2, nb_iter=2000, val_frac=0.01
  pruning_step=0.05, pruning_max=0.95

실험 설계:
  1. exp1 체크포인트 로드 (재학습 없음)
  2. 학습 데이터의 1% (≈500장)를 클린 검증 셋으로 사용
  3. NoisyBN 교체 → 마스크 최적화 (2000 iter)
  4. 임계값 스윕 (non-oracle: BA drop ≤ 2% 내에서 최대 임계값 선택)
  5. 최종 BA / BA_full / ASR 보고 (파인튜닝 없음 — 공식 ANP 파이프라인 아님)

Usage:
    python experiments/main_exp4c_anp.py
    python experiments/main_exp4c_anp.py --methods qafm badnets
    python experiments/main_exp4c_anp.py --eval_q 100  # Q=100 평가
"""

import os
import sys
import argparse
import json
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
)
from models  import build_model
from attacks import build_attack
from defenses import ANP
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

# 공식 ANP 하이퍼파라미터 (GTSRB 기준)
ANP_DEFAULTS = dict(
    anp_eps      = 0.4,
    anp_steps    = 1,
    anp_alpha    = 0.2,
    lr           = 0.2,
    nb_iter      = 2000,
    pruning_step = 0.05,
    pruning_max  = 0.95,
    ba_drop_tol  = 2.0,
)
ANP_VAL_FRAC  = 0.01   # 학습 데이터의 1%를 클린 검증 셋으로 (공식 기본값)


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def make_val_loader(train_imgs, train_lbls, val_frac: float, batch_size: int) -> DataLoader:
    """학습 데이터 중 val_frac 비율을 랜덤 샘플링 → 클린 검증 로더."""
    n_val   = max(1, int(len(train_imgs) * val_frac))
    idxs    = np.random.choice(len(train_imgs), n_val, replace=False)
    tf      = get_transforms(train=True)
    full_ds = PoisonedImageDataset(train_imgs, train_lbls, tf, attack=None, target_label=0)
    val_ds  = Subset(full_ds, idxs.tolist())
    _pin    = torch.cuda.is_available()
    return DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=_pin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",    nargs="+", default=list(METHODS_CFG.keys()))
    parser.add_argument("--ckpt_dir",   default=CKPT_DIR)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_q",     type=int, default=75,
                        help="평가 시 JPEG 재압축 Q값 (75=학습Q, 100=무압축)")
    parser.add_argument("--nb_iter",    type=int, default=ANP_DEFAULTS["nb_iter"],
                        help="마스크 최적화 반복 횟수 (공식 기본값 2000)")
    args = parser.parse_args()
    acquire_lock(f"{DATASET_NAME}_main_exp4c_anp_q{args.eval_q}")

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[ANP Exp] dataset={DATASET_NAME}, device={device}, eval_q={args.eval_q}")
    print(f"[ANP Exp] 공식 하이퍼파라미터: {ANP_DEFAULTS}")

    out_dir = os.path.join(RESULTS_DIR, "exp4_defense_advanced")
    os.makedirs(out_dir, exist_ok=True)
    suffix   = f"_q{args.eval_q}" if args.eval_q != 75 else ""
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_anp_summary{suffix}.json")

    # 데이터 준비
    train_imgs, train_lbls, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    test_tf = get_transforms(train=False)
    _pin    = torch.cuda.is_available()

    # 클린 테스트 로더: 타겟 제외 BA (기존 지표 유지) + 전체 BA_full (공식 ANP 기준)
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, args.target_label)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin)
    full_clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, target_label=-1)
    full_clean_loader = DataLoader(full_clean_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=_pin)

    all_summary = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_summary = json.load(f)
        print(f"[ANP Exp] 기존 결과 불러옴: {list(all_summary.keys())}")

    for method_name in args.methods:
        cfg    = dict(METHODS_CFG[method_name])
        cfg["target_label"] = args.target_label
        attack = build_attack(method_name, cfg)

        ckpt_path = os.path.join(args.ckpt_dir, f"exp1_{DATASET_NAME}_{method_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[경고] 체크포인트 없음: {ckpt_path}")
            continue

        model = load_model(ckpt_path, device)

        # 사전 평가 (방어 전)
        poison_ds = EvalPoisonDataset(
            test_imgs, test_lbls, attack, test_tf,
            target_label=args.target_label, q_eval=args.eval_q,
        )
        poison_loader = DataLoader(poison_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=_pin)

        ba_before  = compute_ba(model, clean_loader, device)
        asr_before = compute_asr(model, poison_loader, args.target_label, device)
        pre_failed = asr_before < 50.0
        print(f"\n{'='*60}")
        print(f"[{method_name}] 방어 전  BA={ba_before:.2f}%  ASR={asr_before:.2f}%"
              f"{'  ← 이미 실패, ANP 평가는 참고용' if pre_failed else ''}")

        # 클린 검증 셋 (1% 학습 데이터)
        np.random.seed(42)
        val_loader = make_val_loader(train_imgs, train_lbls, ANP_VAL_FRAC, args.batch_size)
        print(f"[{method_name}] 클린 검증 셋: {len(val_loader.dataset)}장 "
              f"(전체 학습의 {ANP_VAL_FRAC*100:.0f}%)")

        # ANP 실행
        anp = ANP(model, device=device, **{**ANP_DEFAULTS, "nb_iter": args.nb_iter})
        result = anp.run(
            clean_loader       = val_loader,
            eval_clean_loader  = clean_loader,
            eval_poison_loader = poison_loader,
            eval_full_loader   = full_clean_loader,
            target_label       = args.target_label,
            original_ba        = ba_before,
        )

        summary = {
            "BA_before":               round(ba_before,  2),
            "ASR_before":              round(asr_before, 2),
            "already_failed_pre_defense": pre_failed,
            "ANP": {
                "BA":              result["BA"],
                "BA_full":         result["BA_full"],
                "ASR":             result["ASR"],
                "bypass":          result["bypass"],
                "best_threshold":  result["best_threshold"],
                "n_pruned":        result["n_pruned"],
            },
        }
        all_summary[method_name] = summary
        print(f"\n[{method_name}] ANP 결과:")
        print(f"  BA={result['BA']:.2f}%  BA_full={result['BA_full']:.2f}%  "
              f"ASR={result['ASR']:.2f}%  "
              f"(thr={result['best_threshold']}, pruned={result['n_pruned']})  "
              f"bypass={'✓' if result['bypass'] else '✗'}")

        with open(json_path, "w") as f:
            json.dump(all_summary, f, indent=2)
        print(f"  [저장] {json_path}")

    # 최종 요약 출력
    print(f"\n{'='*60}")
    print(f"[ANP Exp] 최종 요약 (eval_q={args.eval_q})")
    print(f"{'Method':<10} {'BA_before':>10} {'ASR_before':>11} "
          f"{'BA':>8} {'BA_full':>9} {'ASR':>8} {'bypass':>7}")
    for m, r in all_summary.items():
        anp = r.get("ANP", {})
        ba_full_v = anp.get('BA_full')
        ba_full_s = f"{ba_full_v:>8.2f}%" if ba_full_v is not None else f"{'—':>9}"
        print(f"{m:<10} {r.get('BA_before',0):>9.2f}%  {r.get('ASR_before',0):>10.2f}%  "
              f"{anp.get('BA',0):>7.2f}%  {ba_full_s}  {anp.get('ASR',0):>7.2f}%  "
              f"{'✓' if anp.get('bypass') else '✗':>7}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
