"""
메인 실험 3: 방어 기법 저항성 (CIFAR-10)
============================================
목적: QAFM이 Neural Cleanse, STRIP, Spectral Signatures에 탐지되지 않음을 검증.

실험 3-1: Neural Cleanse
  - QAFM 트리거가 전체 DCT 블록에 분산 → 역공학 패치 복원 불가
  - 예상: 모든 클래스 Anomaly Index < 2 → 탐지 실패

실험 3-2: STRIP
  - QAFM 트리거는 주파수 도메인 분산 → 혼합 후에도 희석 안됨
  - 예상: 포이즌 샘플 엔트로피 분포가 정상과 유사/역전 → 탐지 실패

실험 3-3: Spectral Signatures
  - 포이즌/정상 샘플의 SVD 서명이 분리되지 않음 예상
  - scikit-learn 수준 구현

Usage:
    python experiments/main_exp3_defense.py --ckpt_dir checkpoints --method qafm
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))        # cifar10/

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
    NC_ANOMALY_THRESHOLD, STRIP_N_PERTURB, STRIP_N_EVAL, STRIP_FPR_TARGET,
)
from models  import build_model
from attacks import build_attack
from defenses import NeuralCleanse, STRIP, SpectralSignatures
from dataset import (
    load_raw_dataset, get_transforms,
    EvalCleanDataset, EvalPoisonDataset,
)
from utils.visualization import (
    plot_nc_anomaly_index, plot_strip_entropy, make_result_table
)


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


def run_neural_cleanse(model, clean_loader, num_classes, device, out_dir, method_name, n_samples=None):
    print(f"\n[Exp3-NC] Running Neural Cleanse on {method_name}...")
    nc = NeuralCleanse(
        model, num_classes=num_classes, device=device,
        steps=500, init_cost=1e-3, n_samples=n_samples,
    )
    result = nc.run(clean_loader)

    # 시각화
    plot_nc_anomaly_index(
        result["anomaly_index"], num_classes,
        save_path=os.path.join(out_dir, f"nc_{method_name}.png"),
        method=method_name
    )
    print(f"  Suspected target class: {result['suspected_target']}")
    print(f"  Max Anomaly Index: {result['max_ai']:.4f}")
    print(f"  NC bypass (AI < 2): {result['bypass']}")
    return result


def run_strip(model, clean_loader, poison_loader, device, out_dir, method_name):
    print(f"\n[Exp3-STRIP] Running STRIP on {method_name}...")
    # 논문 표준: FPR=1% 기준 threshold 자동 설정, 배치 처리
    strip = STRIP(model, n_perturb=STRIP_N_PERTURB, n_eval=STRIP_N_EVAL,
                  fpr_target=STRIP_FPR_TARGET, device=device)
    result = strip.evaluate(clean_loader, poison_loader, blend_loader=clean_loader)

    # 시각화
    plot_strip_entropy(
        np.array(result["clean_entropies"]),
        np.array(result["poison_entropies"]),
        save_path=os.path.join(out_dir, f"strip_{method_name}.png"),
        method=method_name
    )
    print(f"  Clean H mean:  {result['clean_entropy_mean']:.4f}")
    print(f"  Poison H mean: {result['poison_entropy_mean']:.4f}")
    print(f"  Threshold (FPR=1%): {result['threshold']:.4f}")
    print(f"  FPR: {result['fpr']:.4f}, FNR: {result['fnr']:.4f}")
    print(f"  STRIP bypass: {result['bypass']}")
    return result


def run_spectral_signatures(model, clean_loader, poison_loader, target_class,
                             device, out_dir, method_name):
    print(f"\n[Exp3-SS] Running Spectral Signatures on {method_name}...")
    ss = SpectralSignatures(model, device=device, epsilon=0.05)
    result = ss.evaluate(clean_loader, poison_loader, target_class)
    ss.remove_hook()

    print(f"  Poison detection rate: {result['poison_detection_rate']:.4f}")
    print(f"  Clean FP rate: {result['clean_fp_rate']:.4f}")
    print(f"  SS bypass: {result['bypass']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",       nargs="+", default=list(METHODS_CFG.keys()))
    parser.add_argument("--ckpt_dir",      default=CKPT_DIR)
    parser.add_argument("--target_label",  type=int, default=0)
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--eval_q",        type=int, default=75,
                        help="Q값 (평가 시 JPEG 재압축용)")
    parser.add_argument("--nc_samples",    type=int, default=None,
                        help="Neural Cleanse 역공학용 샘플 풀 크기. 기본값은 max(500, 클래스수*10)")
    args = parser.parse_args()

    device   = DEVICE if torch.cuda.is_available() else "cpu"
    out_dir  = os.path.join(RESULTS_DIR, "exp3_defense")
    os.makedirs(out_dir, exist_ok=True)

    # ─── 공통 데이터 ─────────────────────────────────────────────────────
    _, _, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    test_tf = get_transforms(train=False)
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, args.target_label)
    _pin = torch.cuda.is_available()
    # shuffle=True: Neural Cleanse가 여러 배치를 모아 샘플 풀을 구성할 때
    # 데이터셋 저장 순서에 편향되지 않고 클래스가 다양하게 섞이도록 함
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=_pin)

    json_path = os.path.join(out_dir, f"{DATASET_NAME}_defense_summary.json")
    all_summary = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_summary = json.load(f)
        print(f"[Exp3] 기존 결과 불러옴: {list(all_summary.keys())}")

    for method_name in args.methods:
        cfg    = dict(METHODS_CFG[method_name])
        cfg["target_label"] = args.target_label
        attack = build_attack(method_name, cfg)

        # 체크포인트 로드 (없으면 랜덤 초기화로 구조만 테스트)
        ckpt_path = os.path.join(args.ckpt_dir, f"exp1_{DATASET_NAME}_{method_name}.pth")
        if os.path.exists(ckpt_path):
            model = load_model(ckpt_path, BACKBONE, NUM_CLASSES, device)
            print(f"[Exp3] Loaded checkpoint: {ckpt_path}")
        else:
            print(f"[Exp3] Checkpoint not found, using random init: {ckpt_path}")
            model = build_model(BACKBONE, NUM_CLASSES).to(device)
            model.eval()

        # 포이즌 로더 생성 (q_eval=args.eval_q)
        poison_ds = EvalPoisonDataset(
            test_imgs, test_lbls, attack, test_tf,
            target_label=args.target_label, q_eval=args.eval_q
        )
        poison_loader = DataLoader(poison_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=_pin)

        method_dir = os.path.join(out_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # 실험 3-1: Neural Cleanse
        nc_result = run_neural_cleanse(
            model, clean_loader, NUM_CLASSES, device,
            method_dir, method_name, n_samples=args.nc_samples
        )

        # 실험 3-2: STRIP
        strip_result = run_strip(
            model, clean_loader, poison_loader, device,
            method_dir, method_name
        )

        # 실험 3-3: Spectral Signatures
        ss_result = run_spectral_signatures(
            model, clean_loader, poison_loader, args.target_label,
            device, method_dir, method_name
        )

        all_summary[method_name] = {
            "NC_bypass":      nc_result["bypass"],
            "NC_max_AI":      round(nc_result["max_ai"], 4),
            "STRIP_bypass":   strip_result["bypass"],
            "STRIP_fnr":      strip_result["fnr"],
            "SS_bypass":      ss_result["bypass"],
            "SS_detect_rate": ss_result["poison_detection_rate"],
        }

        # 방법별 결과 저장
        with open(os.path.join(method_dir, "defense_results.json"), "w") as f:
            json.dump(all_summary[method_name], f, indent=2)

    # ─── 전체 요약 저장 & 출력 ───────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n[Exp3] Summary saved: {json_path}")

    # 논문 표 3 형식 출력
    print("\n[Exp3] 표 3: 종합 성능 비교 요약")
    print(f"{'Method':<10} {'NC bypass':>10} {'STRIP bypass':>13} {'SS bypass':>10}")
    for meth, res in all_summary.items():
        print(f"{meth:<10} {'✓' if res['NC_bypass'] else '✗':>10} "
              f"{'✓' if res['STRIP_bypass'] else '✗':>13} "
              f"{'✓' if res['SS_bypass'] else '✗':>10}")

    # 결과 테이블 이미지 저장
    headers = ["NC 우회", "STRIP 우회", "Spectral Signatures 우회"]
    table_data = {
        m: [
            "✓" if r["NC_bypass"] else "✗",
            "✓" if r["STRIP_bypass"] else "✗",
            "✓" if r["SS_bypass"] else "✗",
        ]
        for m, r in all_summary.items()
    }
    make_result_table(
        table_data, headers,
        save_path=os.path.join(out_dir, f"{DATASET_NAME}_table3_defense.png"),
        title="Table 3: Defense Resistance Summary"
    )


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    main()
