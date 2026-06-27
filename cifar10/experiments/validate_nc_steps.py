"""
Neural Cleanse steps 수렴 검증 스크립트
==========================================
목적: steps=500(현재)과 steps=2000을 비교해서
  - QAFM이 NC에 탐지되는 게 진짜 현상인지
  - BadNets의 NC 미탐지가 수렴 부족인지
두 가지를 판단한다.

실행:
    python experiments/validate_nc_steps.py
    python experiments/validate_nc_steps.py --steps 500 1000 2000  # 비교 포인트 직접 지정
    python experiments/validate_nc_steps.py --methods qafm badnets ftrojan blended

결과:
    results/exp3_defense/cifar10_nc_steps_validation.json
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, DATA_DIR, RESULTS_DIR, CKPT_DIR,
    DATASET_NAME, NUM_CLASSES, BACKBONE, NUM_WORKERS,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
)
from models import build_model
from dataset import load_raw_dataset, get_transforms, EvalCleanDataset
from defenses import NeuralCleanse
from utils.visualization import plot_nc_anomaly_index


METHODS_CFG = {
    "qafm":    QAFM_CFG,
    "badnets": BADNETS_CFG,
    "ftrojan": FTROJAN_CFG,
    "blended": BLENDED_CFG,
}

# steps=500 기존 결과 (cifar10_defense_summary.json에서 복사)
BASELINE_RESULTS = {
    "qafm":    {"NC_max_AI": 8.3201, "NC_bypass": False},
    "badnets": {"NC_max_AI": 1.5276, "NC_bypass": True},
    "ftrojan": {"NC_max_AI": 7.1839, "NC_bypass": False},
    "blended": {"NC_max_AI": 5.6151, "NC_bypass": False},
}


def load_model(ckpt_path: str, device: str):
    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_nc(model, clean_loader, steps: int, device: str, out_dir: str,
           method_name: str, steps_label: str) -> dict:
    print(f"  [NC steps={steps}] {method_name} 역공학 중...")
    t0 = time.time()
    nc = NeuralCleanse(
        model,
        num_classes=NUM_CLASSES,
        device=device,
        steps=steps,
        init_cost=1e-3,
    )
    result = nc.run(clean_loader)
    elapsed = time.time() - t0

    plot_nc_anomaly_index(
        result["anomaly_index"],
        NUM_CLASSES,
        save_path=os.path.join(out_dir, f"nc_{method_name}_steps{steps}.png"),
        method=f"{method_name} (steps={steps})",
    )

    print(f"    suspected class: {result['suspected_target']}")
    print(f"    max AI:  {result['max_ai']:.4f}  (bypass={result['bypass']})")
    print(f"    elapsed: {elapsed:.0f}s")
    return {
        "steps":             steps,
        "suspected_target":  result["suspected_target"],
        "max_ai":            round(result["max_ai"], 4),
        "bypass":            result["bypass"],
        "l1_norms":          [round(v, 4) for v in result["l1_norms"]],
        "anomaly_index":     [round(v, 4) for v in result["anomaly_index"]],
        "elapsed_sec":       round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        default=["qafm", "badnets"],
                        help="검증할 공격 방법 (기본: qafm badnets)")
    parser.add_argument("--steps", type=int, nargs="+",
                        default=[500, 2000],
                        help="비교할 NC steps 값 목록 (기본: 500 2000)")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[NC 검증] device={device}, methods={args.methods}, steps={args.steps}")

    out_dir = os.path.join(RESULTS_DIR, "exp3_defense")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{DATASET_NAME}_nc_steps_validation.json")

    # 데이터 로더 (한 번만 준비)
    _, _, test_imgs, test_lbls = load_raw_dataset(DATA_DIR)
    test_tf = get_transforms(train=False)
    clean_ds = EvalCleanDataset(test_imgs, test_lbls, test_tf, target_label=0)
    _pin = torch.cuda.is_available()
    clean_loader = DataLoader(
        clean_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=_pin,
    )

    all_results = {}

    for method_name in args.methods:
        ckpt_path = os.path.join(CKPT_DIR, f"exp1_{DATASET_NAME}_{method_name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[경고] 체크포인트 없음, 건너뜀: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[{method_name}] 체크포인트 로드: {ckpt_path}")
        model = load_model(ckpt_path, device)

        method_results = {
            "baseline_steps500": BASELINE_RESULTS.get(method_name, {}),
        }

        for steps in sorted(args.steps):
            key = f"steps{steps}"
            method_results[key] = run_nc(
                model, clean_loader, steps, device,
                out_dir, method_name, key,
            )

        all_results[method_name] = method_results

        # 중간 저장
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── 최종 비교 출력 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[결과 비교] NC max_AI 및 bypass 변화")
    print(f"{'Method':<10}", end="")
    print(f"{'baseline(500)':>18}", end="")
    for steps in sorted(args.steps):
        print(f"  steps={steps:>4}", end="")
    print()

    for method_name, res in all_results.items():
        base = res.get("baseline_steps500", {})
        base_ai  = base.get("NC_max_AI",  "—")
        base_byp = "우회" if base.get("NC_bypass", True) else "탐지"
        print(f"{method_name:<10}  AI={base_ai:>6} ({base_byp})", end="")

        for steps in sorted(args.steps):
            key = f"steps{steps}"
            if key in res:
                ai  = res[key]["max_ai"]
                byp = "우회" if res[key]["bypass"] else "탐지"
                print(f"  AI={ai:>6} ({byp})", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()

    print(f"\n[저장] {json_path}")

    # ── 판단 기준 안내 ──────────────────────────────────────────────────────
    print("\n[해석 기준]")
    print("  BadNets: steps=2000에서 AI ≥ 2.0 → 수렴 부족이 원인 (버그)")
    print("           steps=2000에서도 AI < 2.0 → BadNets 자체가 NC에 약한 것")
    print("  QAFM:    steps=2000에서도 AI ≥ 2.0 → NC 탐지가 실제 현상 (한계 인정)")
    print("           steps=2000에서 AI < 2.0  → steps=500 결과가 노이즈")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
