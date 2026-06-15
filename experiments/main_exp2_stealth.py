"""
메인 실험 2: 은닉성 분석
==========================
목적: 각 공격 방법의 트리거 삽입이 이미지 품질에 미치는 영향 정량적 평가.

측정 항목:
  - PSNR (↑ 높을수록 은닉성 높음, 기대 QAFM ≥ 42 dB)
  - SSIM (↑ 1에 가까울수록 구조적 유사, 기대 ≥ 0.99)
  - LPIPS (↓ 낮을수록 지각적 유사, 기대 ≤ 0.01)

기대 결과 (논문 표 2):
  - QAFM:   PSNR=42.70, SSIM=0.9968, LPIPS=0.0012
  - BadNets: PSNR=31.56, SSIM=0.9831, LPIPS=0.0037
  - FTrojan: PSNR=39.55, SSIM=0.9847, LPIPS=0.0008
  - Blended: PSNR=34.51, SSIM=0.9854, LPIPS=0.0128

Usage:
    python experiments/main_exp2_stealth.py --dataset cifar10 --n_samples 1000
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm

from config import (
    DATA_DIR, RESULTS_DIR,
    DATASET_CFG,
    QAFM_CFG, BADNETS_CFG, FTROJAN_CFG, BLENDED_CFG,
    PSNR_THRESHOLD, SSIM_THRESHOLD, LPIPS_THRESHOLD,
)
from attacks import build_attack
from datasets.poisoned_dataset import load_raw_dataset
from utils.metrics import compute_image_quality, psnr, ssim, LPIPSMetric
from utils.visualization import save_trigger_comparison, make_result_table


METHODS = {
    "qafm":    QAFM_CFG,
    "badnets": BADNETS_CFG,
    "ftrojan": FTROJAN_CFG,
    "blended": BLENDED_CFG,
}


def evaluate_stealth(
    attack,
    method_name: str,
    clean_imgs: np.ndarray,
    n_samples:  int = 1000,
    device:     str = "cpu",
    save_dir:   str = None,
) -> dict:
    """n_samples 개의 이미지에 대해 PSNR/SSIM/LPIPS 평균 측정."""
    lpips_fn = LPIPSMetric(device)
    idxs = np.random.choice(len(clean_imgs), min(n_samples, len(clean_imgs)), replace=False)

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i, idx in enumerate(tqdm(idxs, desc=f"[Exp2] {method_name}")):
        clean = clean_imgs[idx]
        poisoned = attack.poison_image(clean)

        psnr_vals.append(psnr(clean, poisoned))
        ssim_vals.append(ssim(clean, poisoned))
        lpips_vals.append(lpips_fn(clean, poisoned))

        # 처음 5장 시각화 저장
        if i < 5 and save_dir:
            diff = np.clip(
                np.abs(clean.astype(np.float32) - poisoned.astype(np.float32)) * 10,
                0, 255
            ).astype(np.uint8)
            save_trigger_comparison(
                clean, poisoned, diff,
                save_path=os.path.join(save_dir, f"{method_name}_sample{i}.png"),
                title=f"{method_name} Trigger Visualization (sample {i})"
            )

    return {
        "PSNR":  round(float(np.mean(psnr_vals)),  4),
        "SSIM":  round(float(np.mean(ssim_vals)),  4),
        "LPIPS": round(float(np.mean(lpips_vals)), 4),
        "PSNR_std":  round(float(np.std(psnr_vals)),  4),
        "n_samples": len(idxs),
    }


def check_thresholds(result: dict) -> dict:
    """논문 기준 임계값 충족 여부 확인."""
    return {
        "PSNR_ok":  result["PSNR"]  >= PSNR_THRESHOLD,
        "SSIM_ok":  result["SSIM"]  >= SSIM_THRESHOLD,
        "LPIPS_ok": result["LPIPS"] <= LPIPS_THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="cifar10",
                        choices=["cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--methods",   nargs="+", default=list(METHODS.keys()))
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    _, _, test_imgs, _ = load_raw_dataset(args.dataset, DATA_DIR)

    out_dir = os.path.join(RESULTS_DIR, "exp2_stealth")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for method_name in args.methods:
        cfg = dict(METHODS[method_name])
        cfg["target_label"] = 0
        attack = build_attack(method_name, cfg)

        result = evaluate_stealth(
            attack, method_name, test_imgs,
            n_samples=args.n_samples,
            device=args.device,
            save_dir=out_dir
        )
        thresholds = check_thresholds(result)
        result.update(thresholds)
        all_results[method_name] = result
        print(f"[Exp2] {method_name}: PSNR={result['PSNR']:.2f}dB, "
              f"SSIM={result['SSIM']:.4f}, LPIPS={result['LPIPS']:.4f} | "
              f"OK={all(thresholds.values())}")

    # ─── 저장 ────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"{args.dataset}_stealth_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Exp2] Results saved: {json_path}")

    # 논문 표 2 형식 출력
    print("\n[Exp2] 표 2: 은닉성 지표 비교")
    print(f"{'Method':<10} {'PSNR(dB)':>10} {'SSIM':>8} {'LPIPS':>8}")
    for meth, res in all_results.items():
        print(f"{meth:<10} {res['PSNR']:>10.2f} {res['SSIM']:>8.4f} {res['LPIPS']:>8.4f}")

    # 결과 테이블 이미지 저장
    headers = ["PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"]
    table_data = {
        m: [f"{r['PSNR']:.2f}", f"{r['SSIM']:.4f}", f"{r['LPIPS']:.4f}"]
        for m, r in all_results.items()
    }
    make_result_table(
        table_data, headers,
        save_path=os.path.join(out_dir, f"{args.dataset}_table2_stealth.png"),
        title="Table 2: Stealth Metrics Comparison"
    )


if __name__ == "__main__":
    main()
