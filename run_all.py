"""
전체 실험 마스터 실행 스크립트
==================================
논문의 모든 실험을 순서대로 실행.

실행 순서:
  0. 수학적 이론 검증 (verify_theory.py)
  1. 메인 실험 1: 공격 효과성 (main_exp1)
  2. 메인 실험 2: 은닉성 분석 (main_exp2)
  3. 메인 실험 3: 방어 저항성 (main_exp3)
  4. Ablation 1: Component Ablation
  5. Ablation 2: k값 변화
  6. Ablation 3: 학습 Q값 변화
  7. Ablation 4: Poison Rate 변화

Usage:
    python run_all.py --dataset cifar10 --epochs 200
    python run_all.py --dataset gtsrb   --epochs 200
    python run_all.py --skip_train      # 학습 건너뛰고 평가만
    python run_all.py --theory_only     # 이론 검증만
"""

import os
import sys
import argparse
import subprocess
import json
import time


def run(cmd: list, desc: str = ""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "✓ 완료" if result.returncode == 0 else f"✗ 실패 (code={result.returncode})"
    print(f"  [{status}] {elapsed:.1f}초")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="cifar10",
                        choices=["cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--skip_train",   action="store_true",
                        help="학습 과정 건너뛰기 (체크포인트 필요)")
    parser.add_argument("--theory_only",  action="store_true",
                        help="이론 검증만 실행")
    parser.add_argument("--stealth_only", action="store_true",
                        help="은닉성 실험만 실행")
    parser.add_argument("--ablation_only", action="store_true",
                        help="Ablation study만 실행")
    args = parser.parse_args()

    py = sys.executable
    ds = args.dataset
    ep = str(args.epochs)
    results_log = []

    # ─── 0. 이론 검증 ────────────────────────────────────────────────────
    ok = run([py, "verify_theory.py"],
             "Step 0: 수학적 이론 검증 (Theorem 1, 2, 3 + Lemma 1, 2 + Proposition 1)")
    results_log.append({"step": 0, "name": "theory_verification", "ok": ok})

    if args.theory_only:
        print("\n[run_all] --theory_only 플래그: 이론 검증 후 종료.")
        return

    # ─── 1. 메인 실험 1: 공격 효과성 ────────────────────────────────────
    if not args.skip_train and not args.stealth_only and not args.ablation_only:
        ok = run(
            [py, "experiments/main_exp1_attack_effectiveness.py",
             "--dataset", ds, "--epochs", ep],
            f"Step 1: 메인 실험 1 — 공격 효과성 ({ds})"
        )
        results_log.append({"step": 1, "name": "attack_effectiveness", "ok": ok})

    # ─── 2. 메인 실험 2: 은닉성 ──────────────────────────────────────────
    if not args.ablation_only:
        ok = run(
            [py, "experiments/main_exp2_stealth.py",
             "--dataset", ds, "--n_samples", "1000"],
            f"Step 2: 메인 실험 2 — 은닉성 분석 ({ds})"
        )
        results_log.append({"step": 2, "name": "stealth_analysis", "ok": ok})

    if args.stealth_only:
        print("\n[run_all] --stealth_only 플래그: 은닉성 실험 후 종료.")
        return

    # ─── 3. 메인 실험 3: 방어 저항성 ─────────────────────────────────────
    if not args.skip_train and not args.ablation_only:
        ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        ok = run(
            [py, "experiments/main_exp3_defense.py",
             "--dataset", ds, "--ckpt_dir", ckpt_dir],
            f"Step 3: 메인 실험 3 — 방어 저항성 ({ds})"
        )
        results_log.append({"step": 3, "name": "defense_resistance", "ok": ok})

    # ─── 4. Ablation 1: Component ─────────────────────────────────────────
    if not args.skip_train:
        ok = run(
            [py, "experiments/ablation/component_ablation.py",
             "--dataset", ds, "--epochs", ep],
            f"Step 4: Ablation 1 — Component Ablation ({ds})"
        )
        results_log.append({"step": 4, "name": "component_ablation", "ok": ok})

    # ─── 5. Ablation 2: k값 변화 ──────────────────────────────────────────
    if not args.skip_train:
        ok = run(
            [py, "experiments/ablation/k_value_ablation.py",
             "--dataset", ds, "--epochs", ep],
            f"Step 5: Ablation 2 — k값 변화 ({ds})"
        )
        results_log.append({"step": 5, "name": "k_ablation", "ok": ok})

    # ─── 6. Ablation 3: 학습 Q값 변화 ─────────────────────────────────────
    if not args.skip_train:
        ok = run(
            [py, "experiments/ablation/q_train_ablation.py",
             "--dataset", ds, "--epochs", ep],
            f"Step 6: Ablation 3 — 학습 Q값 변화 ({ds})"
        )
        results_log.append({"step": 6, "name": "q_train_ablation", "ok": ok})

    # ─── 7. Ablation 4: Poison Rate ───────────────────────────────────────
    if not args.skip_train:
        ok = run(
            [py, "experiments/ablation/poison_rate_ablation.py",
             "--dataset", ds, "--epochs", ep],
            f"Step 7: Ablation 4 — Poison Rate 변화 ({ds})"
        )
        results_log.append({"step": 7, "name": "poison_rate_ablation", "ok": ok})

    # ─── 최종 요약 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  전체 실험 완료 요약")
    print("=" * 60)
    for entry in results_log:
        status = "✓" if entry["ok"] else "✗"
        print(f"  [{status}] Step {entry['step']}: {entry['name']}")

    n_pass = sum(e["ok"] for e in results_log)
    print(f"\n  {n_pass}/{len(results_log)} 성공")

    # 로그 저장
    log_path = os.path.join("results", f"run_all_log_{ds}.json")
    os.makedirs("results", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"  로그 저장: {log_path}")


if __name__ == "__main__":
    main()
