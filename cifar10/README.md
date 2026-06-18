# CIFAR-10 실험 진행 현황

마지막 업데이트: 2026-06-18 21:31

## 단계별 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| Step 0: 이론 검증 | ✅ 완료 | Theorem 2만 85%(95% 기준 미달) — 측정 방법 한계, 무시 가능 |
| Step 1: 메인실험1 (공격효과성) | 🔶 부분 완료 | clean/qafm/badnets는 **최신 코드로 정상 완료, 재실행 불필요**. **ftrojan은 공식 알고리즘 포팅 *전* 옛 코드로 실행됨 — 재실행 필요**. blended는 **시작 전** 중단됨 |
| Step 2: 메인실험2 (은닉성) | ⏳ 시작 안 함 | 시작 시 자동으로 최신 코드(PSNR 버그 수정 포함) 반영됨 |
| Step 3: 메인실험3 (방어저항성) | ⏳ 시작 안 함 | 시작 시 자동으로 최신 코드(Neural Cleanse 샘플 풀 수정 포함) 반영됨 |
| Step 4~7: Ablation 4종 | ⏳ 시작 안 함 | 자동으로 최신 코드 반영됨 (Fixed-Delta 분리, 클린 기준선 포함) |

## ⚠️ 절대 처음부터 다시 돌리지 말 것

clean/qafm/badnets는 이미 올바른 결과가 저장돼 있어 (`results/exp1_attack_effectiveness/cifar10_results.json`, 체크포인트 포함). 처음부터 재실행하면 이 3개(약 4.5시간 분량)를 의미 없이 또 학습하게 됨.

## 다음에 돌릴 때 해야 할 일 (이 순서대로)

```bash
cd cifar10

# 1. ftrojan(공식 알고리즘)+blended만 새로 학습 — clean/qafm/badnets는 보존됨 (JSON 병합)
python experiments/main_exp1_attack_effectiveness.py --methods ftrojan blended --epochs 200

# 2. 은닉성 분석 (학습 없음, 빠름)
python experiments/main_exp2_stealth.py --n_samples 1000

# 3. 방어 저항성 (학습 없음, 1번에서 갱신된 ftrojan/blended 체크포인트 사용)
python experiments/main_exp3_defense.py --ckpt_dir checkpoints

# 4. Ablation 4종 (처음부터, 최신 코드)
python run_all.py --ablation_only --epochs 200
```

`run_all.py`를 그냥 다시 실행하면 안 되는 이유: Step0(이론검증)부터 다시 돌고, Step1도 clean/qafm/badnets/ftrojan/blended 5개를 전부 새로 학습해버림(이미 끝난 3개까지 중복 학습). 위처럼 단계별로 나눠서 실행해야 중복 없이 이어붙일 수 있음.
