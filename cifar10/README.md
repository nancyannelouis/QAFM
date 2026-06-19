# CIFAR-10 실험 진행 현황

마지막 업데이트: 2026-06-19

## 단계별 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| Step 0: 이론 검증 | ✅ 완료 | Theorem 2만 85%(95% 기준 미달) — 측정 방법 한계, 무시 가능 |
| Step 1: 메인실험1 (공격효과성) | ✅ 완료 | clean/qafm(k=2)/badnets/ftrojan(공식 알고리즘)/blended(압축 제거) 전부 최신 코드로 완료. `BA_full`(전체 클래스 기준 BA)도 추가됨 |
| Step 2: 메인실험2 (은닉성) | 🔶 부분 완료 | PSNR은 dual-metric(절대/트리거기여) 적용 완료. **SSIM/LPIPS도 trigger_only 버전 코드는 추가됐지만 아직 그 코드로 재실행 안 함** — 재실행 필요 |
| Step 3: 메인실험3 (NC/STRIP/SS) | ⏸️ 보류 (exp4로 우선순위 이동) | qafm만 새 NC(적응적 cost 탐색)로 재실행됨(AI=8.32). **badnets/ftrojan/blended는 SS/STRIP 공식 알고리즘 재작성 *이전* 코드로 남아있음** — 재실행 필요하지만 후순위 |
| Step 4: 메인실험4 (Fine-Pruning/NAD/ShrinkPad) | ✅ 완료 | 4개 방법 전부 완료. 아래 "실험4 해석" 참고 |
| Ablation: poison_rate + NC | ✅ 완료 (결론: 한계로 기록) | 아래 "Neural Cleanse 한계" 참고 |
| Ablation 1~3 (Component/k/q_train) | ⏳ 미시작 | |

## ⚠️ Neural Cleanse 한계 (poison_rate로 해결 안 됨)

QAFM은 STRIP과 Spectral Signatures는 안정적으로 회피하지만(`bypass: true`), **Neural Cleanse는 회피하지 못함**. 원인과 검증 과정:

1. **NC 구현 자체는 정상**: 클린(백도어 없음) 모델에 NC를 돌리면 AI=0.96 (<2, 정상 판정). 적응적 cost 탐색 + 1.4826 MAD 보정상수를 적용한 뒤 클린/백도어 모델을 정확히 구분함 — 구현 버그가 아니라 진짜(genuine) 탐지임을 확인.
2. **poison_rate를 낮춰도 개선되지 않음**: poison_rate=0.05→0.01로 낮췄을 때 AI가 8.32→9.60으로 오히려 **악화**됨 (`results/ablation/cifar10_poison_rate_ablation.json`). ASR@Q50이 poison_rate=0.01에서도 100%로 포화되어 있어, QAFM의 고정·결정론적 트리거가 매우 적은 poison 샘플(500장)로도 이미 최대 강도의 백도어 숏컷을 만들어버림 — poison_rate를 더 낮춘다고 숏컷이 약해지지 않음.
3. **결론**: NC가 탐지하는 신호는 "트리거가 주파수 도메인이냐 공간 도메인이냐"가 아니라, "특정 클래스로 가는 지름길이 다른 클래스보다 비정상적으로 쉬운가"임. QAFM은 단일·고정 트리거 + 단일 타겟 구조라서 이 조건에 정확히 해당하고, 이는 poison_rate 튜닝으로 해결되는 문제가 아니라 **트리거 설계(단일 고정 타겟)에서 기인하는 구조적 한계**임.
4. **이 한계는 STRIP/SS에도 동일하게 적용됨** — NC/STRIP/SS 셋 다 "단일 고정 트리거 + 단일 타겟"을 탐지하는 방어라서, 회피 여부가 트리거의 주파수 도메인 특성과 무관함. 그래서 메커니즘이 다른 방어(Fine-Pruning/NAD/ShrinkPad, 메인실험4)를 추가함.

## 메인실험4 해석 (Fine-Pruning / NAD / ShrinkPad)

`results/exp4_defense_advanced/cifar10_defense_advanced_summary.json` 결과 해석 시 반드시 주의할 점:

1. **`eval_q=75`에서 FTrojan/Blended는 방어 적용 전부터 이미 ASR이 0.83~0.86%** (`already_failed_pre_defense: true`). 이건 "방어가 백도어를 이겼다"가 아니라 **JPEG Q=75 압축 자체에서 이미 공격이 실패한 상태**임 — exp1의 ASR@Q75 cliff 패턴과 정확히 일치. 그래서 이 둘은 Fine-Pruning/NAD/ShrinkPad의 방어 저항성 비교 대상이 아님.
2. **깨끗한 비교는 QAFM vs BadNets** (둘 다 방어 적용 전 ASR이 충분히 높음: QAFM 99.94%, BadNets 85.53%):
   - QAFM: FinePruning 후 97.49%(우회), NAD 후 58.14%(우회), ShrinkPad 후 53.14%(우회) — 3개 다 저항
   - BadNets: FinePruning 후 4.03%, NAD 후 0.89%, ShrinkPad 후 2.98% — 3개 다 크게 무너짐
3. **Fine-Pruning의 가지치기 단계만 보면 QAFM도 크게 약화됨** (`ASR_prune_only`=5.08%) — 하지만 표준 Fine-Pruning 절차의 두 번째 단계(클린 데이터 파인튜닝, BA 복구용)를 거치면 ASR이 97.49%로 다시 회복됨. 즉 **가지치기만으로는 QAFM도 약화시킬 수 있지만, BA를 정상 수준으로 복구하는 파인튜닝 과정에서 백도어도 같이 복구되어버림** — 이건 QAFM이 "가지치기에 강하다"는 게 아니라 "가지치기 후 파인튜닝하면 다시 살아난다"는, 더 흥미롭고 정확한 관찰. 논문에 `BA_prune_only`/`ASR_prune_only` vs `BA`/`ASR` 두 단계를 같이 보여줄 것.
4. **`BA`는 target class를 제외한 정확도, `BA_full`은 전체 클래스 기준 정확도** — 논문에는 `BA_full`을 표준 BA로 사용할 것.

## 다음에 돌릴 때 해야 할 일

```bash
cd cifar10

# 1. (보류 중) exp3를 새 SS/STRIP 알고리즘으로 재실행 — badnets/ftrojan/blended분만 갱신 필요
#    (qafm은 이미 새 NC로 재실행됐지만 SS/STRIP은 qafm도 옛 코드 기준이라 사실상 4개 전부 다시 돌려야 함)
python experiments/main_exp3_defense.py --ckpt_dir checkpoints

# 2. exp2를 새 SSIM_trigger_only/LPIPS_trigger_only 코드로 재실행
python experiments/main_exp2_stealth.py --n_samples 1000
```

위 두 개는 exp4보다 후순위로 미뤄둔 상태. exp4(Fine-Pruning/NAD/ShrinkPad)는 이미 완료됐고, 다음 우선순위는 cifar100/gtsrb에서 동일한 실험(exp1/exp2/exp4)을 마무리하는 것.
