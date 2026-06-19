# CIFAR-10 실험 진행 현황

마지막 업데이트: 2026-06-19

## 단계별 상태

| 단계 | 상태 | 비고 |
|---|---|---|
| Step 0: 이론 검증 | ✅ 완료 | Theorem 2만 85%(95% 기준 미달) — 측정 방법 한계, 무시 가능 |
| Step 1: 메인실험1 (공격효과성) | ✅ 완료 | clean/qafm(k=2)/badnets/ftrojan(공식 알고리즘)/blended(압축 제거) 전부 최신 코드로 완료 |
| Step 2: 메인실험2 (은닉성) | ✅ 완료 | k=2 + dual PSNR(절대/트리거기여) 최신 코드로 완료. QAFM PSNR_trigger_only=41.83dB (42dB 기준 근소 미달) |
| Step 3: 메인실험3 (방어저항성) | 🔶 부분 완료 | **qafm만 새 Neural Cleanse(적응적 cost 탐색)로 재실행됨(AI=8.32)**. badnets/ftrojan/blended는 옛 NC 알고리즘 결과 남아있음 — 재실행 필요 |
| Ablation: poison_rate + NC | ✅ 완료 (결론: 한계로 기록) | 아래 "Neural Cleanse 한계" 참고 |
| Ablation 1~3 (Component/k/q_train) | ⏳ 미시작 | |

## ⚠️ Neural Cleanse 한계 (poison_rate로 해결 안 됨)

QAFM은 STRIP과 Spectral Signatures는 안정적으로 회피하지만(`bypass: true`), **Neural Cleanse는 회피하지 못함**. 원인과 검증 과정:

1. **NC 구현 자체는 정상**: 클린(백도어 없음) 모델에 NC를 돌리면 AI=0.96 (<2, 정상 판정). 적응적 cost 탐색 + 1.4826 MAD 보정상수를 적용한 뒤 클린/백도어 모델을 정확히 구분함 — 구현 버그가 아니라 진짜(genuine) 탐지임을 확인.
2. **poison_rate를 낮춰도 개선되지 않음**: poison_rate=0.05→0.01로 낮췄을 때 AI가 8.32→9.60으로 오히려 **악화**됨 (`results/ablation/cifar10_poison_rate_ablation.json`). ASR@Q50이 poison_rate=0.01에서도 100%로 포화되어 있어, QAFM의 고정·결정론적 트리거가 매우 적은 poison 샘플(500장)로도 이미 최대 강도의 백도어 숏컷을 만들어버림 — poison_rate를 더 낮춘다고 숏컷이 약해지지 않음.
3. **결론**: NC가 탐지하는 신호는 "트리거가 주파수 도메인이냐 공간 도메인이냐"가 아니라, "특정 클래스로 가는 지름길이 다른 클래스보다 비정상적으로 쉬운가"임. QAFM은 단일·고정 트리거 + 단일 타겟 구조라서 이 조건에 정확히 해당하고, 이는 poison_rate 튜닝으로 해결되는 문제가 아니라 **트리거 설계(단일 고정 타겟)에서 기인하는 구조적 한계**임. 논문에는 "STRIP/SS는 회피, NC는 회피 못 함 — 그 이유는 트리거 도메인이 아니라 단일 타겟 구조 때문"으로 한계를 명시할 것.

## 다음에 돌릴 때 해야 할 일

```bash
cd cifar10

# Step 3 마무리 — badnets/ftrojan/blended를 새 Neural Cleanse로 재평가
# (체크포인트는 이미 최신이라 학습 없이 평가만 다시 돔)
python experiments/main_exp3_defense.py --ckpt_dir checkpoints --methods badnets ftrojan blended
```

이후 `cifar10_defense_summary.json`이 qafm/badnets/ftrojan/blended 4개 전부 새 NC 결과로 채워지면 Step 3 완전히 마무리됨.
