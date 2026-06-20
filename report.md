# QAFM 메인실험 1·2·4 결과 보고

마지막 업데이트: 2026-06-21

## 1. 생성된 표/그림 매핑

`paper_assets/`에 생성된 산출물과 각 실험의 대응 관계.

| 파일 | 실험 | 내용 |
|---|---|---|
| `tables/table4_ba_asr_{ds}.png` | **메인실험 1** | 데이터셋별 BA(전체클래스)/ASR@Q100~50 종합 비교표 |
| `figures/figure6_asr_vs_q_{ds}.png` | **메인실험 1** | ASR vs JPEG Q값 꺾은선 그래프 (Q=100→50) |
| `tables/table5_stealth_{ds}.png` | **메인실험 2** | PSNR/SSIM/LPIPS 비교표 (절대 vs 트리거기여 두 버전) |
| `figures/figure7_visual_comparison_{ds}.png` | **메인실험 2** | Clean/Poisoned/Diff×10 시각 비교 (4개 방법) |
| `tables/table6_defense_resistance_{ds}.png` | **메인실험 4** | Fine-Pruning/NAD/ShrinkPad 우회 여부 종합표 |
| `figures/figure8_defense_bar_{ds}.png` | **메인실험 4** | 방어별 적용 후 ASR 막대그래프 (50% 우회 임계선) |
| `figures/figure9_defense_before_after_{ds}.png` | **메인실험 4** | 방법별 적용전→FP→NAD→SP ASR 변화 비교 |
| `tables/table1_baseline_comparison.png` | 배경(RELATED WORK) | 기존 공격 4종 정성적 비교 |
| `tables/table2_survival_classification.png` | 이론(3.B) | Lemma 1 케이스 A-D 분류표 |
| `tables/table3_experiment_setup.png` | 실험설정 | 하이퍼파라미터 요약 |
| `figures/figure1~5` | 이론/개념도 | 압축 전후 비교, JPEG 파이프라인, 생존조건, QAFM 파이프라인, 주파수 히트맵 |
| `tables/table7~10`, `figures/figure10` | Ablation | **[PENDING]** — 아래 4절 참고 |

(`{ds}` = cifar10, cifar100, gtsrb)

---

## 2. 메인실험 1: 공격 효과성

`table4_ba_asr_{ds}.png`, `figure6_asr_vs_q_{ds}.png` 참고.

| | QAFM | BadNets | FTrojan | Blended |
|---|---|---|---|---|
| CIFAR-10 | BA 94.72% / ASR@Q50 **100%** | BA 93.89% / 58.2% | BA 94.95% / 1.5% | BA 94.92% / 1.8% |
| CIFAR-100 | BA 77.15% / ASR@Q50 **100%** | BA 75.16% / 64.6% | BA 76.72% / 0.2% | BA 77.46% / 0.2% |
| GTSRB | BA 96.76% / ASR@Q50 **100%** | BA 96.10% / 86.4% | BA 96.98% / 0.0% | BA 96.83% / 0.1% |

QAFM만 3개 데이터셋 전부 최악 압축(Q=50)에서도 ASR 100% 유지. FTrojan/Blended는 Q90 부근에서 절벽처럼 무너짐(고정 델타가 양자화 테이블과 정렬되지 않아 강한 압축에서 반올림 과정에 트리거가 소멸).

---

## 3. 메인실험 2: 은닉성

`table5_stealth_{ds}.png`, `figure7_visual_comparison_{ds}.png` 참고. (트리거 순수 기여도 기준 — Step5 JPEG 재압축 손실 제외)

| | PSNR | SSIM | 임계값 |
|---|---|---|---|
| CIFAR-10 | 41.84dB | 0.9958 | 42dB / 0.99 |
| CIFAR-100 | 41.81dB | 0.9958 | 근소 미달 |
| GTSRB | **42.12dB (통과)** | 0.9912 | |

---

## 4. 메인실험 4: 방어 저항성 (Fine-Pruning / NAD / ShrinkPad)

`table6_defense_resistance_{ds}.png`, `figure8_defense_bar_{ds}.png`, `figure9_defense_before_after_{ds}.png` 참고.

- **QAFM**: 3개 데이터셋 전부 3개 방어 모두 우회
- **BadNets**: CIFAR-10/100에서는 3개 다 무너짐, GTSRB에서는 2/3(Fine-Pruning·NAD) 우회하지만 ShrinkPad엔 무너짐
- **FTrojan/Blended**: `already_failed_pre_defense=true` — 아래 4.1 참고

### 4.1 ⚠️ FTrojan/Blended는 "방어가 이긴 것"이 아니라 "압축에서 이미 죽은 것"

`table6`/`figure8`/`figure9`에서 FTrojan/Blended의 ASR이 방어 적용 전부터(`ASR_before`) 이미 0~2% 수준인 게 보임. 이건 메인실험4가 `eval_q=75`로 평가하기 때문 — 메인실험1에서 이미 확인했듯 FTrojan/Blended는 Q=75에서 ASR이 1%대로 무너진 상태라, Fine-Pruning/NAD/ShrinkPad를 적용하기도 전에 공격이 이미 실패해 있음. 즉 이 두 방법의 "방어 우회 실패"는 **방어 기법의 효과를 보여주는 게 아니라, 그보다 앞서 일어난 압축 손실을 한 번 더 확인한 것**에 불과함. 그래서 메인실험4의 깨끗한 비교는 방어 적용 전 ASR이 충분히 높은 **QAFM vs BadNets**뿐이고, FTrojan/Blended는 비교 대상에서 제외해야 함(이미 `already_failed_pre_defense` 플래그로 표시해둠).

### 4.2 고민: 메인실험4를 압축 이전 ASR 기준으로 다시 봐야 하는가?

지금처럼 `eval_q=75`(학습 Q와 동일) 기준으로 방어를 적용하면, FTrojan/Blended처럼 압축 자체에 취약한 방법은 방어 효과를 측정할 기회조차 없이 탈락함. 대안으로 **압축을 적용하지 않은(또는 Q=100) ASR을 "방어 적용 전" 기준으로 잡으면** 4개 방법 모두 비슷하게 높은 시작점(ASR≈100%)에서 출발하게 되어, Fine-Pruning/NAD/ShrinkPad 자체의 효과만 순수하게 비교할 수 있음.

다만 이렇게 바꾸면:
- 위협모델상 "실세계 이미지는 거의 항상 압축을 거친다"는 QAFM 논문의 핵심 전제와 다소 어긋남 (압축 없는 ASR을 기준으로 삼는 게 비현실적일 수 있음)
- 반대로 지금 방식(압축 후 ASR 기준)은 "압축"과 "방어"라는 두 가지 다른 효과가 한 수치에 섞여 들어가, 방어 자체의 영향력을 분리해서 보기 어려움

**결론을 못 내리고 있는 부분** — 둘 중 어느 기준이 논문의 주장(QAFM은 압축에도 강건하고, 방어에도 강건하다)을 더 정확하게/공정하게 보여주는지 판단이 필요함. 가능하면 두 기준 다 측정해서 같이 보여주는 절충안도 검토 가능.

---

## 5. 메인실험3(NC/STRIP/SS) → 메인실험4 교체 이유

### 5.1 메인실험3가 사실은 "표준" 실험 세팅이라는 점

Neural Cleanse·STRIP·Spectral Signatures는 BadNets/FTrojan/Blended를 포함해 이 분야 거의 모든 선행 연구가 공통으로 사용하는 **사실상의 표준 방어 평가 세트**임. Research Proposal에도 이 세 가지를 메인실험3로 명시했고, 처음 계획도 이 표준을 따르는 것이었음.

### 5.2 그런데 왜 적용이 어려웠는가

세 방어 모두 메커니즘이 "**단일 고정 트리거 + 단일 타겟 클래스**" 구조를 전제로 함 — 트리거가 주파수 도메인이든 공간 도메인이든 무관하게, 이 구조 자체를 탐지함. 검증 과정:
- NC 구현 자체는 정상임을 클린 모델로 확인(AI=0.96, 정상 판정)
- poison_rate를 5%→1%로 낮춰도 AI가 8.32→9.60으로 오히려 악화 — QAFM의 고정·결정론적 트리거가 매우 적은 샘플로도 최대 강도 숏컷을 만들기 때문
- 즉 QAFM이 NC/STRIP/SS를 회피 못 하는 건 **주파수 도메인 설계와 무관한, 트리거가 단일·고정·단일타겟이라는 구조에서 오는 한계**이고, 이건 poison_rate나 다른 하이퍼파라미터로 해결되는 문제가 아님

### 5.3 고민: 그래도 표준 실험이니 강행해야 하는가

이 분야 리뷰어들이 "왜 가장 표준적인 방어 기법 결과가 없냐"고 의심할 수 있다는 우려가 있음. 그래서 두 가지 선택지를 두고 고민 중:

1. **메인실험3를 그대로 강행해서 정직하게 보고**: "QAFM은 NC/STRIP/SS를 회피하지 못하며, 그 이유는 트리거 도메인이 아니라 단일 타겟 구조 자체"라고 한계로 명시. 표준 실험을 누락했다는 의심은 피할 수 있지만, "방어 회피"라는 주장이 약해짐.
2. **메인실험4로 교체해서 메커니즘이 다른 방어로 강건성을 보임**: QAFM의 실제 기여(압축 강건성, 분산된 트리거)가 의미 있는 차이를 만드는 방어(Fine-Pruning/NAD/ShrinkPad)에 집중. 다만 "표준 실험을 왜 안 했냐"는 질문에 대응할 설명이 필요함.

**현재는 4번 결과를 메인으로 놓고, 3번 결과(이미 갖고 있는 수치)는 한계/부록으로 같이 보여주는 절충안 쪼으로 기울어 있으나, 최종 결정은 아직 안 함.**

---

## 6. Ablation 진행 상황

`cifar10/experiments/ablation/component_ablation.py` 실행 중 (시작 03:30:48).

| Ablation | 변수 개수 | 변수당 예상 시간 | 총 예상 시간(cifar10) |
|---|---|---|---|
| Component | 3 (QAFM/Fixed-Delta/No-JPEG) | ~90분 | ~4.5시간 |
| k값 | 5 (1~5) | ~90분 | ~7.5시간 |
| q_train | 6 (50,60,70,75,85,95) | ~90분 | ~9시간 |
| Poison rate | 6 (0~0.2) | ~90분 | ~9시간 |

cifar10 전체 ablation 4종 완료까지 **약 30시간** 예상, cifar100/gtsrb까지 동일하게 진행하면 약 3배(전체 ~90시간) 소요 예상. 현재는 cifar10 component ablation만 순차 실행 중이며 중간 결과 없음(첫 variant 진행 중).
