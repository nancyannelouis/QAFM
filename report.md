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

### 4.0 왜 메인실험3(NC/STRIP/SS) 대신 이 세 가지를 선택했는가

메인실험3의 NC/STRIP/SS는 모두 "**단일 고정 트리거 + 단일 타겟 클래스**" 구조 자체를 탐지하는 방어라서, 트리거가 주파수 도메인이든 공간 도메인이든 무관하게 같은 결과(탐지됨)가 나옴 — 즉 QAFM의 설계 기여(압축 강건성, 8×8 블록 전체에 분산된 트리거)를 변별하지 못함(5절 참고). 그래서 메인실험4는 질문을 바꿔서 던짐: "구조적으로 탐지 가능하다는 건 이미 알았으니, **일단 의심된 모델/입력을 사후에 복구·교란하는 방어가 실제로 트리거를 제거할 수 있는가**?"를 봄. 이 질문에 답하려면 개입 방식이 서로 다른 방어가 필요해서, 메커니즘이 겹치지 않는 3가지를 골랐음:

| 방어 | 개입 대상 | 재학습 필요 | 핵심 가정 |
|---|---|---|---|
| Fine-Pruning | 모델 내부(채널) | O (파인튜닝) | 트리거가 일부 "휴면" 채널에 집중돼 있다 |
| NAD | 모델 내부(표현) | O (증류) | 트리거가 어텐션 맵에 국소적 핫스팟을 만든다 |
| ShrinkPad | 입력 전처리만 | X | 트리거가 정확한 픽셀 좌표에 의존한다 |

세 가정 모두 "트리거가 모델/이미지의 일부 좁은 영역에 집중돼 있다"는 전제를 공유함. QAFM은 이 전제와 정반대로 설계됐기 때문에(모든 8×8 블록에 동일하게 분산), 이 실험이 QAFM 고유의 설계가 만드는 차이를 가장 직접적으로 보여줄 수 있다고 판단해서 선택함.

### 4.1 방어 기법별 개요·핵심 메커니즘·QAFM의 우회 방식

#### (1) Fine-Pruning — Liu, Dolan-Gavitt & Garg, *RAID 2018* [R1]

- **메커니즘**: 클린 데이터를 모델에 통과시켜 지정 레이어(여기서는 ResNet `layer2`)의 채널별 평균 활성값을 구하고, 활성값이 가장 낮은(=클린 입력에 거의 반응 안 하는 "휴면" 채널이 트리거 전용 통로일 것이라는 가정) `prune_rate`(=20%) 채널을 0으로 마스킹한 뒤, 클린 데이터로 전체 모델을 파인튜닝(10 epoch)해 깎인 정확도를 복구함.
- **QAFM의 우회 방식**: 가지치기 직후만 보면 CIFAR-10/100에서는 실제로 ASR이 크게 꺾임(CIFAR-10 99.94%→**5.08%**, CIFAR-100 99.97%→**3.53%**) — 즉 QAFM도 일부 저활성 채널에 신호가 걸려 있는 건 맞음. 그런데 그 다음 파인튜닝 단계에서 ASR이 다시 **97.52%, 98.93%**로 거의 완전히 복귀함. 이는 파인튜닝에 쓰는 클린 데이터에는 트리거가 전혀 없어서, "트리거를 만났을 때 다르게 반응하라"고 교정해줄 신호 자체가 손실 함수에 없기 때문 — QAFM 신호가 모든 블록에 중복으로 박혀 있어서 남은 80%의 채널만으로도 파인튜닝 과정에서 그 패턴을 다시 포착하는 경로가 재형성됨. GTSRB는 가지치기 단계에서부터 거의 영향이 없음(98.66%→99.88%) — 중복도가 더 높다는 뜻.
- **대조(BadNets)**: 고정 위치의 작은 패치 트리거는 소수 채널에 집중되는 경향이 있어 가지치기+파인튜닝으로 영구히 억제됨 (CIFAR-10 85.53%→**4.42%**, CIFAR-100 81.31%→**0.18%**, bypass 모두 false).

#### (2) NAD (Neural Attention Distillation) — Li, Lyu, Koren, Lyu, Li & Ma, *ICLR 2021* [R2]

- **메커니즘**: 백도어 모델을 클린 데이터로 살짝 파인튜닝해 "teacher"를 만들고(트리거에 약하게라도 덜 반응하는 상태), 이 teacher를 고정한 뒤 원본(student) 모델을 `CE + Σ β·AT(student, teacher)`로 재학습. AT(attention transfer)는 지정 레이어(layer2/3/4, β=500 공식 기본값)의 어텐션 맵(채널 축 L2 제곱합 후 정규화) 간 MSE — student의 어텐션을 teacher 쪽으로 끌어당김.
- **QAFM의 우회 방식**: 이 방어는 트리거가 어텐션 맵에 "국소적으로 튀는 핫스팟"을 만든다는 전제에 의존함. QAFM의 변형은 이미지 전체 8×8 블록에 동일하게 퍼져 있어서 어텐션 맵 상에 그런 국소 핫스팟이 생기지 않고, teacher와 student의 어텐션 분포 자체가 트리거 유무와 무관하게 비슷함 — 증류가 "지워야 할 대상"을 찾지 못함. 결과: CIFAR-10 99.94%→**60.29%**, CIFAR-100 99.97%→**93.33%**, GTSRB 99.99%→**99.79%**(전부 50% 우회 임계선 위).
- **대조(BadNets)**: CIFAR-10/100에서는 패치가 만드는 핫스팟이 또렷해서 거의 완전히 제거됨(1.58%, 1.22%) — 단 GTSRB에서는 BadNets도 NAD를 우회(73.08%)해, 핫스팟 가정이 데이터셋에 따라 흔들릴 수 있음을 보여줌(이 경우는 ShrinkPad가 대신 BadNets를 잡아냄, 0.16%).

#### (3) ShrinkPad — Li, Li, Wu, Li, He & Lyu, *"Backdoor Attack in the Physical World"*, ICLR 2021 Workshop [R3]

- **메커니즘**: 재학습 없는 입력 전처리 방어. 32×32 입력을 (32−pad)×(32−pad)=28×28로 축소한 뒤, 가능한 모든 패딩 배치 중 하나를 무작위로 골라 다시 32×32로 패딩 — 트리거가 학습 시 의존한 "정확한 픽셀 좌표"를 추론 시점에 깨뜨리는 것이 목표.
- **QAFM의 우회 방식**: 세 방어 중 QAFM에 가장 큰 손상을 줌(CIFAR-10 99.94%→**53.46%**, CIFAR-100 99.97%→**62.78%**, GTSRB 99.99%→**78.8%**) — 리사이즈가 8×8 블록 정렬 자체를 흔들기 때문에 메커니즘상 가장 QAFM을 직접 겨냥하는 방어이지만, 원본 이미지의 모든 블록에 동일한 주파수 변화가 중복으로 들어가 있어서 리사이즈+보간 후에도 충분히 많은 블록이 일관되게 변형된 주파수 신호를 유지함 → 50% 선은 넘기지 못하지만 임계선을 살짝 넘는 정도로 간신히 우회.
- **대조(BadNets)**: 정확히 ShrinkPad가 겨냥하는 "고정 위치 단일 패치"라서 가장 확실하게 무너짐(CIFAR-10 2.98%, CIFAR-100 5.27%, GTSRB 0.16%).

### 4.2 결과 요약

- **QAFM**: 3개 데이터셋 전부 3개 방어 모두 우회
- **BadNets**: CIFAR-10/100에서는 3개 다 무너짐, GTSRB에서는 2/3(Fine-Pruning·NAD) 우회하지만 ShrinkPad엔 무너짐
- **FTrojan/Blended**: `already_failed_pre_defense=true` — 아래 4.3 참고

### 4.3 ⚠️ FTrojan/Blended는 "방어가 이긴 것"이 아니라 "압축에서 이미 죽은 것"

`table6`/`figure8`/`figure9`에서 FTrojan/Blended의 ASR이 방어 적용 전부터(`ASR_before`) 이미 0~2% 수준인 게 보임. 이건 메인실험4가 `eval_q=75`로 평가하기 때문 — 메인실험1에서 이미 확인했듯 FTrojan/Blended는 Q=75에서 ASR이 1%대로 무너진 상태라, Fine-Pruning/NAD/ShrinkPad를 적용하기도 전에 공격이 이미 실패해 있음. 즉 이 두 방법의 "방어 우회 실패"는 **방어 기법의 효과를 보여주는 게 아니라, 그보다 앞서 일어난 압축 손실을 한 번 더 확인한 것**에 불과함. 그래서 메인실험4의 깨끗한 비교는 방어 적용 전 ASR이 충분히 높은 **QAFM vs BadNets**뿐이고, FTrojan/Blended는 비교 대상에서 제외해야 함(이미 `already_failed_pre_defense` 플래그로 표시해둠).

### 4.4 고민: 메인실험4를 압축 이전 ASR 기준으로 다시 봐야 하는가?

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

---

## 참고문헌 (메인실험4 방어 기법)

- **[R1]** Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg. *"Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks."* RAID 2018.
- **[R2]** Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, Xingjun Ma. *"Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks."* ICLR 2021.
- **[R3]** Yuezun Li, Yiming Li, Baoyuan Wu, Longkang Li, Ran He, Siwei Lyu. *"Backdoor Attack in the Physical World."* ICLR 2021 Workshop (RobustML).

구현은 `defenses/fine_pruning.py`, `defenses/nad.py`, `defenses/shrinkpad.py`의 docstring에 명시된 대로, 공식 알고리즘 충실도를 위해 [THUYimingLi/BackdoorBox] 라이브러리(Fine-Pruning, ShrinkPad)와 NAD 원저자 공식 repo([bboylyg/NAD], R2 구현 기준)를 참고해 포팅함. 하이퍼파라미터(Fine-Pruning `prune_rate=0.2`/`layer2`, NAD `β=[500,500,500]`/`layer2,3,4`, ShrinkPad `pad=4`)는 각 공식 구현의 기본값을 그대로 사용.
