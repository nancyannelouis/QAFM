# Paper Assets (IEEE Access 초안 기준)

`260524_IEEE Access.docx` 초안의 Figure/Table 요구사항을 실측 데이터로 생성.
방어 저항성 항목(원래 Neural Cleanse/STRIP)은 메인실험4(Fine-Pruning/NAD/ShrinkPad)로
**기법명만 교체**, 표/그림 형식은 동일하게 유지.

## 재생성 방법

```bash
python paper_assets/generate_data_assets.py       # Table 1,3,4,5,6 / Figure 5,6,7,8,9 (실측 데이터)
python paper_assets/generate_theory_assets.py     # Table 2 / Figure 1,2,3,4 (이론/개념도)
python paper_assets/generate_ablation_assets.py   # Table 7,8,9,10 / Figure 10 (ablation — 현재 대부분 PENDING)
```

## 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| Table 1 (기존 공격 비교) | ✅ | 정성적 비교 |
| Table 2 (트리거 생존 분류) | ✅ | Lemma 1 Case A-D |
| Table 3 (실험 설정) | ✅ | |
| Table 4 (BA/ASR 종합) | ✅ | 3개 데이터셋 |
| Table 5 (은닉성) | ✅ | 3개 데이터셋, trigger-only 기준 |
| **Table 6 (방어 저항성)** | ✅ | **NC/STRIP → Fine-Pruning/NAD/ShrinkPad로 교체** |
| Table 7 (Component Ablation) | ⏳ PENDING | 미실행/구버전 — 재실행 필요 |
| Table 8 (k값 Ablation) | ⏳ PENDING | 미실행 |
| Table 9 (q_train Ablation) | ⏳ PENDING | 미실행 |
| Table 10 (Poison Rate Ablation) | ⏳ PENDING | cifar10에 0.01 1개 값만 존재 |
| Figure 1 (압축 전후 overview) | ✅ | 실제 이미지 + 실측 ASR |
| Figure 2 (JPEG 파이프라인) | ✅ | 개념도 |
| Figure 3 (트리거 생존 조건) | ✅ | r 분포 + 케이스 분류 |
| **Figure 4 (QAFM 파이프라인)** | ✅ | ⚠️ **실제 구현 기준 Y채널** (아래 참고) |
| Figure 5 (주파수 위치 히트맵) | ✅ | k·Q_ij 실값 |
| Figure 6 (ASR vs Q) | ✅ | 3개 데이터셋, 기존 산출물 통합 |
| Figure 7 (포이즌 이미지 비교) | ✅ | 3개 데이터셋, 새로 그림 |
| **Figure 8 (방어별 ASR 막대그래프)** | ✅ | **NC anomaly index → exp4 ASR로 교체** |
| **Figure 9 (방어 전/후 비교)** | ✅ | **STRIP entropy histogram → exp4 전/후 비교로 교체** |
| Figure 10 (q_train ASR 그래프) | ⏳ PENDING | q_train ablation 의존 |

## QAFM 파이프라인 (Figure 4 설명)

`utils/jpeg_utils.py::insert_dct_trigger`가 수행하는 5단계:

1. RGB → YCbCr 변환
2. Y(밝기) 채널을 8×8 블록으로 분할 후 2D DCT
3. 모든 블록의 (i,j) 위치에 Δ_ij = k·Q_tr[i,j] 삽입 (Q_tr = q_train 품질의 양자화 테이블)
4. IDCT → 블록 병합 → YCbCr → RGB
5. **q_train으로 JPEG 압축** (학습 데이터 생성 시에도 압축을 거침)

### Step5(학습 시 재압축)가 있는 이유

- **실배포 환경과 학습 분포를 맞추기 위해**: QAFM의 위협모델 자체가 "실제 서비스에 올라간 이미지는 거의 항상 JPEG로 재압축된다"는 전제임. 평가/배포 시 모델이 보는 이미지는 항상 "트리거 삽입 → 압축"을 거친 모습인데, 학습을 압축 없이 시키면 모델이 실제로는 절대 나타나지 않는 분포(압축 안 된 트리거)를 학습하게 됨. 실제 JPEG 압축은 트리거 좌표(i,j) 하나만이 아니라 블록 내 다른 모든 AC 계수, 색차 서브샘플링까지 한꺼번에 양자화하므로 압축 전/후 픽셀 분포가 꽤 다름.
- **정리2/정리3가 이 단계에 의존함**: Δ=k·Q_tr[i,j]를 넣고 같은 품질(q_train)로 압축하면 정리1(정수 이동 불변성: Q(C+k·Q)=Q(C)+k·Q)에 의해 방금 넣은 변조량이 자기 압축 단계에서 그대로 보존됨 — 그래서 학습 데이터는 "이미 q_train 양자화 격자 위에 정확히 놓인" 패턴이 됨. 정리3(r=k·Q_tr/Q_ev≥1이면 평가 단계 생존)이 의미를 가지려면 모델이 배운 게 이런 "이미 양자화된 패턴"이어야 함 — 압축 안 한 원본을 학습시키면 이 사슬이 끊김.
- **실증**: ablation에서 Step5만 제거한 `No-JPEG` 변형은 트리거 위치/정렬이 QAFM과 동일한데도 ASR@Q50이 100%→37%로 떨어짐 — Step5가 이론적 장식이 아니라 실질적 효과를 낸다는 증거.

## ⚠️ 문서 초안과 실제 구현이 다른 부분 (확인 필요)

문서 초안(3.C절, line 74)은 "Cb/Cr 색차 채널에 트리거 적용"이라고 설계 근거를 적었지만,
**실제 구현(`attacks/qafm.py`, `utils/jpeg_utils.py::insert_dct_trigger`)은 Y(밝기) 채널을
사용함.** Figure 4는 실제 구현(Y채널) 기준으로 그렸고, 문서 본문의 "Cb/Cr 선택 근거"
서술(인간 시각 시스템이 색차 변화에 둔감하다는 논리)은 Y채널에는 적용되지 않으므로
**직접 수정 필요** — 2026-06-21 대화에서 사용자 확인: "실제 구현 그대로 두고, 문서를
나중에 수정"하기로 합의함.

## Ablation을 마친 뒤 할 일

Table 7-10, Figure 10은 자동으로 비어있지 않게 채워지지 않음 — ablation 실행 후
`generate_ablation_assets.py`를 다시 실행하면 [PENDING] placeholder가 실제 표/그림으로
자동 교체됨 (각 [PENDING] 이미지에 정확한 실행 명령어가 적혀 있음).
