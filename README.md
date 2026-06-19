# QAFM: Quantization-Aligned Frequency Manipulation

**JPEG 압축에 강건한 이미지 백도어 삽입 기법 (IEEE Access 확장)**
이성은 | 지도교수: 서정택 교수님 | 가천대학교

---

## 개요

딥러닝 이미지 분류 모델에 삽입되는 백도어 트리거를 JPEG 양자화 테이블과 수학적으로 정렬하여,
압축 후에도 트리거가 소멸되지 않음을 증명하고 실험으로 검증하는 연구.

**핵심 수학 (Theorem 1)**:
```
Δ = k · Q_ij  →  Q(C + k·Q) = Q(C) + k·Q
```
k가 정수이기만 하면 f값에 무관하게 트리거 성분이 압축 과정에서 보존됨.

---

## 파일 구조

데이터셋(CIFAR-10 / CIFAR-100 / GTSRB)별로 최상위 디렉토리가 분리되어 있으며,
각 디렉토리는 자체 `config.py` / `data/` / `results/` / `checkpoints/` 를 가진
독립적인 실험 단위입니다. 공격·모델·방어·유틸리티 구현체는 데이터셋에 무관하므로
루트에서 공용으로 사용합니다.

```
QAFM/
│
├── utils/
│   ├── jpeg_utils.py                  # JPEG 양자화 연산자 / 트리거 삽입 알고리즘
│   ├── metrics.py                     # BA / ASR / PSNR / SSIM / LPIPS 계산
│   └── visualization.py              # 실험 결과 시각화 (matplotlib Agg 백엔드)
│
├── models/
│   ├── resnet18.py                    # ResNet-18 (CIFAR-10 / CIFAR-100)
│   └── preact_resnet18.py            # PreAct-ResNet-18 (GTSRB)
│
├── attacks/
│   ├── qafm.py                        # [제안 기법] QAFM 트리거 삽입
│   ├── badnets.py                     # [기준선] BadNets (공간 도메인 패치)
│   ├── ftrojan.py                     # [기준선] FTrojan (고정 델타 주파수)
│   └── blended.py                     # [기준선] Blended (혼합 방식)
│
├── defenses/
│   ├── neural_cleanse.py              # Neural Cleanse (IEEE S&P 2019)
│   ├── strip.py                       # STRIP (ACSAC 2019)
│   ├── spectral_signatures.py        # Spectral Signatures (NeurIPS 2018)
│   ├── fine_pruning.py               # Fine-Pruning (RAID 2018)
│   ├── nad.py                        # NAD: Neural Attention Distillation (ICLR 2021)
│   └── shrinkpad.py                  # ShrinkPad (ICLR Workshop 2021)
│
├── verify_theory.py                   # 수학적 이론 수치 검증 (Theorem 1~3, Lemma 1~2, 데이터셋 공용)
│
├── cifar10/                            # ── CIFAR-10 실험 단위 (독립 실행) ──
│   ├── config.py                       # NUM_CLASSES=10, mean/std, backbone=resnet18 등
│   ├── dataset.py                      # CIFAR-10 로더 + Poisoned/Eval 데이터셋
│   ├── train.py                        # 단독 학습 스크립트
│   ├── evaluate.py                     # 단독 평가 스크립트
│   ├── run_all.py                      # 이 데이터셋의 전체 실험 순차 실행
│   ├── experiments/
│   │   ├── main_exp1_attack_effectiveness.py
│   │   ├── main_exp2_stealth.py
│   │   ├── main_exp3_defense.py             # Neural Cleanse / STRIP / Spectral Signatures
│   │   ├── main_exp4_defense_advanced.py    # Fine-Pruning / NAD / ShrinkPad
│   │   └── ablation/
│   │       ├── component_ablation.py
│   │       ├── k_value_ablation.py
│   │       ├── q_train_ablation.py
│   │       └── poison_rate_ablation.py
│   ├── data/                           # (런타임 생성) 다운로드된 CIFAR-10
│   ├── results/                        # (런타임 생성) 실험 결과
│   └── checkpoints/                    # (런타임 생성) 학습된 모델
│
├── cifar100/                           # ── CIFAR-100 실험 단위 (cifar10/과 동일 구조) ──
│   └── ... (NUM_CLASSES=100, backbone=resnet18)
│
├── gtsrb/                               # ── GTSRB 실험 단위 (cifar10/과 동일 구조) ──
│   └── ... (NUM_CLASSES=43, backbone=preact_resnet18)
│
└── requirements.txt                    # 패키지 의존성
```

> 세 데이터셋 폴더는 서로 독립적으로 설정·실행됩니다. 예를 들어 `gtsrb/config.py`의
> `ABLATION_K_VALUES`만 바꿔도 다른 데이터셋에는 영향이 없습니다.

---

## 환경 설정

### 요구 사항

- Python 3.9 이상
- CUDA 가능한 NVIDIA GPU (논문 실험 환경: RTX 3060)
- Windows 10/11 또는 macOS / Linux

### 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 주요 패키지:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
Pillow>=9.5.0
tqdm>=4.65.0
lpips>=0.1.4
```

> **Windows 주의**: `lpips`가 설치 안 될 경우 `pip install lpips` 재시도 또는 생략 가능.
> LPIPS 없이도 PSNR/SSIM만으로 은닉성 평가 동작.

---

## 실행 방법

### 0. 수학적 이론 검증 (학습 불필요, 즉시 실행 가능)

```bash
python verify_theory.py
```

Theorem 1·2·3, Lemma 1·2, Proposition 1을 각 10만 회 수치 검증.
데이터셋 다운로드 없이 실행 가능.

---

### 1. 전체 실험 한 번에 실행

데이터셋 폴더로 이동한 뒤 `run_all.py`를 실행합니다 (더 이상 `--dataset` 플래그가 없습니다).

```bash
# CIFAR-10, 200 에폭
cd cifar10 && python run_all.py --epochs 200

# GTSRB
cd gtsrb && python run_all.py --epochs 200

# 이론 검증만 (데이터셋 무관)
python verify_theory.py

# 은닉성 분석만 (학습 없이)
cd cifar10 && python run_all.py --stealth_only
```

---

### 2. 개별 실험 실행

아래 명령은 모두 해당 데이터셋 폴더(`cifar10/`, `cifar100/`, `gtsrb/`) 안에서 실행합니다.

#### 학습 (단독)
```bash
cd cifar10
python train.py --method qafm
python train.py --method badnets
python train.py --method ftrojan
python train.py --method blended
python train.py --method clean   # 클린 모델 (기준선)
```

#### 평가 (단독)
```bash
python evaluate.py --method qafm --ckpt checkpoints/cifar10_qafm.pth
```

#### 메인 실험 1: 공격 효과성
```bash
python experiments/main_exp1_attack_effectiveness.py
```

출력: `results/exp1_attack_effectiveness/cifar10_results.json` + 그래프

#### 메인 실험 2: 은닉성 분석
```bash
python experiments/main_exp2_stealth.py --n_samples 1000
```

출력: `results/exp2_stealth/cifar10_stealth_results.json`

#### 메인 실험 3: 방어 저항성 (Neural Cleanse / STRIP / Spectral Signatures)
```bash
python experiments/main_exp3_defense.py --ckpt_dir checkpoints
```

출력: `results/exp3_defense/cifar10_defense_summary.json`

> 이 세 방어는 전부 "단일 고정 트리거 + 단일 타겟 클래스" 구조를 탐지하는 방어라서,
> 트리거가 주파수 도메인이냐 공간 도메인이냐와 무관하게 동작함 (자세한 근거는
> `cifar10/README.md`의 "Neural Cleanse 한계" 절 참고). 그래서 메커니즘이 다른
> 아래 실험 4도 같이 봐야 함.

#### 메인 실험 4: 방어 저항성 (Fine-Pruning / NAD / ShrinkPad)
```bash
python experiments/main_exp4_defense_advanced.py --ckpt_dir checkpoints
```

출력: `results/exp4_defense_advanced/cifar10_defense_advanced_summary.json`

모델 복구(클린 데이터로 가지치기·파인튜닝, attention distillation)와 입력 전처리
(축소+패딩) 기반 방어 3종. exp3와 달리 "단일 타겟이 유독 쉬운가"가 아니라 "클린
데이터로 모델을 복구/입력을 교란해도 트리거가 살아남는가"를 봄.

---

### 3. Ablation Study

데이터셋 폴더 안에서 실행 (예: `cifar10/`):

```bash
# Ablation 1: Component (QAFM vs Fixed-Delta vs No-JPEG)
python experiments/ablation/component_ablation.py

# Ablation 2: k값 변화 (k=1~5, 강건성-은닉성 트레이드오프)
python experiments/ablation/k_value_ablation.py

# Ablation 3: 학습 Q값 변화 (q_train=50,60,70,75,85,95)
python experiments/ablation/q_train_ablation.py

# Ablation 4: Poison Rate 변화 (1%~20%)
python experiments/ablation/poison_rate_ablation.py
```

다른 데이터셋(`cifar100/`, `gtsrb/`)도 동일한 명령을 해당 폴더에서 그대로 사용합니다.

---

## 평가 Q값 범위

전 실험에서 아래 11개 Q값을 사용:

| Q값 | 의미 |
|-----|------|
| 100 | 거의 무손실 (스텝=1) |
| 95  | 매우 높은 품질 |
| 90  | 높은 품질 |
| 85  | 표준 고품질 |
| 80  | 일반 고품질 |
| **75** | **학습 Q값 (q_train)** |
| 70  | 보통 품질 |
| 65  | 보통 이하 |
| 60  | 낮은 품질 |
| 55  | 낮은 품질 |
| **50** | **libjpeg 스케일 기준점 (최악 케이스)** |

> Proposition 1: k=2(=k_min), q_train=75 기준으로 Q∈[50,100] 전 범위에서 r≥1 이론 보장.
> (k=3은 k_min을 만족하지만 은닉성 측면에서 불필요하게 큰 값이라 k=2로 변경됨 — 자세한 근거는 `cifar10/README.md` 참고.)

---

## 실험 결과 위치

이론 검증 결과는 루트 `results/`에, 그 외 모든 실험 결과는 각 데이터셋 폴더 안의
`results/`에 저장됩니다 (예: `cifar10/results/...`).

```
results/                                # 루트 — verify_theory.py 전용
└── theory_verification.json           # 이론 검증 결과

cifar10/results/                        # 데이터셋별 실험 결과 (cifar100/, gtsrb/도 동일 구조)
├── exp1_attack_effectiveness/
│   ├── cifar10_results.json           # BA / ASR 표
│   └── cifar10_asr_vs_q.png          # ASR vs Q 꺾은선 그래프
├── exp2_stealth/
│   ├── cifar10_stealth_results.json   # PSNR / SSIM / LPIPS
│   └── *_sample*.png                 # Clean / Poisoned / Diff 이미지
├── exp3_defense/
│   └── cifar10_defense_summary.json   # NC / STRIP / SS 우회 결과
├── exp4_defense_advanced/
│   └── cifar10_defense_advanced_summary.json   # Fine-Pruning / NAD / ShrinkPad 결과
└── ablation/
    ├── cifar10_component_ablation.json
    ├── cifar10_k_ablation.json
    ├── cifar10_qtrain_ablation.json
    └── cifar10_poison_rate_ablation.json
```

---

## Windows 호환성

이 코드는 **Windows / macOS / Linux** 모두에서 동작하도록 설계되었습니다.

| 항목 | 처리 방식 |
|------|-----------|
| `num_workers` | Windows에서 자동으로 `0`으로 설정 (spawn 방식 대응) |
| `pin_memory` | `torch.cuda.is_available()` 기반 자동 설정 |
| `freeze_support` | 모든 `__main__` 블록에 `multiprocessing.freeze_support()` 포함 |
| matplotlib | `Agg` 백엔드 고정 (GUI 없는 환경 포함) |
| 경로 구분자 | 전부 `os.path.join()` 사용 (슬래시 고정 없음) |

---

## 연구계획서 사전 목표치 (참고용 — 실제 결과 아님)

> ⚠️ 아래 두 표는 연구계획서 단계의 **사전 목표치**입니다. 실제 실험 결과와
> 다르며, 논문/보고서 최종 산출물에 이 숫자를 그대로 쓰면 안 됩니다.
> 실제 측정값은 각 데이터셋 폴더의 `results/*.json`과 `README.md`를 참고하세요
> (예: `cifar10/results/exp1_attack_effectiveness/cifar10_results.json`).

### 표 1 (사전 목표치): 공격 효과성 (CIFAR-10)

| 방법 | BA | Q=100 | Q=75 | Q=50 |
|------|-----|-------|------|------|
| **QAFM** | ~80.7% | ~92% | ~88% | **~79%** |
| BadNets | ~80.5% | ~11% | ~9% | ~9% |
| FTrojan | ~80.4% | ~88% | ~20% | ~15% |
| Blended | ~80.2% | ~72% | ~12% | ~8% |

### 표 2 (사전 목표치): 은닉성 지표 (CIFAR-10)

| 방법 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|------|-------------|--------|---------|
| **QAFM** | **42.70** | **0.9968** | **0.0012** |
| BadNets | 31.56 | 0.9831 | 0.0037 |
| FTrojan | 39.55 | 0.9847 | 0.0008 |
| Blended | 34.51 | 0.9854 | 0.0128 |

## 실제 측정 결과 (CIFAR-10, 최신)

### 표 1: 공격 효과성 (k=2, 최신 코드 기준 — `cifar10/results/exp1_attack_effectiveness/cifar10_results.json`)

| 방법 | BA | ASR@Q100 | ASR@Q75 | ASR@Q50 |
|------|-----|----------|---------|---------|
| clean | 95.23% | – | – | – |
| **QAFM** | 94.64% | 99.87% | 99.94% | **100.0%** |
| BadNets | 93.64% | 95.60% | 85.53% | 58.17% |
| FTrojan | 94.70% | 99.99% | 0.83% | 1.53% |
| Blended | 94.83% | 100.0% | 0.86% | 1.84% |

QAFM만 Q=50(최악 압축)까지 ASR이 거의 그대로 유지됨 — FTrojan/Blended는 Q=90 부근에서
압축 강건성 없이 급격히 무너짐("cliff" 패턴), BadNets는 서서히 감소.

### 표 2: 은닉성 지표 (PSNR은 dual-metric 적용됨 — `cifar10/results/exp2_stealth/cifar10_stealth_results.json`)

| 방법 | PSNR 절대(dB) | PSNR 트리거기여(dB) ↑ | SSIM ↑ | LPIPS ↓ |
|------|---------------|------------------------|--------|---------|
| **QAFM** | 31.72 | 41.83 (42dB 기준 근소 미달) | 0.9681 | 0.0017 |
| BadNets | 27.72 | 27.72 | 0.9649 | 0.0010 |
| FTrojan | 45.31 | 45.31 | 0.9981 | 0.0000 |
| Blended | 28.40 | 28.40 | 0.9286 | 0.0034 |

> PSNR만 "절대(원본 대비)"와 "트리거기여(무트리거 자기 기준선 대비)" 두 버전으로
> 분리됨 — QAFM은 Step5(JPEG 재압축)가 트리거와 무관한 손실을 추가로 만들기 때문.
> SSIM/LPIPS도 같은 논리로 trigger-only 버전(`SSIM_trigger_only`, `LPIPS_trigger_only`)이
> 코드에는 추가됐지만, 위 표는 아직 그 코드로 재실행하기 전 값입니다 — 재실행 후 갱신 필요.

---

## 참고 문헌

1. Gu et al., "BadNets", IEEE Access, 2019
2. Wang et al., "FTrojan (An Invisible Black-Box Backdoor Attack through Frequency Domain)", ECCV, 2022
3. Chen et al., "Blended", arXiv, 2017
4. Wang et al., "Neural Cleanse", IEEE S&P, 2019
5. Gao et al., "STRIP", ACSAC, 2019
6. Xue et al., "Compression-resistant Backdoor Attack", Applied Intelligence, 2023
