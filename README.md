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

```
SCI_260615/
│
├── config.py                          # 전역 설정 (Q값 범위, 경로, GPU 등)
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
│   └── spectral_signatures.py        # Spectral Signatures (scikit-learn SVD)
│
├── datasets/
│   └── poisoned_dataset.py           # CIFAR-10 / CIFAR-100 / GTSRB 포이즌 파이프라인
│
├── experiments/
│   ├── main_exp1_attack_effectiveness.py   # 메인 실험 1: BA / ASR @ Q=100~50
│   ├── main_exp2_stealth.py               # 메인 실험 2: PSNR / SSIM / LPIPS
│   ├── main_exp3_defense.py               # 메인 실험 3: NC / STRIP / SS 저항성
│   └── ablation/
│       ├── component_ablation.py          # Ablation 1: QAFM vs Fixed-Delta vs No-JPEG
│       ├── k_value_ablation.py            # Ablation 2: k=1~5 강건성-은닉성 트레이드오프
│       ├── q_train_ablation.py            # Ablation 3: q_train=50~95 일반화 검증
│       └── poison_rate_ablation.py        # Ablation 4: poison_rate=1%~20%
│
├── verify_theory.py                   # 수학적 이론 수치 검증 (Theorem 1~3, Lemma 1~2)
├── train.py                           # 단독 학습 스크립트
├── evaluate.py                        # 단독 평가 스크립트
├── run_all.py                         # 전체 실험 순차 실행
└── requirements.txt                   # 패키지 의존성
```

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

```bash
# CIFAR-10 기준, 200 에폭
python run_all.py --dataset cifar10 --epochs 200

# GTSRB
python run_all.py --dataset gtsrb --epochs 200

# 이론 검증만
python run_all.py --theory_only

# 은닉성 분석만 (학습 없이)
python run_all.py --stealth_only
```

---

### 2. 개별 실험 실행

#### 학습 (단독)
```bash
python train.py --dataset cifar10 --method qafm
python train.py --dataset cifar10 --method badnets
python train.py --dataset cifar10 --method ftrojan
python train.py --dataset cifar10 --method blended
python train.py --dataset cifar10 --method clean   # 클린 모델 (기준선)
```

#### 평가 (단독)
```bash
python evaluate.py --dataset cifar10 --method qafm \
    --ckpt checkpoints/cifar10_qafm.pth
```

#### 메인 실험 1: 공격 효과성
```bash
python experiments/main_exp1_attack_effectiveness.py --dataset cifar10
```

출력: `results/exp1_attack_effectiveness/cifar10_results.json` + 그래프

#### 메인 실험 2: 은닉성 분석
```bash
python experiments/main_exp2_stealth.py --dataset cifar10 --n_samples 1000
```

출력: `results/exp2_stealth/cifar10_stealth_results.json`

#### 메인 실험 3: 방어 저항성
```bash
python experiments/main_exp3_defense.py --dataset cifar10 \
    --ckpt_dir checkpoints
```

출력: `results/exp3_defense/cifar10_defense_summary.json`

---

### 3. Ablation Study

```bash
# Ablation 1: Component (QAFM vs Fixed-Delta vs No-JPEG)
python experiments/ablation/component_ablation.py --dataset cifar10

# Ablation 2: k값 변화 (k=1~5, 강건성-은닉성 트레이드오프)
python experiments/ablation/k_value_ablation.py --dataset cifar10

# Ablation 3: 학습 Q값 변화 (q_train=50,60,70,75,85,95)
python experiments/ablation/q_train_ablation.py --dataset cifar10

# Ablation 4: Poison Rate 변화 (1%~20%)
python experiments/ablation/poison_rate_ablation.py --dataset cifar10
```

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

> Proposition 1: k=3, q_train=75 기준으로 Q∈[50,100] 전 범위에서 r≥1 이론 보장.

---

## 실험 결과 위치

```
results/
├── theory_verification.json           # 이론 검증 결과
├── exp1_attack_effectiveness/
│   ├── cifar10_results.json           # BA / ASR 표
│   └── cifar10_asr_vs_q.png          # ASR vs Q 꺾은선 그래프
├── exp2_stealth/
│   ├── cifar10_stealth_results.json   # PSNR / SSIM / LPIPS
│   └── *_sample*.png                 # Clean / Poisoned / Diff 이미지
├── exp3_defense/
│   └── cifar10_defense_summary.json   # NC / STRIP / SS 우회 결과
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

## 논문 기대 결과 요약

### 표 1: 공격 효과성 (CIFAR-10)

| 방법 | BA | Q=100 | Q=75 | Q=50 |
|------|-----|-------|------|------|
| **QAFM** | ~80.7% | ~92% | ~88% | **~79%** |
| BadNets | ~80.5% | ~11% | ~9% | ~9% |
| FTrojan | ~80.4% | ~88% | ~20% | ~15% |
| Blended | ~80.2% | ~72% | ~12% | ~8% |

### 표 2: 은닉성 지표 (CIFAR-10)

| 방법 | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|------|-------------|--------|---------|
| **QAFM** | **42.70** | **0.9968** | **0.0012** |
| BadNets | 31.56 | 0.9831 | 0.0037 |
| FTrojan | 39.55 | 0.9847 | 0.0008 |
| Blended | 34.51 | 0.9854 | 0.0128 |

---

## 참고 문헌

1. Gu et al., "BadNets", IEEE Access, 2019
2. Wang et al., "FTrojan (An Invisible Black-Box Backdoor Attack through Frequency Domain)", ECCV, 2022
3. Chen et al., "Blended", arXiv, 2017
4. Wang et al., "Neural Cleanse", IEEE S&P, 2019
5. Gao et al., "STRIP", ACSAC, 2019
6. Xue et al., "Compression-resistant Backdoor Attack", Applied Intelligence, 2023
