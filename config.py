"""
Global configuration for QAFM backdoor attack experiments.
Research: JPEG 압축에 강건한 이미지 백도어 삽입 기법 (IEEE Access 확장)
"""

import os
import platform

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")

for _d in [DATA_DIR, RESULTS_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Platform detection ──────────────────────────────────────────────────────
# Windows는 DataLoader의 multiprocessing이 spawn 방식이므로 num_workers=0 필수
IS_WINDOWS  = platform.system() == "Windows"
NUM_WORKERS = 0 if IS_WINDOWS else 4

# ─── Experiment hardware ──────────────────────────────────────────────────────
DEVICE = "cuda"   # NVIDIA RTX 3060

# ─── Dataset configs ─────────────────────────────────────────────────────────
DATASET_CFG = {
    "cifar10": {
        "num_classes": 10,
        "img_size":    32,
        "mean":        (0.4914, 0.4822, 0.4465),
        "std":         (0.2023, 0.1994, 0.2010),
        "backbone":    "resnet18",
    },
    "cifar100": {
        "num_classes": 100,
        "img_size":    32,
        "mean":        (0.5071, 0.4867, 0.4408),
        "std":         (0.2675, 0.2565, 0.2761),
        "backbone":    "resnet18",
    },
    "gtsrb": {
        "num_classes": 43,
        "img_size":    32,
        "mean":        (0.3403, 0.3121, 0.3214),
        "std":         (0.2724, 0.2608, 0.2669),
        "backbone":    "preact_resnet18",
    },
}

# ─── Training defaults ────────────────────────────────────────────────────────
TRAIN_CFG = {
    "epochs":        200,
    "batch_size":    128,
    "lr":            0.1,
    "momentum":      0.9,
    "weight_decay":  5e-4,
    "lr_milestones": [100, 150],
    "lr_gamma":      0.1,
    "num_workers":   NUM_WORKERS,
}

# ─── QAFM attack defaults (Section 5 Research Design) ────────────────────────
QAFM_CFG = {
    "trigger_pos":   (0, 1),   # (i, j) DCT 중주파 위치 — 강건성·은닉성 균형 최적
    "k":             3,        # 트리거 강도 (k ≥ k_min=2, 경험적 최적값)
    "q_train":       75,       # 학습 Q값
    "poison_rate":   0.05,     # Poison rate
    "target_label":  0,        # 타겟 클래스
    "color_channel": "Y",      # YCbCr의 Y 채널에만 삽입
}

# ─── Evaluation Q values (Section 6 Experiment Plan) ─────────────────────────
EVAL_Q_VALUES = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]

# ─── Baseline attack configs ──────────────────────────────────────────────────
BADNETS_CFG = {
    "patch_size":    3,        # 3×3 white patch
    "patch_pos":     "br",     # bottom-right
    "poison_rate":   0.05,
    "target_label":  0,
}

FTROJAN_CFG = {
    "trigger_pos":   (0, 1),   # same DCT position as QAFM for fair comparison
    "delta":         20.0,     # 고정 델타 (양자화 테이블과 미정렬)
    "poison_rate":   0.05,
    "target_label":  0,
}

BLENDED_CFG = {
    "alpha":         0.1,      # 혼합 투명도
    "pattern":       "random", # 랜덤 노이즈 패턴
    "poison_rate":   0.05,
    "target_label":  0,
}

# ─── Ablation ranges ─────────────────────────────────────────────────────────
ABLATION_K_VALUES           = [1, 2, 3, 4, 5]
ABLATION_Q_TRAIN_VALUES     = [50, 60, 70, 75, 85, 95]
ABLATION_POISON_RATE_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20]

# ─── Stealth thresholds ───────────────────────────────────────────────────────
PSNR_THRESHOLD  = 42.0   # dB
SSIM_THRESHOLD  = 0.99
LPIPS_THRESHOLD = 0.01

# ─── Defense evaluation ───────────────────────────────────────────────────────
NC_ANOMALY_THRESHOLD = 2.0     # Neural Cleanse: AI < 2 → bypass
STRIP_N_PERTURB      = 100     # STRIP: 배치당 혼합 이미지 수 (논문 기본값)
STRIP_N_EVAL         = 1000    # STRIP: 평가 샘플 수 (논문 관행: 500~2000)
STRIP_FPR_TARGET     = 0.01    # STRIP: threshold 설정 목표 FPR (논문 기본 1%)
