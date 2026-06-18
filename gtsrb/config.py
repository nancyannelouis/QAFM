"""
GTSRB 실험 설정 — QAFM backdoor attack.
Research: JPEG 압축에 강건한 이미지 백도어 삽입 기법 (IEEE Access 확장)
"""

import os
import platform

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")

for _d in [DATA_DIR, RESULTS_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Platform detection ──────────────────────────────────────────────────────
# Windows는 DataLoader의 multiprocessing이 spawn 방식이므로 num_workers=0 필수
IS_WINDOWS  = platform.system() == "Windows"
NUM_WORKERS = 0 if IS_WINDOWS else 4

# ─── Experiment hardware ──────────────────────────────────────────────────────
DEVICE = "cuda"   # NVIDIA RTX 3060

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "gtsrb"
NUM_CLASSES  = 43
IMG_SIZE     = 32
MEAN         = (0.3403, 0.3121, 0.3214)
STD          = (0.2724, 0.2608, 0.2669)
BACKBONE     = "preact_resnet18"

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
    "k":             2,        # 트리거 강도 (k_min=2, 이론적 최소충분값 — 실측 PSNR 기준 k=3보다 은닉성 유리)
    "q_train":       75,       # 학습 Q값
    "poison_rate":   0.05,     # Poison rate
    "target_label":  0,        # 타겟 클래스
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
    "magnitude":     20.0,                  # 공식 repo 기본값
    "channels":      (1, 2),                # Cb, Cr (색차 채널) — 공식 repo 기본값
    "positions":     ((31, 31), (15, 15)),  # 32×32 전체 이미지 DCT 좌표 — 공식 repo 기본값
    "poison_rate":   0.05,                  # 4개 공격 통제 비교를 위해 QAFM 등과 동일하게 유지
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
ABLATION_POISON_RATE_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]   # 0.0 = 클린 기준선

# ─── Stealth thresholds ───────────────────────────────────────────────────────
PSNR_THRESHOLD  = 42.0   # dB
SSIM_THRESHOLD  = 0.99
LPIPS_THRESHOLD = 0.01

# ─── Defense evaluation ───────────────────────────────────────────────────────
NC_ANOMALY_THRESHOLD = 2.0     # Neural Cleanse: AI < 2 → bypass
STRIP_N_PERTURB      = 100     # STRIP: 배치당 혼합 이미지 수 (논문 기본값)
STRIP_N_EVAL         = 1000    # STRIP: 평가 샘플 수 (논문 관행: 500~2000)
STRIP_FPR_TARGET     = 0.01    # STRIP: threshold 설정 목표 FPR (논문 기본 1%)
