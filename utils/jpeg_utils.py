"""
JPEG quantization utilities for QAFM.

Mathematical foundation:
  Q(C) = round(C / Q_ij) * Q_ij            [양자화 연산자 정의]
  Theorem 1: Q(C + k*Q) = Q(C) + k*Q      [정수 이동 불변성]
  Proposition 1: k >= ceil(Q_ev / Q_tr)    [평가 단계 생존 충분조건]

JPEG standard luminance quantization table (libjpeg baseline, Q=50).
Scale formula per libjpeg: Q_q = floor((100-q)*Q_base/50 + 0.5), q in [50,100]
"""

import io
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ─── JPEG Standard Luminance Quantization Table (base at Q=50) ───────────────
# Source: JPEG standard / libjpeg reference implementation
JPEG_LUMA_BASE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

# JPEG standard chrominance quantization table (base at Q=50)
JPEG_CHROMA_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_quantization_table(quality: int, channel: str = "luma") -> np.ndarray:
    """
    libjpeg 스케일링 공식에 따라 quality ∈ [1, 100] 의 양자화 테이블 반환.

    libjpeg scaling (Q ∈ [50, 100]):
        scale = (100 - quality) / 50
        Q_q = floor(scale * Q_base + 0.5)   → 논문 3.5절 수식
        Q=100: scale=0 → 모든 스텝=1 (무손실에 근접, clamp 적용)

    Args:
        quality: JPEG quality factor (1 ~ 100)
        channel: "luma" or "chroma"

    Returns:
        8×8 float32 quantization table
    """
    base = JPEG_LUMA_BASE if channel == "luma" else JPEG_CHROMA_BASE

    if quality >= 50:
        scale = (100 - quality) / 50.0
    else:
        # Q < 50: libjpeg uses 5000/quality scaling (참고용, 본 연구 범위 밖)
        scale = 50.0 / quality

    table = np.floor(scale * base + 0.5).astype(np.float32)
    table = np.clip(table, 1, 255)   # quantization step must be ≥ 1
    return table


def quantize(C: np.ndarray, Q_table: np.ndarray) -> np.ndarray:
    """
    JPEG 양자화 연산자 Q(C) = round(C / Q_ij) * Q_ij.

    Args:
        C:       DCT 계수 배열 (8, 8) or (N, 8, 8)
        Q_table: 양자화 테이블 (8, 8)

    Returns:
        양자화된 DCT 계수 (same shape as C)
    """
    return np.round(C / Q_table) * Q_table


def dequantize(C_q: np.ndarray, Q_table: np.ndarray) -> np.ndarray:
    """역양자화 — 이미 quantize()에서 Q가 곱해져 있으므로 동일."""
    return C_q


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II (scipy 또는 수동 구현)."""
    from scipy.fft import dctn
    return dctn(block, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    """2D IDCT."""
    from scipy.fft import idctn
    return idctn(block, norm="ortho")


# ─── Theorem 1 verification (수학적 불변성 검증 유틸) ────────────────────────
def verify_theorem1(C: float, Q: float, k: int) -> bool:
    """
    Theorem 1: Q(C + k·Q) = Q(C) + k·Q

    f(C,Q) = round(C/Q)*Q 이므로 직접 계산하여 등식 확인.
    """
    lhs = round((C + k * Q) / Q) * Q
    rhs = round(C / Q) * Q + k * Q
    return abs(lhs - rhs) < 1e-6


def compute_r(delta: float, Q_val: float) -> float:
    """r = delta / Q_val (트리거 대 양자화 스텝 비율)."""
    return delta / Q_val


def compute_k_min(Q_train: int, Q_eval_min: int = 50) -> int:
    """
    Proposition 1: Q ∈ [50,100] 전 범위 트리거 생존 보장하는 k 최솟값.
    Q=100은 스텝이 모두 1이므로 r이 가장 크고 트리거 생존이 가장 쉬운 케이스.
    따라서 최악 케이스는 여전히 Q_eval=50.

    k_min = ceil(max Q_ev[i,j] / Q_tr[i,j])

    Q_max_scale / Q_min_scale = ((100-50)/50) / ((100-95)/50) = 50/5 = 10?
    Wait: 논문에서는 Q_train=75, Q_eval=50일 때 ratio=2로 설명.
    ((100-50)/50) / ((100-75)/50) = 2 이므로 k_min=2.

    일반적으로 Q_eval_min이 주어지면:
      scale_ev = (100 - Q_eval_min) / 50
      scale_tr = (100 - Q_train) / 50
      ratio = scale_ev / scale_tr
    """
    scale_ev = (100 - Q_eval_min) / 50.0
    scale_tr = (100 - Q_train) / 50.0
    ratio = scale_ev / scale_tr
    return math.ceil(ratio)


def r_range_classification(r: float) -> dict:
    """
    보조 정리 1 r 범위별 이동량 분류표 (논문 표).

    Returns dict with case, movement_range, trigger_survival.
    """
    if 0 < r < 0.5:
        return {"case": "A", "r_range": "(0, 0.5)", "movement": "0 or 1 (f-dependent)", "survival": False}
    elif 0.5 <= r < 1.0:
        return {"case": "B/C/D", "r_range": "[0.5, 1)", "movement": "0 or 1 (f-dependent)", "survival": "conditional"}
    elif r >= 1.0 and r == int(r):
        m = int(r)
        return {"case": "Case1", "r_range": f"r={m} (integer)", "movement": f"{m} or {m+1}", "survival": True}
    else:
        m = int(r)
        return {"case": "Case2", "r_range": f"[{m},{m+1})", "movement": f"{m} or {m+1} or {m+2}", "survival": True}


# ─── Whole-image JPEG compress / decompress ───────────────────────────────────
def jpeg_compress(image_np: np.ndarray, quality: int) -> np.ndarray:
    """
    numpy uint8 이미지 (H, W, 3) or (H, W) → JPEG 압축 → 복원 → numpy uint8.

    실제 libjpeg 파이프라인을 PIL을 통해 그대로 사용.
    """
    if image_np.ndim == 2:
        pil_img = Image.fromarray(image_np, mode="L")
    else:
        pil_img = Image.fromarray(image_np, mode="RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    result = np.array(Image.open(buf))
    return result


def jpeg_compress_tensor(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Tensor (C, H, W) in [0,1] → JPEG → Tensor (C, H, W) in [0,1].
    """
    np_img = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    np_out = jpeg_compress(np_img, quality)
    out = torch.from_numpy(np_out).float() / 255.0
    return out.permute(2, 0, 1)


def jpeg_compress_batch(batch: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Batch of tensors (N, C, H, W) in [0,1] → JPEG → (N, C, H, W).
    """
    return torch.stack([jpeg_compress_tensor(x, quality) for x in batch])


# ─── Block-level DCT JPEG pipeline (used for trigger insertion) ──────────────
def rgb_to_ycbcr(image_np: np.ndarray) -> np.ndarray:
    """RGB float32 (H,W,3) → YCbCr float32 (H,W,3), Y in [0,255]."""
    R, G, B = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128
    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr_to_rgb(ycbcr_np: np.ndarray) -> np.ndarray:
    """YCbCr float32 (H,W,3) → RGB float32 (H,W,3)."""
    Y, Cb, Cr = ycbcr_np[:, :, 0], ycbcr_np[:, :, 1] - 128, ycbcr_np[:, :, 2] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)


def insert_dct_trigger(image_np: np.ndarray,
                       trigger_pos: tuple,
                       k: int,
                       q_train: int,
                       periodic: bool = True) -> np.ndarray:
    """
    QAFM 트리거 삽입 알고리즘 (논문 5. Research Design 3) 트리거 삽입 알고리즘).

    Step 1: RGB → YCbCr
    Step 2: Y채널을 8×8 블록으로 분할 후 2D DCT
    Step 3: (i,j) 위치에 Δ = k * Q_{ij} 삽입
    Step 4: IDCT → 블록 병합 → YCbCr → RGB
    Step 5: q_train으로 JPEG 압축

    Args:
        image_np:    uint8 RGB (H, W, 3)
        trigger_pos: (i, j) DCT 블록 내 위치
        k:           트리거 강도 정수 (k ≥ k_min)
        q_train:     학습 Q값
        periodic:    True → 모든 8×8 블록에 삽입 (전역 트리거)

    Returns:
        uint8 RGB (H, W, 3) — JPEG 압축 후에도 트리거 보존 (정리 2)
    """
    i_pos, j_pos = trigger_pos
    Q_table = get_quantization_table(q_train, channel="luma")
    delta_ij = k * Q_table[i_pos, j_pos]   # Δ = k · Q_{ij}

    img_float = image_np.astype(np.float32)
    ycbcr = rgb_to_ycbcr(img_float)
    Y = ycbcr[:, :, 0].copy()

    H, W = Y.shape
    Y_modified = Y.copy()

    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            block = Y[row:row + 8, col:col + 8]
            dct_block = dct2(block)
            dct_block[i_pos, j_pos] += delta_ij     # C'_{ij} = C_{ij} + k·Q_{ij}
            Y_modified[row:row + 8, col:col + 8] = idct2(dct_block)

    ycbcr[:, :, 0] = np.clip(Y_modified, 0, 255)
    rgb_modified = ycbcr_to_rgb(ycbcr)
    result = np.clip(rgb_modified, 0, 255).astype(np.uint8)

    # Step 5: JPEG 압축 — 정리 2에 의해 트리거 k·Q_{ij} 완벽 보존
    result = jpeg_compress(result, q_train)
    return result


def verify_trigger_survival(
    image_np: np.ndarray,
    trigger_pos: tuple,
    k: int,
    q_train: int,
    q_eval: int,
) -> dict:
    """
    트리거 삽입 후 q_eval로 재압축 시 트리거 생존 여부 수치 확인.

    정리 3: r = k * Q_tr / Q_ev ≥ 1 이면 모든 f에서 트리거 생존.
    """
    i_pos, j_pos = trigger_pos
    Q_tr = get_quantization_table(q_train, "luma")[i_pos, j_pos]
    Q_ev = get_quantization_table(q_eval,  "luma")[i_pos, j_pos]
    delta = k * Q_tr
    r = delta / Q_ev
    survival_guaranteed = r >= 1.0

    # 실제 삽입 후 압축해서 DCT 계수 차이 확인
    clean = jpeg_compress(image_np, q_eval)
    poisoned = insert_dct_trigger(image_np, trigger_pos, k, q_train)
    poisoned_recompressed = jpeg_compress(poisoned, q_eval)

    clean_ycbcr    = rgb_to_ycbcr(clean.astype(np.float32))
    poisoned_ycbcr = rgb_to_ycbcr(poisoned_recompressed.astype(np.float32))
    Y_clean    = clean_ycbcr[:, :, 0]
    Y_poisoned = poisoned_ycbcr[:, :, 0]

    diffs = []
    H, W = Y_clean.shape
    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            d_c = dct2(Y_clean[row:row + 8, col:col + 8])
            d_p = dct2(Y_poisoned[row:row + 8, col:col + 8])
            diffs.append(d_p[i_pos, j_pos] - d_c[i_pos, j_pos])

    diffs = np.array(diffs)
    survival_rate = np.mean(np.abs(diffs) > 0.5)

    return {
        "q_train":              q_train,
        "q_eval":               q_eval,
        "k":                    k,
        "Q_tr_ij":              Q_tr,
        "Q_ev_ij":              Q_ev,
        "delta":                delta,
        "r":                    r,
        "survival_guaranteed":  survival_guaranteed,
        "empirical_survival":   survival_rate,
        "mean_dct_diff":        float(np.mean(diffs)),
        "std_dct_diff":         float(np.std(diffs)),
    }
