"""
IEEE Access 논문용 그림/표 생성 — 이론/개념 다이어그램 (Figure 1,2,3,4 / Table 2).

Figure 1,3은 실제 이미지/이론 수식을 그대로 사용하고(임의 수치 없음),
Figure 2,4는 알고리즘 단계를 보여주는 개념 다이어그램(블록+화살표)임.

주의: Figure 4(QAFM 파이프라인)는 문서 초안에 적힌 "Cb/Cr 색차 채널" 설계가
아니라, 실제 구현(attacks/qafm.py, utils/jpeg_utils.py:insert_dct_trigger)이
사용하는 **Y(밝기) 채널** 기준으로 그림. 문서 본문의 "Cb/Cr 선택 근거" 서술은
실제 구현과 다르므로 사용자가 추후 직접 수정 필요 (대화에서 확인/합의됨).

Usage:
    python paper_assets/generate_theory_assets.py
"""

import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

_KOREAN_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(_KOREAN_FONT_PATH):
    fm.fontManager.addfont(_KOREAN_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_KOREAN_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

from utils.jpeg_utils import get_quantization_table, jpeg_compress, insert_dct_trigger
from utils.visualization import make_result_table

FIG_DIR = os.path.join(ROOT, "paper_assets", "figures")
TAB_DIR = os.path.join(ROOT, "paper_assets", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _draw_box(ax, xy, w, h, text, fontsize=10, color="#E3F2FD"):
    box = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                  linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="black", linewidth=1.2)
    ax.add_patch(arrow)


# ============================================================================
# Figure 1: 원본/FTrojan/QAFM 압축 전후 비교 (overview)
# ============================================================================
def figure1_overview():
    print("[Figure 1] 원본/FTrojan/QAFM 압축 전후 비교")
    sys.path.insert(0, os.path.join(ROOT, "cifar10"))
    import config as ds_config
    from dataset import load_raw_dataset
    from attacks import build_attack

    _, _, test_imgs, _ = load_raw_dataset(ds_config.DATA_DIR)
    clean = test_imgs[0]

    ftrojan_cfg = dict(ds_config.FTROJAN_CFG); ftrojan_cfg["target_label"] = 0
    qafm_cfg = dict(ds_config.QAFM_CFG); qafm_cfg["target_label"] = 0
    ftrojan = build_attack("ftrojan", ftrojan_cfg)
    qafm = build_attack("qafm", qafm_cfg)

    rows = [
        ("Original\n(no trigger)", clean, None),
        ("FTrojan\n(fixed delta)", ftrojan.poison_image(clean), "ASR@Q50 약 1.5% (소멸)"),
        ("QAFM\n(quantization-aligned)", qafm.poison_image(clean), "ASR@Q50=100% (생존)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(5.5, 7))
    for r, (label, img, note) in enumerate(rows):
        img_q50 = jpeg_compress(img, 50)
        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        axes[r, 1].imshow(img_q50); axes[r, 1].axis("off")
        axes[r, 0].text(-0.25, 0.5, label, transform=axes[r, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
        if note:
            axes[r, 1].text(0.5, -0.12, note, transform=axes[r, 1].transAxes,
                             ha="center", fontsize=9, color="#d62728")
        if r == 0:
            axes[r, 0].set_title("Before Compression", fontsize=11)
            axes[r, 1].set_title("After JPEG Q=50", fontsize=11)
    fig.suptitle("Figure 1: Trigger Survival under JPEG Compression — Overview", fontsize=12)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG_DIR, "figure1_overview_comparison.png"), dpi=150)
    plt.close()

    sys.path.pop(0)
    for mod in ["config", "dataset", "attacks", "models"]:
        sys.modules.pop(mod, None)


# ============================================================================
# Figure 2: JPEG 압축 파이프라인 다이어그램 (개념도)
# ============================================================================
def figure2_jpeg_pipeline():
    print("[Figure 2] JPEG 압축 파이프라인 다이어그램")
    steps = ["RGB", "YCbCr\n변환", "8×8 블록\n분할", "2D DCT", "양자화\n(Quantize)",
             "역양자화\n(Dequantize)", "IDCT", "RGB\n복원"]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    n = len(steps)
    w, h, gap = 1.2, 0.8, 0.35
    for idx, s in enumerate(steps):
        x = idx * (w + gap)
        _draw_box(ax, (x, 0), w, h, s, fontsize=9)
        if idx < n - 1:
            _draw_arrow(ax, (x + w, h / 2), (x + w + gap, h / 2))
    ax.set_xlim(-0.2, n * (w + gap))
    ax.set_ylim(-0.3, h + 0.3)
    ax.axis("off")
    ax.set_title("Figure 2: JPEG Compression Pipeline", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure2_jpeg_pipeline.png"), dpi=150)
    plt.close()


# ============================================================================
# Figure 3: 트리거 생존 조건(r) 분포 그래프 + 케이스 분류 다이어그램
# ============================================================================
def figure3_survival_condition(k: int = 2, q_train: int = 75):
    print("[Figure 3] 트리거 생존 조건 r 분포 + 케이스 분류 다이어그램")
    Q_eval_range = list(range(50, 101, 5))
    Q_tr_table = get_quantization_table(q_train, "luma")
    i, j = 0, 1  # QAFM 기본 삽입 위치

    r_vals = []
    for q_ev in Q_eval_range:
        Q_ev_table = get_quantization_table(q_ev, "luma")
        r_vals.append(k * Q_tr_table[i, j] / Q_ev_table[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) r 분포 그래프
    ax = axes[0]
    ax.plot(Q_eval_range, r_vals, "o-", color="#1f77b4", linewidth=2)
    ax.axhline(1.0, color="green", linestyle="--", label="r=1 (생존 임계)")
    ax.set_xlabel("Q_eval", fontsize=11)
    ax.set_ylabel(f"r = k·Q_tr[{i},{j}] / Q_ev[{i},{j}]", fontsize=11)
    ax.set_title(f"(a) r vs Q_eval (k={k}, q_train={q_train})", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) 케이스 분류 다이어그램 (Lemma 1: r,f 기준 A~D 케이스)
    ax2 = axes[1]
    ax2.set_xlim(0, 2); ax2.set_ylim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle=":")
    ax2.axhline(0.5, color="gray", linestyle=":")
    cases = [
        (0.25, 0.25, "Case A\nr<0.5, f<0.5\n이동량=0"),
        (1.25, 0.75, "Case B\nr≥0.5, f≥0.5\n이동량=1"),
        (1.25, 0.25, "Case C\nr≥0.5, f<0.5\n이동량=0"),
        (0.25, 0.75, "Case D\nr<0.5, f≥0.5\n이동량=0"),
    ]
    for x, y, label in cases:
        ax2.text(x, y, label, ha="center", va="center", fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="black"))
    ax2.set_xlabel("r = δ/Q (고정 델타 비율)", fontsize=11)
    ax2.set_ylabel("f = frac(C/Q) (DCT 계수 소수부)", fontsize=11)
    ax2.set_title("(b) Lemma 1: Rounding Movement Case Classification", fontsize=11)

    fig.suptitle("Figure 3: Trigger Survival Condition Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure3_survival_condition.png"), dpi=150)
    plt.close()


# ============================================================================
# Figure 4: QAFM 전체 공격 파이프라인 (핵심 그림) — 실제 구현 기준 Y채널
# ============================================================================
def figure4_qafm_pipeline():
    print("[Figure 4] QAFM 공격 파이프라인 (실제 구현: Y채널 기준)")
    steps = [
        "RGB\n입력",
        "YCbCr\n변환",
        "Y채널\n8×8 블록\n분할",
        "2D DCT",
        "Δ_ij = k·Q_ij\n삽입\n(0,1) 위치",
        "IDCT +\n블록 병합",
        "YCbCr→RGB",
        "q_train\nJPEG 압축\n(Step 5)",
    ]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    n = len(steps)
    w, h, gap = 1.4, 1.0, 0.35
    for idx, s in enumerate(steps):
        x = idx * (w + gap)
        color = "#FFCDD2" if "Δ_ij" in s else ("#C8E6C9" if "압축" in s else "#E3F2FD")
        _draw_box(ax, (x, 0), w, h, s, fontsize=8.5, color=color)
        if idx < n - 1:
            _draw_arrow(ax, (x + w, h / 2), (x + w + gap, h / 2))
    ax.set_xlim(-0.2, n * (w + gap))
    ax.set_ylim(-0.5, h + 0.5)
    ax.axis("off")
    ax.set_title("Figure 4: QAFM Attack Pipeline (Y-channel, as implemented)", fontsize=13)
    ax.text(0.5, -0.15,
            "Note: 실제 구현은 Y(밝기) 채널 기준 — 문서 초안의 'Cb/Cr 색차 채널' 서술은 수정 필요",
            transform=ax.transAxes, ha="center", fontsize=9, color="#d62728")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure4_qafm_pipeline.png"), dpi=150)
    plt.close()


# ============================================================================
# Table 2: 트리거 생존 비율(r) 범위별 분류표
# ============================================================================
def table2_survival_classification():
    print("[Table 2] 트리거 생존 비율 범위별 분류표")
    headers = ["r 범위", "케이스 (f 기준)", "양자화 이동량", "트리거 생존 보장"]
    data = {
        "r ∈ [0, 0.5)":       ["Case A(f<0.5) / D(f≥0.5)", "0",        "미보장 (소멸 가능)"],
        "r ∈ [0.5, 1)":       ["Case C(f<0.5) / B(f≥0.5)", "0 또는 1", "미보장 (경계 의존)"],
        "r이 정수, r ≥ 1":     ["정수 r=m",                  "m 또는 m+1", "보장 (정리 3)"],
        "r이 정수가 아닌 실수, r≥1": ["실수 r=m+α (0<α<1)",  "m, m+1 또는 m+2", "보장 (정리 3)"],
    }
    make_result_table(data, headers,
                       save_path=os.path.join(TAB_DIR, "table2_survival_classification.png"),
                       title="Table 2: Trigger Survival Ratio Classification")
    with open(os.path.join(TAB_DIR, "table2_survival_classification.csv"), "w", encoding="utf-8") as f:
        f.write("r_range," + ",".join(headers[1:]) + "\n")
        for r, row in data.items():
            f.write(f"{r}," + ",".join(row) + "\n")


if __name__ == "__main__":
    figure1_overview()
    figure2_jpeg_pipeline()
    figure3_survival_condition()
    figure4_qafm_pipeline()
    table2_survival_classification()
    print("\n완료.")
