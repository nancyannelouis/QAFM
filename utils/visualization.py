"""
Visualization utilities for QAFM experiments.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict


def save_trigger_comparison(clean: np.ndarray,
                            poisoned: np.ndarray,
                            diff_amplified: np.ndarray,
                            save_path: str,
                            title: str = "QAFM Trigger Visualization"):
    """Clean / Poisoned / Difference × 10 나란히 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(clean);             axes[0].set_title("Clean");    axes[0].axis("off")
    axes[1].imshow(poisoned);          axes[1].set_title("Poisoned"); axes[1].axis("off")
    axes[2].imshow(diff_amplified);    axes[2].set_title("Diff ×10"); axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_asr_vs_quality(results: Dict[str, List],
                        q_values: List[int],
                        save_path: str,
                        title: str = "ASR vs JPEG Quality"):
    """
    메인 실험 1: 방법별 ASR @ 각 Q값 꺾은선 그래프.

    Args:
        results: {"QAFM": [asr@q1, ...], "BadNets": [...], ...}
        q_values: [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = ["o", "s", "^", "D"]
    for (method, asrs), m in zip(results.items(), markers):
        ax.plot(q_values, asrs, marker=m, label=method, linewidth=2)

    ax.set_xlabel("JPEG Quality Factor (Q)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(q_values)
    ax.set_xlim(min(q_values) - 3, max(q_values) + 3)
    ax.set_ylim(-5, 105)
    ax.axhline(60, color="gray", linestyle="--", alpha=0.5, label="60% threshold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ablation_k(k_values: List[int],
                    asr_list: List[float],
                    psnr_list: List[float],
                    save_path: str):
    """
    Ablation Study: k값 변화에 따른 ASR-PSNR 트레이드오프.
    """
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1, color2 = "#1f77b4", "#d62728"

    ax1.plot(k_values, asr_list, "o-", color=color1, label="ASR (%)", linewidth=2)
    ax1.set_xlabel("k (trigger strength)", fontsize=12)
    ax1.set_ylabel("ASR (%)", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(k_values, psnr_list, "s--", color=color2, label="PSNR (dB)", linewidth=2)
    ax2.set_ylabel("PSNR (dB)", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(42, color=color2, linestyle=":", alpha=0.5)
    ax2.set_ylim(30, 55)

    ax1.axvline(3, color="green", linestyle=":", alpha=0.7, label="k=3 (optimal)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    plt.title("k Ablation: ASR vs PSNR Trade-off", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_strip_entropy(clean_entropies: np.ndarray,
                       poison_entropies: np.ndarray,
                       save_path: str,
                       method: str = "QAFM"):
    """STRIP 방어 실험: 정상/포이즌 샘플 엔트로피 분포 비교."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.log2(10) + 0.5, 50)
    ax.hist(clean_entropies,  bins=bins, alpha=0.6, label="Clean",   color="#1f77b4")
    ax.hist(poison_entropies, bins=bins, alpha=0.6, label="Poison",  color="#d62728")
    ax.set_xlabel("Normalized Entropy", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"STRIP Entropy Distribution — {method}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_nc_anomaly_index(anomaly_indices: List[float],
                          num_classes: int,
                          save_path: str,
                          method: str = "QAFM"):
    """Neural Cleanse Anomaly Index 클래스별 막대 그래프."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(num_classes), anomaly_indices, color="#2196F3", edgecolor="black")
    ax.axhline(2.0, color="red", linestyle="--", label="Threshold (AI=2)")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Anomaly Index", fontsize=12)
    ax.set_title(f"Neural Cleanse Anomaly Index — {method}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_r_analysis(Q_train: int, Q_eval_range: List[int], k: int, save_path: str):
    """
    정리 3 r = k * Q_tr / Q_ev 값을 Q_eval 별로 시각화.
    r ≥ 1 이면 트리거 생존 보장.
    """
    from utils.jpeg_utils import get_quantization_table
    Q_tr_table = get_quantization_table(Q_train, "luma")
    r_min_per_q = []
    r_max_per_q = []

    for q_ev in Q_eval_range:
        Q_ev_table = get_quantization_table(q_ev, "luma")
        r_vals = k * Q_tr_table / Q_ev_table
        r_min_per_q.append(r_vals.min())
        r_max_per_q.append(r_vals.max())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(Q_eval_range, r_min_per_q, r_max_per_q, alpha=0.3, label="r range (min-max)")
    ax.plot(Q_eval_range, r_min_per_q, "o-", label="r_min", color="#d62728")
    ax.plot(Q_eval_range, r_max_per_q, "s-", label="r_max", color="#1f77b4")
    ax.axhline(1.0, color="green", linestyle="--", label="r=1 threshold")
    ax.set_xlabel("Q_eval", fontsize=12)
    ax.set_ylabel("r = k·Q_tr / Q_ev", fontsize=12)
    ax.set_title(f"Theorem 3: r values (k={k}, Q_train={Q_train})", fontsize=12)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def make_result_table(data: dict, headers: List[str], save_path: str, title: str = ""):
    """실험 결과를 matplotlib table로 저장."""
    rows = list(data.values())
    row_labels = list(data.keys())
    n_rows, n_cols = len(rows), len(headers)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.5), max(3, n_rows * 0.6 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    if title:
        ax.set_title(title, fontsize=13, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
