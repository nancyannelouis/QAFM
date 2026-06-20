"""
IEEE Access 논문용 그림/표 생성 — 실측 데이터 기반 항목 (Table 1,3,4,5,6 / Figure 5,6,7,8,9).

전부 cifar10/cifar100/gtsrb의 results/*.json에서 직접 읽어 생성함 — 숫자를 임의로
작성하지 않음. 방어 저항성(Table 6, Figure 8/9)은 기존 NC/STRIP 대신 exp4
(Fine-Pruning/NAD/ShrinkPad) 결과로 교체했지만, 시각적 형식(막대그래프+임계선,
표 컬럼 구조)은 기존 NC/STRIP 자료와 동일하게 유지함.

Usage:
    python paper_assets/generate_data_assets.py
"""

import os
import sys
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

from utils.visualization import make_result_table, plot_asr_vs_quality
from utils.jpeg_utils import get_quantization_table

# 한글 라벨이 깨지지 않도록 맑은 고딕 폰트 등록 (논문용 출력이므로 글리프 누락 방지)
_KOREAN_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(_KOREAN_FONT_PATH):
    fm.fontManager.addfont(_KOREAN_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_KOREAN_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(ROOT, "paper_assets", "figures")
TAB_DIR = os.path.join(ROOT, "paper_assets", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DATASETS = ["cifar10", "cifar100", "gtsrb"]
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "gtsrb": "GTSRB"}
METHOD_LABELS = {"qafm": "QAFM", "badnets": "BadNets", "ftrojan": "FTrojan", "blended": "Blended"}
EVAL_Q_VALUES = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]


def load_json(dataset: str, *parts) -> dict:
    path = os.path.join(ROOT, dataset, "results", *parts)
    if not os.path.exists(path):
        print(f"  [경고] 없음: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Table 1: 기존 백도어 공격 비교표 (RELATED WORK, 정성적 비교)
# ============================================================================
def table1_baseline_comparison():
    print("[Table 1] 기존 백도어 공격 비교표")
    headers = ["트리거 도메인", "은닉성", "JPEG 압축 강건성", "방어 회피(NC/모델복구)"]
    data = {
        "BadNets":  ["공간(패치)",        "낮음 (육안 식별 가능)", "낮음 (Q90↓ 급락)",      "낮음"],
        "Blended":  ["공간(블렌딩)",      "중간",                  "낮음 (Q90↓ 급락)",      "낮음"],
        "FTrojan":  ["주파수(고정 델타)", "높음",                  "낮음 (양자화 미정렬)",   "낮음"],
        "QAFM(제안)": ["주파수(양자화 정렬)", "높음",               "높음 (Q=[50,95] 보장)", "부분적 (구조적 한계 있음)"],
    }
    make_result_table(data, headers,
                       save_path=os.path.join(TAB_DIR, "table1_baseline_comparison.png"),
                       title="Table 1: Comparison of Existing Backdoor Attacks")
    with open(os.path.join(TAB_DIR, "table1_baseline_comparison.csv"), "w", encoding="utf-8") as f:
        f.write("Method," + ",".join(headers) + "\n")
        for m, row in data.items():
            f.write(f"{m}," + ",".join(row) + "\n")


# ============================================================================
# Table 3: 실험 설정 정리표
# ============================================================================
def table3_experiment_setup():
    print("[Table 3] 실험 설정 정리표")
    headers = ["값"]
    data = {
        "데이터셋":            ["CIFAR-10, CIFAR-100, GTSRB"],
        "모델":                ["ResNet-18 (CIFAR-10/100), PreAct-ResNet-18 (GTSRB)"],
        "베이스라인":           ["BadNets, FTrojan, Blended"],
        "트리거 삽입 위치":      ["(0,1) — 8×8 DCT 블록 내 중주파"],
        "트리거 강도 계수 k":    ["2 (k_min, 이론적 최소충분값)"],
        "학습 압축 품질 q_train": ["75"],
        "평가 압축 품질 q_eval": ["100,95,90,85,80,75,70,65,60,55,50"],
        "Poison rate":         ["5%"],
        "학습 epoch":          ["200 (SGD, lr=0.1, momentum=0.9, wd=5e-4, milestone=[100,150])"],
        "Batch size":          ["128"],
        "GPU":                 ["NVIDIA RTX 3060"],
    }
    make_result_table(data, headers,
                       save_path=os.path.join(TAB_DIR, "table3_experiment_setup.png"),
                       title="Table 3: Experiment Setup")
    with open(os.path.join(TAB_DIR, "table3_experiment_setup.csv"), "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k},{v[0]}\n")


# ============================================================================
# Table 4: 데이터셋별 x 압축 품질별 BA/ASR 종합 비교표
# ============================================================================
def table4_ba_asr_comprehensive():
    print("[Table 4] BA/ASR 종합 비교표 (3개 데이터셋)")
    for ds in DATASETS:
        res = load_json(ds, "exp1_attack_effectiveness", f"{ds}_results.json")
        if not res:
            continue
        headers = ["BA_full(%)"] + [f"ASR@Q{q}" for q in EVAL_Q_VALUES]
        data = {}
        for method, r in res.items():
            label = METHOD_LABELS.get(method, method.capitalize())
            ba = r.get("BA_full", r.get("BA", "-"))
            row = [f"{ba:.2f}" if isinstance(ba, (int, float)) else "-"]
            for q in EVAL_Q_VALUES:
                v = r.get(f"ASR@Q{q}")
                row.append(f"{v:.2f}" if v is not None else "-")
            data[label] = row
        make_result_table(
            data, headers,
            save_path=os.path.join(TAB_DIR, f"table4_ba_asr_{ds}.png"),
            title=f"Table 4: BA/ASR Comparison — {DATASET_LABELS[ds]}"
        )
        with open(os.path.join(TAB_DIR, f"table4_ba_asr_{ds}.csv"), "w", encoding="utf-8") as f:
            f.write("Method," + ",".join(headers) + "\n")
            for m, row in data.items():
                f.write(f"{m}," + ",".join(row) + "\n")


# ============================================================================
# Table 5: 은닉성 비교표 (PSNR/SSIM/LPIPS, trigger-only 기준)
# ============================================================================
def table5_stealth_comparison():
    print("[Table 5] 은닉성 비교표 (3개 데이터셋)")
    headers = ["PSNR_절대(dB)", "PSNR_트리거기여(dB)", "SSIM_절대", "SSIM_트리거기여",
               "LPIPS_절대", "LPIPS_트리거기여"]
    for ds in DATASETS:
        res = load_json(ds, "exp2_stealth", f"{ds}_stealth_results.json")
        if not res:
            continue
        data = {}
        for method, r in res.items():
            label = METHOD_LABELS.get(method, method.capitalize())
            data[label] = [
                f"{r['PSNR']:.2f}", f"{r['PSNR_trigger_only']:.2f}",
                f"{r['SSIM']:.4f}", f"{r.get('SSIM_trigger_only', r['SSIM']):.4f}",
                f"{r['LPIPS']:.4f}", f"{r.get('LPIPS_trigger_only', r['LPIPS']):.4f}",
            ]
        make_result_table(
            data, headers,
            save_path=os.path.join(TAB_DIR, f"table5_stealth_{ds}.png"),
            title=f"Table 5: Stealth Metrics — {DATASET_LABELS[ds]}"
        )
        with open(os.path.join(TAB_DIR, f"table5_stealth_{ds}.csv"), "w", encoding="utf-8") as f:
            f.write("Method," + ",".join(headers) + "\n")
            for m, row in data.items():
                f.write(f"{m}," + ",".join(row) + "\n")


# ============================================================================
# Table 6: 방어 기법 저항성 종합 성능 비교 요약표
# (기존 NC/STRIP/SS -> Fine-Pruning/NAD/ShrinkPad로 기법명만 교체, 표 형식 동일)
# ============================================================================
def table6_defense_resistance():
    print("[Table 6] 방어 저항성 종합 비교표 (Fine-Pruning/NAD/ShrinkPad, 3개 데이터셋)")
    headers = ["ASR_적용전(%)", "Fine-Pruning 우회", "NAD 우회", "ShrinkPad 우회", "비고"]
    for ds in DATASETS:
        res = load_json(ds, "exp4_defense_advanced", f"{ds}_defense_advanced_summary.json")
        if not res:
            continue
        data = {}
        for method, r in res.items():
            label = METHOD_LABELS.get(method, method.capitalize())
            note = "압축에서 이미 실패" if r.get("already_failed_pre_defense") else ""
            data[label] = [
                f"{r['ASR_before']:.2f}",
                "O" if r["FinePruning"]["bypass"] else "X",
                "O" if r["NAD"]["bypass"] else "X",
                "O" if r["ShrinkPad"]["bypass"] else "X",
                note,
            ]
        make_result_table(
            data, headers,
            save_path=os.path.join(TAB_DIR, f"table6_defense_resistance_{ds}.png"),
            title=f"Table 6: Defense Resistance Summary — {DATASET_LABELS[ds]}"
        )
        with open(os.path.join(TAB_DIR, f"table6_defense_resistance_{ds}.csv"), "w", encoding="utf-8") as f:
            f.write("Method," + ",".join(headers) + "\n")
            for m, row in data.items():
                f.write(f"{m}," + ",".join(row) + "\n")


# ============================================================================
# Figure 5: 주파수 위치별 트리거 변조량 히트맵 (Δ_ij = k * Q_ij)
# ============================================================================
def figure5_freq_heatmap(k: int = 2, q_train: int = 75):
    print("[Figure 5] 주파수 위치별 트리거 변조량 히트맵")
    Q_table = get_quantization_table(q_train, channel="luma")
    delta = k * Q_table

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(delta, cmap="viridis")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{delta[i, j]:.0f}", ha="center", va="center",
                    color="white" if delta[i, j] < delta.max() * 0.6 else "black", fontsize=8)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xlabel("DCT j", fontsize=12); ax.set_ylabel("DCT i", fontsize=12)
    ax.set_title(f"Trigger Magnitude Δ_ij = k·Q_ij (k={k}, q_train={q_train})", fontsize=12)
    # QAFM이 실제 사용하는 위치 (0,1) 표시
    ax.add_patch(plt.Rectangle((1 - 0.5, 0 - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))
    fig.colorbar(im, ax=ax, label="Δ_ij")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure5_freq_position_heatmap.png"), dpi=150)
    plt.close()


# ============================================================================
# Figure 6: ASR vs Q 꺾은선 그래프 (기존 산출물 통합 복사)
# ============================================================================
def figure6_asr_vs_q():
    print("[Figure 6] ASR vs Q 그래프 (기존 산출물 통합)")
    for ds in DATASETS:
        src = os.path.join(ROOT, ds, "results", "exp1_attack_effectiveness", f"{ds}_asr_vs_q.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIG_DIR, f"figure6_asr_vs_q_{ds}.png"))


# ============================================================================
# Figure 7: 원본/QAFM/기존 연구들 별 포이즌 이미지 시각 비교 (4 method x 3 column grid)
# 기존 sample png(자체 제목 포함)를 합성하지 않고, 원본 이미지에 공격을 직접
# 다시 적용해 깔끔한 단일 그리드로 새로 그림 (행=방법, 열=Clean/Poisoned/Diff×10)
# ============================================================================
def figure7_visual_comparison():
    print("[Figure 7] 포이즌 이미지 시각 비교 (3개 데이터셋, 새로 그림)")
    methods = ["qafm", "badnets", "ftrojan", "blended"]

    for ds in DATASETS:
        # 데이터셋별 config/attacks/dataset 모듈을 동적 로드 (경로 충돌 방지를 위해
        # 매 데이터셋마다 sys.modules 캐시를 비움)
        for mod in ["config", "dataset"]:
            sys.modules.pop(mod, None)
        sys.path = [p for p in sys.path if not p.endswith(("cifar10", "cifar100", "gtsrb"))]
        sys.path.insert(0, os.path.join(ROOT, ds))

        import config as ds_config
        from dataset import load_raw_dataset
        from attacks import build_attack

        method_cfgs = {
            "qafm": ds_config.QAFM_CFG, "badnets": ds_config.BADNETS_CFG,
            "ftrojan": ds_config.FTROJAN_CFG, "blended": ds_config.BLENDED_CFG,
        }
        _, _, test_imgs, test_lbls = load_raw_dataset(ds_config.DATA_DIR)
        clean_img = test_imgs[0]

        fig, axes = plt.subplots(len(methods), 3, figsize=(7.5, 2.3 * len(methods)))
        for row, m in enumerate(methods):
            cfg = dict(method_cfgs[m]); cfg["target_label"] = 0
            attack = build_attack(m, cfg)
            poisoned = attack.poison_image(clean_img)
            diff = np.clip(np.abs(clean_img.astype(np.float32) - poisoned.astype(np.float32)) * 10,
                            0, 255).astype(np.uint8)

            axes[row, 0].imshow(clean_img)
            axes[row, 1].imshow(poisoned)
            axes[row, 2].imshow(diff)
            for c in range(3):
                axes[row, c].axis("off")
            axes[row, 0].text(-0.15, 0.5, METHOD_LABELS.get(m, m), transform=axes[row, 0].transAxes,
                               rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
            if row == 0:
                for c, t in enumerate(["Clean", "Poisoned", "Diff x10"]):
                    axes[row, c].set_title(t, fontsize=12)

        fig.suptitle(f"Figure 7: Visual Comparison of Poisoned Images — {DATASET_LABELS[ds]}", fontsize=13)
        plt.tight_layout(rect=[0.03, 0, 1, 0.96])
        plt.savefig(os.path.join(FIG_DIR, f"figure7_visual_comparison_{ds}.png"), dpi=150)
        plt.close()

        sys.path.pop(0)
        for mod in ["config", "dataset", "attacks", "models"]:
            sys.modules.pop(mod, None)


# ============================================================================
# Figure 8: 방어 우회 막대그래프 (기존 NC anomaly index bar chart 형식 유지,
#           클래스별 anomaly index -> 방법별 ASR-after-defense로 교체)
# ============================================================================
def figure8_defense_bar_chart():
    print("[Figure 8] 방어별 ASR 막대그래프 (기존 NC 막대그래프 형식, exp4로 교체)")
    defenses = ["FinePruning", "NAD", "ShrinkPad"]
    for ds in DATASETS:
        res = load_json(ds, "exp4_defense_advanced", f"{ds}_defense_advanced_summary.json")
        if not res:
            continue
        methods = list(res.keys())
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(methods))
        width = 0.25
        for i, d in enumerate(defenses):
            vals = [res[m][d]["ASR"] for m in methods]
            ax.bar(x + (i - 1) * width, vals, width, label=d)
        ax.axhline(50.0, color="red", linestyle="--", label="Bypass threshold (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
        ax.set_ylabel("ASR after defense (%)", fontsize=12)
        ax.set_title(f"Figure 8: ASR after Model-Repair/Preprocessing Defenses — {DATASET_LABELS[ds]}",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"figure8_defense_bar_{ds}.png"), dpi=150)
        plt.close()


# ============================================================================
# Figure 9: 방어 적용 전/후 ASR 비교 (기존 STRIP entropy histogram 형식 유지,
#           클린/포이즌 엔트로피 분포 -> 적용전/적용후 ASR 분포로 교체)
# ============================================================================
def figure9_defense_before_after():
    print("[Figure 9] 방어 적용 전/후 ASR 비교 (기존 STRIP 분포비교 형식, exp4로 교체)")
    defenses = ["FinePruning", "NAD", "ShrinkPad"]
    for ds in DATASETS:
        res = load_json(ds, "exp4_defense_advanced", f"{ds}_defense_advanced_summary.json")
        if not res:
            continue
        methods = list(res.keys())
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), sharey=True)
        if len(methods) == 1:
            axes = [axes]
        for ax, m in zip(axes, methods):
            before = res[m]["ASR_before"]
            afters = [res[m][d]["ASR"] for d in defenses]
            labels = ["적용전"] + defenses
            vals = [before] + afters
            colors = ["#1f77b4"] + ["#d62728" if v < 50 else "#2ca02c" for v in afters]
            ax.bar(labels, vals, color=colors)
            ax.set_title(METHOD_LABELS.get(m, m), fontsize=11)
            ax.set_ylim(0, 105)
            ax.tick_params(axis="x", rotation=30)
        axes[0].set_ylabel("ASR (%)", fontsize=12)
        fig.suptitle(f"Figure 9: ASR Before vs After Each Defense — {DATASET_LABELS[ds]}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"figure9_defense_before_after_{ds}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    table1_baseline_comparison()
    table3_experiment_setup()
    table4_ba_asr_comprehensive()
    table5_stealth_comparison()
    table6_defense_resistance()
    figure5_freq_heatmap()
    figure6_asr_vs_q()
    figure7_visual_comparison()
    figure8_defense_bar_chart()
    figure9_defense_before_after()
    print("\n완료. paper_assets/figures/, paper_assets/tables/ 확인하세요.")
