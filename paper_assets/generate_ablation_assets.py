"""
IEEE Access 논문용 그림/표 생성 — Ablation Study 항목 (Table 7,8,9,10 / Figure 10).

이 항목들은 component/k_value/q_train/poison_rate ablation 실험 결과가 필요한데,
대화 시점 기준 전부 미실행(또는 옛 코드로 부분 실행)된 상태임. 그래서 이 스크립트는:
  - 필요한 ablation JSON이 "완전한 형태"로 존재하면 실제 표/그림을 생성
  - 없거나 불완전하면, 무엇이 빠졌는지와 실행할 명령어를 적은 PENDING placeholder를 생성
ablation을 돌린 뒤 이 스크립트를 다시 실행하면 자동으로 실제 결과로 채워짐.

Usage:
    python paper_assets/generate_ablation_assets.py
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_KOREAN_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
if os.path.exists(_KOREAN_FONT_PATH):
    fm.fontManager.addfont(_KOREAN_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_KOREAN_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

from utils.visualization import make_result_table

FIG_DIR = os.path.join(ROOT, "paper_assets", "figures")
TAB_DIR = os.path.join(ROOT, "paper_assets", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DATASETS = ["cifar10", "cifar100", "gtsrb"]
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "gtsrb": "GTSRB"}

ABLATION_K_VALUES = [1, 2, 3, 4, 5]
ABLATION_Q_TRAIN_VALUES = [50, 60, 70, 75, 85, 95]
ABLATION_POISON_RATE_VALUES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
ABLATION_Q_EVAL_VALUES = [100, 90, 75, 50]  # q_train ablation에서 보는 평가 Q값 부분집합


def load_json(dataset: str, *parts) -> dict:
    path = os.path.join(ROOT, dataset, "results", *parts)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pending_table(save_path: str, title: str, missing_msg: str, command: str):
    """ablation 데이터가 없을 때 표 대신 보여줄 placeholder."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.text(0.5, 0.7, f"[PENDING] {title}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.45, missing_msg, ha="center", fontsize=10, color="#d62728")
    ax.text(0.5, 0.2, f"실행: {command}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [PENDING] {save_path}")


# ============================================================================
# Table 7: Component Ablation (QAFM vs Fixed-Delta vs No-JPEG, ASR@Q50)
# ============================================================================
def table7_component_ablation():
    print("[Table 7] Component Ablation")
    expected_variants = {"QAFM", "Fixed-Delta", "No-JPEG"}
    for ds in DATASETS:
        res = load_json(ds, "ablation", f"{ds}_component_ablation.json")
        save_path = os.path.join(TAB_DIR, f"table7_component_ablation_{ds}.png")
        if res and expected_variants.issubset(res.keys()):
            headers = ["BA(%)", "ASR@Q50(%)"]
            data = {v: [f"{res[v]['BA']:.2f}", f"{res[v]['ASR@Q50']:.2f}"] for v in expected_variants}
            make_result_table(data, headers, save_path=save_path,
                               title=f"Table 7: Component Ablation — {DATASET_LABELS[ds]}")
        else:
            have = set(res.keys()) if res else set()
            _pending_table(
                save_path, f"Table 7: Component Ablation — {DATASET_LABELS[ds]}",
                f"필요: {expected_variants}, 현재: {have or '없음'} (옛 코드 결과면 재실행 필요)",
                f"cd {ds} && python experiments/ablation/component_ablation.py"
            )


# ============================================================================
# Table 8: k값 변화 ASR/PSNR 결과표
# ============================================================================
def table8_k_ablation():
    print("[Table 8] k값 Ablation")
    for ds in DATASETS:
        res = load_json(ds, "ablation", f"{ds}_k_ablation.json")
        save_path = os.path.join(TAB_DIR, f"table8_k_ablation_{ds}.png")
        have_keys = {str(k) for k in ABLATION_K_VALUES}
        if res and have_keys.issubset(set(res.keys())):
            headers = ["ASR@Q50(%)", "PSNR(dB)"]
            data = {f"k={k}": [f"{res[str(k)]['ASR@Q50']:.2f}", f"{res[str(k)]['PSNR']:.2f}"]
                    for k in ABLATION_K_VALUES}
            make_result_table(data, headers, save_path=save_path,
                               title=f"Table 8: k-value Ablation — {DATASET_LABELS[ds]}")
        else:
            _pending_table(
                save_path, f"Table 8: k-value Ablation — {DATASET_LABELS[ds]}",
                f"필요: k={ABLATION_K_VALUES} 전부, 현재: {list(res.keys()) if res else '없음'}",
                f"cd {ds} && python experiments/ablation/k_value_ablation.py"
            )


# ============================================================================
# Table 9 / Figure 10: q_train x q_eval ASR 결과 (표 + 꺾은선 그래프)
# ============================================================================
def table9_figure10_qtrain_ablation():
    print("[Table 9 / Figure 10] q_train Ablation")
    for ds in DATASETS:
        res = load_json(ds, "ablation", f"{ds}_qtrain_ablation.json")
        save_path_tab = os.path.join(TAB_DIR, f"table9_qtrain_ablation_{ds}.png")
        save_path_fig = os.path.join(FIG_DIR, f"figure10_qtrain_ablation_{ds}.png")
        have_keys = {str(q) for q in ABLATION_Q_TRAIN_VALUES}
        if res and have_keys.issubset(set(res.keys())):
            headers = [f"ASR@Q{q}(%)" for q in ABLATION_Q_EVAL_VALUES]
            data = {}
            for qt in ABLATION_Q_TRAIN_VALUES:
                row = [f"{res[str(qt)].get(f'ASR@Q{qe}', '-')}" for qe in ABLATION_Q_EVAL_VALUES]
                data[f"q_train={qt}"] = row
            make_result_table(data, headers, save_path=save_path_tab,
                               title=f"Table 9: q_train Ablation — {DATASET_LABELS[ds]}")

            fig, ax = plt.subplots(figsize=(7, 4))
            for qt in ABLATION_Q_TRAIN_VALUES:
                q_evs = sorted([int(k.replace("ASR@Q", "")) for k in res[str(qt)] if k.startswith("ASR@Q")])
                asrs = [res[str(qt)][f"ASR@Q{q}"] for q in q_evs]
                ax.plot(q_evs, asrs, "o-", label=f"q_train={qt}")
            ax.set_xlabel("Q_eval"); ax.set_ylabel("ASR (%)")
            ax.set_title(f"Figure 10: ASR vs Q_eval by q_train — {DATASET_LABELS[ds]}")
            ax.invert_xaxis()  # 100 -> 50 순으로 (다른 ASR-vs-Q 그림들과 동일한 컨벤션)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path_fig, dpi=150)
            plt.close()
        else:
            msg = f"필요: q_train={ABLATION_Q_TRAIN_VALUES} 전부, 현재: {list(res.keys()) if res else '없음'}"
            cmd = f"cd {ds} && python experiments/ablation/q_train_ablation.py"
            _pending_table(save_path_tab, f"Table 9: q_train Ablation — {DATASET_LABELS[ds]}", msg, cmd)
            _pending_table(save_path_fig, f"Figure 10: q_train Ablation — {DATASET_LABELS[ds]}", msg, cmd)


# ============================================================================
# Table 10: Poison Rate 변화 ASR/BA 결과표
# ============================================================================
def table10_poison_rate_ablation():
    print("[Table 10] Poison Rate Ablation")
    for ds in DATASETS:
        res = load_json(ds, "ablation", f"{ds}_poison_rate_ablation.json")
        save_path = os.path.join(TAB_DIR, f"table10_poison_rate_ablation_{ds}.png")
        have_keys = {str(p) for p in ABLATION_POISON_RATE_VALUES}
        if res and have_keys.issubset(set(res.keys())):
            headers = ["BA(%)", "ASR@Q50(%)"]
            data = {f"{p*100:.0f}%": [f"{res[str(p)]['BA']:.2f}", f"{res[str(p)]['ASR@Q50']:.2f}"]
                    for p in ABLATION_POISON_RATE_VALUES}
            make_result_table(data, headers, save_path=save_path,
                               title=f"Table 10: Poison Rate Ablation — {DATASET_LABELS[ds]}")
        else:
            _pending_table(
                save_path, f"Table 10: Poison Rate Ablation — {DATASET_LABELS[ds]}",
                f"필요: poison_rate={ABLATION_POISON_RATE_VALUES} 전부, 현재: {list(res.keys()) if res else '없음'}",
                f"cd {ds} && python experiments/ablation/poison_rate_ablation.py"
            )


if __name__ == "__main__":
    table7_component_ablation()
    table8_k_ablation()
    table9_figure10_qtrain_ablation()
    table10_poison_rate_ablation()
    print("\n완료 — [PENDING] 표시된 항목은 해당 ablation 실행 후 이 스크립트를 다시 돌리면 채워짐.")
