#!/usr/bin/env python3
"""Utility script to generate benchmark comparison bar chart from JSON results.

Usage:
    python scripts/generate_benchmark_plot.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
json_path = REPO / "data" / "eval" / "benchmark_results.json"
out_path = REPO / "submission" / "screenshots" / "07-benchmark-comparison.png"


def main():
    if not json_path.exists():
        print(f"Error: {json_path} does not exist. Run benchmark notebook/script first.")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Task mappings and names
    task_keys = ["ifeval", "gsm8k", "mmlu_sampled", "alpacaeval_lite"]
    display_names = {
        "ifeval": "IFEval",
        "gsm8k": "GSM8K",
        "mmlu_sampled": "MMLU (sampled)",
        "alpacaeval_lite": "AlpacaEval-lite"
    }

    bench_names = [display_names[k] for k in task_keys if k in data]
    sft_scores = [data[k]["sft_only"] for k in task_keys if k in data]
    dpo_scores = [data[k]["sft_dpo"] for k in task_keys if k in data]

    x = np.arange(len(bench_names))
    width = 0.35

    # Color choices matching DPO plots theme (deep blue vs warm red)
    sft_color = "#2e548a"
    dpo_color = "#c83538"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, sft_scores, width, label="SFT-only", color=sft_color, edgecolor="black", linewidth=0.7)
    rects2 = ax.bar(x + width/2, dpo_scores, width, label="SFT+DPO", color=dpo_color, edgecolor="black", linewidth=0.7)

    # Add value labels on top of the bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add delta annotations above each pair
    for i, key in enumerate(task_keys):
        if key in data:
            delta = data[key]["delta"]
            s, d = data[key]["sft_only"], data[key]["sft_dpo"]
            max_val = max(s, d)
            color = "#2e548a" if delta > 0 else "#c83538" if delta < 0 else "#666"
            ax.annotate(f"Δ={delta:+.1f}%",
                        xy=(x[i], max_val + 5),
                        ha="center", va="bottom", fontsize=10, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9))

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Benchmark Comparison: SFT-only vs SFT+DPO", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10, shadow=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Styling details to look premium
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Generated benchmark comparison plot and saved to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
