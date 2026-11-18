"""Generate report-ready visualization figures from real results CSVs (Issue #20).

Reads:
- results/model_comparison.csv
- results/hpo_comparison.csv
- results/hmm_comparison.csv

Writes figures to:
- results/figures/report/
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = "results/figures/report"


def setup_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = 300


def plot_model_accuracy(df_model: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_model))
    width = 0.25

    ax.bar(x - width, df_model["cv_top1"], width, label="CV Top-1", color="#3b82f6", alpha=0.9)
    ax.bar(x, df_model["test_top1"], width, label="Student-Test Top-1", color="#10b981", alpha=0.9)
    ax.bar(x + width, df_model["test_macro_f1"], width, label="Student-Test Macro F1", color="#8b5cf6", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df_model["model"].str.upper(), fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 0.95)
    ax.set_title("Candidate Classifier Comparison: GroupKFold CV vs Student-Testing")
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    # Add values on top of test_top1 bars
    for i, v in enumerate(df_model["test_top1"]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "model_comparison_accuracy.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_model_log_loss(df_model: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_sorted = df_model.sort_values("test_log_loss")
    colors = ["#10b981" if m in ["xgb", "rf", "lgbm"] else "#64748b" for m in df_sorted["model"]]

    bars = ax.barh(df_sorted["model"].str.upper(), df_sorted["test_log_loss"], color=colors, alpha=0.9)
    ax.set_xlabel("Student-Test Binary Log Loss (Lower is Better)")
    ax.set_title("Probability Quality Across Model Families (Student-Test)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, max(df_sorted["test_log_loss"]) * 1.15)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "model_comparison_logloss.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_model_efficiency(df_model: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    # Latency vs training time
    ax.scatter(df_model["train_time_s"], df_model["inference_ms_per_1k_rows"] * 1000, s=160, c="#0284c7", alpha=0.8, edgecolors="black")

    for _, row in df_model.iterrows():
        offset_y = 5 if row["model"] != "knn" else -15
        ax.annotate(row["model"].upper(), (row["train_time_s"], row["inference_ms_per_1k_rows"] * 1000),
                    textcoords="offset points", xytext=(8, offset_y), fontweight="bold", fontsize=11)

    ax.set_xlabel("Full Training Time on 80 Trips (seconds)")
    ax.set_ylabel("Inference Latency (ms per 1,000 candidate pairs)")
    ax.set_yscale("log")
    ax.set_title("Computational Efficiency: Training Time vs Inference Latency (Log Scale)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "model_efficiency.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_hpo_comparison(df_hpo: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Metrics comparison
    labels = ["CV Top-1", "Test Top-1", "Continuity"]
    untuned_vals = [df_hpo.loc[0, "cv_top1"], df_hpo.loc[0, "test_top1"], df_hpo.loc[0, "test_route_continuity"]]
    tuned_vals = [df_hpo.loc[1, "cv_top1"], df_hpo.loc[1, "test_top1"], df_hpo.loc[1, "test_route_continuity"]]

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, untuned_vals, w, label="Untuned", color="#94a3b8")
    ax1.bar(x + w / 2, tuned_vals, w, label="Tuned (Bayesian)", color="#3b82f6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylim(0.75, 0.95)
    ax1.set_ylabel("Accuracy / Continuity Score")
    ax1.set_title("Model Performance: Tuned vs Untuned")
    ax1.legend()

    # Latency & Train time speedup
    bars = ax2.bar(["Untuned", "Tuned (Optuna)"], [df_hpo.loc[0, "train_time_s"], df_hpo.loc[1, "train_time_s"]],
                   color=["#94a3b8", "#10b981"], width=0.5)
    ax2.set_ylabel("Training Time (seconds)")
    ax2.set_title("Training Duration: 19% Speedup via HPO")
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}s", ha="center", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "hpo_tuned_vs_untuned.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_hmm_comparison(df_hmm: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    trips = df_hmm["trip_id"].tolist()
    x = np.arange(len(trips))
    width = 0.35

    # Top-1 Accuracy
    ax1.bar(x - width / 2, df_hmm["classifier_only_top1"], width, label="Classifier-Only Top-1", color="#f97316", alpha=0.85)
    ax1.bar(x + width / 2, df_hmm["viterbi_top1"], width, label="Viterbi HMM Top-1", color="#3b82f6", alpha=0.85)
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_title("Sequence Accuracy Impact: Classifier-Only vs Viterbi HMM Decoding")
    ax1.set_ylim(0.65, 1.02)
    ax1.legend(loc="lower right")

    # Route Continuity
    ax2.bar(x - width / 2, df_hmm["classifier_only_continuity"], width, label="Classifier-Only Continuity", color="#ef4444", alpha=0.85)
    ax2.bar(x + width / 2, df_hmm["viterbi_continuity"], width, label="Viterbi HMM Continuity", color="#10b981", alpha=0.85)
    ax2.set_ylabel("Route Continuity Score")
    ax2.set_title("Route Connectivity Impact: Eliminating Disconnected GPS Jumps")
    ax2.set_ylim(0.65, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(trips, fontweight="bold")
    ax2.set_xlabel("Student-Test Trip ID")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "hmm_route_impact.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    df_model = pd.read_csv("results/model_comparison.csv")
    df_hpo = pd.read_csv("results/hpo_comparison.csv")
    df_hmm = pd.read_csv("results/hmm_comparison.csv")

    print(f"Generating report plots into {OUT_DIR}...")
    plot_model_accuracy(df_model)
    plot_model_log_loss(df_model)
    plot_model_efficiency(df_model)
    plot_hpo_comparison(df_hpo)
    plot_hmm_comparison(df_hmm)
    print("All report figures generated successfully.")


if __name__ == "__main__":
    main()
