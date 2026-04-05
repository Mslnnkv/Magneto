from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from magneto.evaluation import BenchmarkConfig, ModelConfig, evaluate_many


OUTPUT_DIR = ROOT / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_MODEL_ORDER = [
    "header_values_verbose",
    "table_context_window_span",
    "table_context_window_starmie_structured",
    "finetuned_context_window_span",
    "finetuned_starmie_structured",
]

PLOT_MODEL_LABELS = {
    "header_values_verbose": "Baseline (no context)",
    "table_context_window_span": "Current contextual",
    "table_context_window_starmie_structured": "Starmie-like",
    "finetuned_context_window_span": "Fine-tuned contextual",
    "finetuned_starmie_structured": "Fine-tuned Starmie-like",
}


def build_benchmarks():
    return [
        BenchmarkConfig(
            name="ambiguity",
            benchmark_type="pair_set",
            source_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_source.csv",
            target_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_target.csv",
            ground_truth_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_ground_truth.csv",
            topk=10,
        ),
        BenchmarkConfig(
            name="context_needed",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_context_needed_benchmark" / "context_needed_source.csv",
            target_path=ROOT / "synthetic_context_needed_benchmark" / "context_needed_target.csv",
            ground_truth_path=ROOT / "synthetic_context_needed_benchmark" / "context_needed_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="hard_context",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_hard_context_benchmark" / "hard_source.csv",
            target_path=ROOT / "synthetic_hard_context_benchmark" / "hard_target.csv",
            ground_truth_path=ROOT / "synthetic_hard_context_benchmark" / "hard_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="heldout_context",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_heldout_context_benchmark" / "heldout_source.csv",
            target_path=ROOT / "synthetic_heldout_context_benchmark" / "heldout_target.csv",
            ground_truth_path=ROOT / "synthetic_heldout_context_benchmark" / "heldout_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="starmie_context",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_starmie_context_benchmark" / "starmie_source.csv",
            target_path=ROOT / "synthetic_starmie_context_benchmark" / "starmie_target.csv",
            ground_truth_path=ROOT / "synthetic_starmie_context_benchmark" / "starmie_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
    ]


def build_models():
    return [
        ModelConfig(label="header_values_verbose", embedding_model="mpnet", encoding_mode="header_values_verbose"),
        ModelConfig(label="table_context_window_target_values", embedding_model="mpnet", encoding_mode="table_context_window_target_values"),
        ModelConfig(label="table_context_window_span", embedding_model="mpnet", encoding_mode="table_context_window_span"),
        ModelConfig(label="table_context_window_starmie_marker", embedding_model="mpnet", encoding_mode="table_context_window_starmie_marker"),
        ModelConfig(label="table_context_window_starmie_structured", embedding_model="mpnet", encoding_mode="table_context_window_starmie_structured"),
        ModelConfig(
            label="finetuned_context_window_span",
            embedding_model="finetuned_context_window_span_mpnet.pth",
            encoding_mode="table_context_window_span",
        ),
        ModelConfig(
            label="finetuned_starmie_structured",
            embedding_model="finetuned_context_window_starmie_structured_mpnet.pth",
            encoding_mode="table_context_window_starmie_structured",
        ),
    ]


def save_plot(summary_df, output_path: Path):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    ranking_df = summary_df[summary_df["benchmark_type"] == "ranking"].copy()
    ranking_df = ranking_df[ranking_df["model_label"].isin(PLOT_MODEL_ORDER)].copy()

    if ranking_df.empty:
        plt.close(fig)
        return

    ranking_df["model_label"] = ranking_df["model_label"].map(PLOT_MODEL_LABELS).fillna(ranking_df["model_label"])

    benchmark_order = ["context_needed", "hard_context", "heldout_context", "starmie_context"]
    ranking_df["benchmark"] = pd.Categorical(
        ranking_df["benchmark"],
        categories=benchmark_order,
        ordered=True,
    )
    ranking_df = ranking_df.sort_values(["benchmark", "model_label"])

    top1_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="top1_accuracy")
    top1_pivot.plot(kind="bar", ax=axes[0, 0], title="Top-1 Accuracy")
    axes[0, 0].set_ylabel("Top-1")
    axes[0, 0].set_xlabel("Benchmark")

    top3_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="top3_accuracy")
    top3_pivot.plot(kind="bar", ax=axes[0, 1], title="Top-3 Accuracy")
    axes[0, 1].set_ylabel("Top-3")
    axes[0, 1].set_xlabel("Benchmark")

    top5_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="top5_accuracy")
    top5_pivot.plot(kind="bar", ax=axes[1, 0], title="Top-5 Accuracy")
    axes[1, 0].set_ylabel("Top-5")
    axes[1, 0].set_xlabel("Benchmark")

    mrr_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="mrr")
    mrr_pivot.plot(kind="bar", ax=axes[1, 1], title="MRR")
    axes[1, 1].set_ylabel("MRR")
    axes[1, 1].set_xlabel("Benchmark")

    for ax in axes.flat:
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    benchmarks = build_benchmarks()
    models = build_models()
    summary_df, errors_df = evaluate_many(benchmarks, models)

    print(summary_df.fillna(""))
    summary_path = OUTPUT_DIR / "all_synthetic_benchmarks_summary.csv"
    errors_path = OUTPUT_DIR / "all_synthetic_benchmarks_errors.csv"
    plot_path = OUTPUT_DIR / "all_synthetic_benchmarks_summary.png"

    summary_df.to_csv(summary_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    save_plot(summary_df, plot_path)

    print("\nSaved summary to:", summary_path.resolve())
    print("Saved errors to:", errors_path.resolve())
    print("Saved plot to:", plot_path.resolve())


if __name__ == "__main__":
    main()
