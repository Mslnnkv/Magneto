from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from magneto.evaluation import BenchmarkConfig, ModelConfig, evaluate_many


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


BENCHMARKS = [
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
        name="ambiguity",
        benchmark_type="ranking",
        source_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_source.csv",
        target_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_target.csv",
        ground_truth_path=ROOT / "synthetic_ambiguity_benchmark" / "ambiguity_ground_truth.csv",
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
]


MODELS = [
    ModelConfig(
        label="baseline_mpnet",
        embedding_model="mpnet",
        encoding_mode="header_values_verbose",
    ),
    ModelConfig(
        label="context_window_span_mpnet",
        embedding_model="mpnet",
        encoding_mode="table_context_window_span",
    ),
    ModelConfig(
        label="finetuned_context_window_span_mpnet",
        embedding_model="finetuned_context_window_span_mpnet.pth",
        encoding_mode="table_context_window_span",
    ),
]


def save_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranking_df = summary_df[summary_df["benchmark_type"] == "ranking"].copy()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    top1_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="top1_accuracy")
    top1_pivot.plot(kind="bar", ax=axes[0], title="Top-1 accuracy by benchmark")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_xlabel("Benchmark")
    axes[0].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    mrr_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="mrr")
    mrr_pivot.plot(kind="bar", ax=axes[1], title="MRR by benchmark")
    axes[1].set_ylabel("MRR")
    axes[1].set_xlabel("Benchmark")
    axes[1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary_df, errors_df = evaluate_many(BENCHMARKS, MODELS)

    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    errors_path = OUTPUT_DIR / "evaluation_errors.csv"
    plot_path = OUTPUT_DIR / "evaluation_summary.png"

    summary_df.to_csv(summary_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    save_plot(summary_df, plot_path)

    print("\nEvaluation summary:")
    print(summary_df.sort_values(["benchmark", "model_label"]).fillna(""))
    print(f"\nSaved summary to: {summary_path.resolve()}")
    print(f"Saved detailed errors to: {errors_path.resolve()}")
    print(f"Saved plot to: {plot_path.resolve()}")


if __name__ == "__main__":
    main()