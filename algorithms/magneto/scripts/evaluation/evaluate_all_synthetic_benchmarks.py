from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from magneto.evaluation import BenchmarkConfig, ModelConfig, evaluate_many


OUTPUT_DIR = ROOT / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_MODEL_ORDER = [
    "header_values_verbose",
    "table_context_window_starmie_structured",
    "finetuned_starmie_structured",
]

PLOT_MODEL_LABELS = {
    "header_values_verbose": "Magneto",
    "table_context_window_starmie_structured": "Contextual Magneto",
    "finetuned_starmie_structured": "Contextual Magneto (с дообучением)",
}

PLOT_MODEL_DISPLAY_ORDER = [
    "Magneto",
    "Contextual Magneto",
    "Contextual Magneto (с дообучением)",
]

PLOT_BENCHMARK_LABELS = {
    "version_5": "контекстный",
    "version_6": "Starmie-like",
}

COLOR_PLOT_STYLE = {
    "Magneto": {"color": "#1f77b4", "hatch": "", "edgecolor": "black"},
    "Contextual Magneto": {"color": "#ff7f0e", "hatch": "", "edgecolor": "black"},
    "Contextual Magneto (с дообучением)": {"color": "#2ca02c", "hatch": "", "edgecolor": "black"},
}

BW_PLOT_STYLE = {
    "Magneto": {"color": "#d9d9d9", "hatch": "", "edgecolor": "black"},
    "Contextual Magneto": {"color": "#ffffff", "hatch": "///", "edgecolor": "black"},
    "Contextual Magneto (с дообучением)": {"color": "#a6a6a6", "hatch": "xx", "edgecolor": "black"},
}


def build_benchmarks():
    return [
        BenchmarkConfig(
            name="version_1",
            benchmark_type="pair_set",
            source_path=ROOT / "synthetic_benchmark_version_1" / "version_1_source.csv",
            target_path=ROOT / "synthetic_benchmark_version_1" / "version_1_target.csv",
            ground_truth_path=ROOT / "synthetic_benchmark_version_1" / "version_1_ground_truth.csv",
            topk=10,
        ),
        BenchmarkConfig(
            name="version_3",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_benchmark_version_3" / "version_3_source.csv",
            target_path=ROOT / "synthetic_benchmark_version_3" / "version_3_target.csv",
            ground_truth_path=ROOT / "synthetic_benchmark_version_3" / "version_3_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="version_4",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_benchmark_version_4" / "version_4_source.csv",
            target_path=ROOT / "synthetic_benchmark_version_4" / "version_4_target.csv",
            ground_truth_path=ROOT / "synthetic_benchmark_version_4" / "version_4_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="version_5",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_benchmark_version_5" / "version_5_source.csv",
            target_path=ROOT / "synthetic_benchmark_version_5" / "version_5_target.csv",
            ground_truth_path=ROOT / "synthetic_benchmark_version_5" / "version_5_ground_truth.csv",
            topk=10,
            ranking_ks=(1, 3, 5),
        ),
        BenchmarkConfig(
            name="version_6",
            benchmark_type="ranking",
            source_path=ROOT / "synthetic_benchmark_version_6" / "version_6_source.csv",
            target_path=ROOT / "synthetic_benchmark_version_6" / "version_6_target.csv",
            ground_truth_path=ROOT / "synthetic_benchmark_version_6" / "version_6_ground_truth.csv",
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


def save_plot(summary_df, output_path: Path, *, black_and_white: bool = False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    plot_style = BW_PLOT_STYLE if black_and_white else COLOR_PLOT_STYLE

    ranking_df = summary_df[summary_df["benchmark_type"] == "ranking"].copy()
    ranking_df = ranking_df[ranking_df["model_label"].isin(PLOT_MODEL_ORDER)].copy()

    if ranking_df.empty:
        plt.close(fig)
        return

    # Поддерживаем и новый, и старый нейминг benchmark-ов при построении графика,
    # чтобы можно было пересобирать диаграмму из уже существующего summary.csv.
    benchmark_label_map = {
        "heldout_context": "version_5",
        "starmie_context": "version_6",
        "version_5": "version_5",
        "version_6": "version_6",
    }
    ranking_df["benchmark"] = ranking_df["benchmark"].map(benchmark_label_map)
    ranking_df = ranking_df[ranking_df["benchmark"].isin(["version_5", "version_6"])].copy()

    if ranking_df.empty:
        plt.close(fig)
        return

    ranking_df["model_label"] = ranking_df["model_label"].map(PLOT_MODEL_LABELS).fillna(ranking_df["model_label"])
    ranking_df["model_label"] = pd.Categorical(
        ranking_df["model_label"],
        categories=PLOT_MODEL_DISPLAY_ORDER,
        ordered=True,
    )

    benchmark_order = ["version_5", "version_6"]
    ranking_df["benchmark"] = pd.Categorical(
        ranking_df["benchmark"],
        categories=benchmark_order,
        ordered=True,
    )
    ranking_df = ranking_df.sort_values(["benchmark", "model_label"])
    ranking_df["benchmark"] = ranking_df["benchmark"].astype(str).map(
        lambda value: PLOT_BENCHMARK_LABELS.get(value, value)
    )
    displayed_benchmark_order = [
        PLOT_BENCHMARK_LABELS["version_5"],
        PLOT_BENCHMARK_LABELS["version_6"],
    ]

    plot_colors = [plot_style[label]["color"] for label in PLOT_MODEL_DISPLAY_ORDER]

    top5_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="top5_accuracy")
    top5_pivot = top5_pivot.reindex(index=displayed_benchmark_order, columns=PLOT_MODEL_DISPLAY_ORDER)
    top5_pivot.plot(kind="bar", ax=axes[0], title="Top-5 Accuracy", fontsize=14, color=plot_colors)
    axes[0].set_ylabel("Top-5", fontsize=16)
    axes[0].set_xlabel("Benchmark", fontsize=16)

    mrr_pivot = ranking_df.pivot(index="benchmark", columns="model_label", values="mrr")
    mrr_pivot = mrr_pivot.reindex(index=displayed_benchmark_order, columns=PLOT_MODEL_DISPLAY_ORDER)
    mrr_pivot.plot(kind="bar", ax=axes[1], title="MRR", fontsize=14, color=plot_colors)
    axes[1].set_ylabel("MRR", fontsize=16)
    axes[1].set_xlabel("Benchmark", fontsize=16)

    for ax in axes.flat:
        for container, label in zip(ax.containers, PLOT_MODEL_DISPLAY_ORDER):
            style = plot_style[label]
            for patch in container.patches:
                patch.set_facecolor(style["color"])
                patch.set_edgecolor(style["edgecolor"])
                patch.set_linewidth(1.2)
                patch.set_hatch(style["hatch"])

        ax.tick_params(axis="x", labelsize=14, rotation=0)
        ax.tick_params(axis="y", labelsize=14)
        ax.title.set_fontsize(18)
        legend_handles = [
            Patch(
                facecolor=plot_style[label]["color"],
                edgecolor=plot_style[label]["edgecolor"],
                hatch=plot_style[label]["hatch"],
                label=label,
            )
            for label in PLOT_MODEL_DISPLAY_ORDER
        ]
        ax.legend(
            handles=legend_handles,
            title="Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=13,
            title_fontsize=14,
        )

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
    bw_plot_path = OUTPUT_DIR / "all_synthetic_benchmarks_summary_bw.png"

    summary_df.to_csv(summary_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    save_plot(summary_df, plot_path)
    save_plot(summary_df, bw_plot_path, black_and_white=True)

    print("\nSaved summary to:", summary_path.resolve())
    print("Saved errors to:", errors_path.resolve())
    print("Saved plot to:", plot_path.resolve())
    print("Saved black-and-white plot to:", bw_plot_path.resolve())


if __name__ == "__main__":
    main()
