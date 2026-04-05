from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magneto.evaluation import BenchmarkConfig, ModelConfig, evaluate_many


BENCHMARK_DIR = ROOT / "synthetic_ambiguity_benchmark"


def main():
    benchmark = BenchmarkConfig(
        name="ambiguity",
        benchmark_type="pair_set",
        source_path=BENCHMARK_DIR / "ambiguity_source.csv",
        target_path=BENCHMARK_DIR / "ambiguity_target.csv",
        ground_truth_path=BENCHMARK_DIR / "ambiguity_ground_truth.csv",
        topk=10,
    )
    models = [
        ModelConfig(label="header_values_verbose", embedding_model="mpnet", encoding_mode="header_values_verbose"),
        ModelConfig(label="table_context_window_span", embedding_model="mpnet", encoding_mode="table_context_window_span"),
    ]
    summary_df, errors_df = evaluate_many([benchmark], models)
    print(summary_df.fillna(""))
    summary_df.to_csv(BENCHMARK_DIR / "ambiguity_evaluation_results.csv", index=False)
    errors_df.to_csv(BENCHMARK_DIR / "ambiguity_evaluation_errors.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "ambiguity_evaluation_results.csv").resolve())


if __name__ == "__main__":
    main()
