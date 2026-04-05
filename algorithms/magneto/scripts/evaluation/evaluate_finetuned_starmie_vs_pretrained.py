from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magneto.evaluation import BenchmarkConfig, ModelConfig, evaluate_many


BENCHMARK_DIR = ROOT / "synthetic_starmie_context_benchmark"


def main():
    benchmark = BenchmarkConfig(
        name="starmie_context",
        benchmark_type="ranking",
        source_path=BENCHMARK_DIR / "starmie_source.csv",
        target_path=BENCHMARK_DIR / "starmie_target.csv",
        ground_truth_path=BENCHMARK_DIR / "starmie_ground_truth.csv",
        topk=10,
        ranking_ks=(1, 3, 5),
    )
    models = [
        ModelConfig(
            label="pretrained_starmie_structured",
            embedding_model="mpnet",
            encoding_mode="table_context_window_starmie_structured",
        ),
        ModelConfig(
            label="finetuned_starmie_structured",
            embedding_model="finetuned_context_window_starmie_structured_mpnet.pth",
            encoding_mode="table_context_window_starmie_structured",
        ),
    ]
    summary_df, errors_df = evaluate_many([benchmark], models)
    print(summary_df.fillna(""))
    summary_df.to_csv(BENCHMARK_DIR / "finetuned_starmie_vs_pretrained_results.csv", index=False)
    errors_df.to_csv(BENCHMARK_DIR / "finetuned_starmie_vs_pretrained_errors.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "finetuned_starmie_vs_pretrained_results.csv").resolve())


if __name__ == "__main__":
    main()
