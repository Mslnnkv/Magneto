from pathlib import Path

import pandas as pd
from magneto import Magneto


BENCHMARK_DIR = Path("synthetic_ambiguity_benchmark")


def load_ground_truth(path: Path):
    gt = pd.read_csv(path)
    return {(row["source_column"], row["target_column"]) for _, row in gt.iterrows()}


def run_mode(source_df, target_df, mode_name):
    matcher = Magneto(
        embedding_model="mpnet",
        encoding_mode=mode_name,
        sampling_mode="mixed",
        sampling_size=5,
        topk=5,
        use_bp_reranker=False,
        max_context_columns=7,
    )
    matches = matcher.get_matches(source_df, target_df)

    normalized_pairs = set()
    for pair in matches.keys():
        source_part, target_part = pair
        source_col = source_part[1]
        target_col = target_part[1]
        normalized_pairs.add((source_col, target_col))

    return normalized_pairs, matches


def main():
    source_df = pd.read_csv(BENCHMARK_DIR / "ambiguity_source.csv")
    target_df = pd.read_csv(BENCHMARK_DIR / "ambiguity_target.csv")
    gt = load_ground_truth(BENCHMARK_DIR / "ambiguity_ground_truth.csv")

    results = []

    for mode in ["header_values_verbose", "table_context_window_span"]:
        predicted_pairs, scored_matches = run_mode(source_df, target_df, mode)
        correct = predicted_pairs & gt

        precision = len(correct) / len(predicted_pairs) if predicted_pairs else 0.0
        recall = len(correct) / len(gt) if gt else 0.0

        results.append({
            "mode": mode,
            "predicted": len(predicted_pairs),
            "ground_truth": len(gt),
            "correct": len(correct),
            "precision": precision,
            "recall": recall,
        })

        print(f"\n=== {mode} ===")
        print("Correct count:", len(correct))
        print("Precision:", precision)
        print("Recall:", recall)
        print("Some correct matches:", list(correct)[:15])

    results_df = pd.DataFrame(results)
    print("\nFinal results:")
    print(results_df)

    results_df.to_csv(BENCHMARK_DIR / "ambiguity_evaluation_results.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "ambiguity_evaluation_results.csv").resolve())


if __name__ == "__main__":
    main()