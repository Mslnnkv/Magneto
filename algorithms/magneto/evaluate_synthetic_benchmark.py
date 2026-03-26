from pathlib import Path

import pandas as pd
from magneto import Magneto


BENCHMARK_DIR = Path("synthetic_context_benchmark")


def load_ground_truth(path: Path):
    gt = pd.read_csv(path)
    return {(row["source_column"], row["target_column"]) for _, row in gt.iterrows()}


def run_mode(source_df, target_df, mode_name):
    matcher = Magneto(
        embedding_model="mpnet",
        encoding_mode=mode_name,
        sampling_mode="mixed",
        sampling_size=5,
        topk=10,
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


def evaluate_domain(domain_name: str):
    source_path = BENCHMARK_DIR / f"{domain_name}_source.csv"
    target_path = BENCHMARK_DIR / f"{domain_name}_target.csv"
    gt_path = BENCHMARK_DIR / f"{domain_name}_ground_truth.csv"

    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)
    gt = load_ground_truth(gt_path)

    results = []

    for mode in ["header_values_verbose", "table_context_window_span"]:
        predicted_pairs, scored_matches = run_mode(source_df, target_df, mode)
        correct = predicted_pairs & gt

        precision = len(correct) / len(predicted_pairs) if predicted_pairs else 0.0
        recall = len(correct) / len(gt) if gt else 0.0

        results.append(
            {
                "domain": domain_name,
                "mode": mode,
                "predicted": len(predicted_pairs),
                "ground_truth": len(gt),
                "correct": len(correct),
                "precision": precision,
                "recall": recall,
            }
        )

        print(f"\n=== {domain_name} | {mode} ===")
        print("Ground truth:", gt)
        print("Predicted (first 15):", list(predicted_pairs)[:15])
        print("Correct:", correct)

    return results


def main():
    domains = ["customers", "products", "orders", "employees"]
    all_results = []

    for domain in domains:
        domain_results = evaluate_domain(domain)
        all_results.extend(domain_results)

    results_df = pd.DataFrame(all_results)
    print("\nFinal results:")
    print(results_df)
    results_df.to_csv(BENCHMARK_DIR / "evaluation_results.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "evaluation_results.csv").resolve())


if __name__ == "__main__":
    main()