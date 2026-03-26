import random
from pathlib import Path

import pandas as pd


random.seed(42)


def load_gt_pairs(gt_path: str):
    gt = pd.read_csv(gt_path)

    if {"source_column", "target_column"}.issubset(gt.columns):
        return [(row["source_column"], row["target_column"]) for _, row in gt.iterrows()]

    raise ValueError(f"Ground truth file {gt_path} must contain source_column and target_column columns.")


def build_triplets_for_pair(source_csv, target_csv, gt_csv, output_csv, negatives_per_positive=3):
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)
    gt_pairs = load_gt_pairs(gt_csv)

    target_columns = list(target_df.columns)
    gt_map = {src: tgt for src, tgt in gt_pairs}

    rows = []

    for source_col, positive_target in gt_pairs:
        negative_pool = [c for c in target_columns if c != positive_target]

        if not negative_pool:
            continue

        sampled_negatives = random.sample(
            negative_pool,
            k=min(negatives_per_positive, len(negative_pool))
        )

        for negative_target in sampled_negatives:
            rows.append(
                {
                    "source_csv": source_csv,
                    "target_csv": target_csv,
                    "source_col": source_col,
                    "positive_target_col": positive_target,
                    "negative_target_col": negative_target,
                }
            )

    triplets_df = pd.DataFrame(rows)
    triplets_df.to_csv(output_csv, index=False)
    print(f"Saved {len(triplets_df)} triplets to {output_csv}")


def main():
    # Пример: synthetic hard/context-needed benchmark
    tasks = [
        {
            "source_csv": "synthetic_context_needed_benchmark/context_needed_source.csv",
            "target_csv": "synthetic_context_needed_benchmark/context_needed_target.csv",
            "gt_csv": "synthetic_context_needed_benchmark/context_needed_ground_truth.csv",
            "output_csv": "synthetic_context_needed_benchmark/context_needed_triplets.csv",
        },
        {
            "source_csv": "synthetic_hard_context_benchmark/hard_source.csv",
            "target_csv": "synthetic_hard_context_benchmark/hard_target.csv",
            "gt_csv": "synthetic_hard_context_benchmark/hard_ground_truth.csv",
            "output_csv": "synthetic_hard_context_benchmark/hard_triplets.csv",
        },
    ]

    for task in tasks:
        build_triplets_for_pair(**task, negatives_per_positive=4)


if __name__ == "__main__":
    main()