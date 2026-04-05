from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import random
import pandas as pd


random.seed(42)


def load_gt_rows(gt_path: str):
    gt = pd.read_csv(gt_path)

    if {"source_column", "target_column"}.issubset(gt.columns):
        return gt.to_dict("records")

    raise ValueError(f"Ground truth file {gt_path} must contain source_column and target_column columns.")


def build_role_aware_hard_negatives(gt_df: pd.DataFrame, positive_row: dict):
    if "role" not in gt_df.columns:
        return []

    candidate_mask = gt_df["role"] == positive_row["role"]

    for column in ["entity", "block", "semantic_column"]:
        if column in gt_df.columns and column in positive_row:
            candidate_mask &= gt_df[column] != positive_row[column]

    candidates = gt_df.loc[candidate_mask, "target_column"].drop_duplicates().tolist()
    return [column for column in candidates if column != positive_row["target_column"]]


def build_triplets_for_pair(source_csv, target_csv, gt_csv, output_csv, negatives_per_positive=3):
    target_df = pd.read_csv(target_csv)
    gt_rows = load_gt_rows(gt_csv)
    gt_df = pd.DataFrame(gt_rows)

    target_columns = list(target_df.columns)
    rows = []

    for positive_row in gt_rows:
        source_col = positive_row["source_column"]
        positive_target = positive_row["target_column"]
        negative_pool = [column for column in target_columns if column != positive_target]

        if not negative_pool:
            continue

        sampled_negatives = []

        hard_negative_pool = build_role_aware_hard_negatives(gt_df, positive_row)
        if hard_negative_pool:
            sampled_negatives.extend(
                random.sample(
                    hard_negative_pool,
                    k=min(negatives_per_positive, len(hard_negative_pool)),
                )
            )

        remaining_slots = negatives_per_positive - len(sampled_negatives)
        if remaining_slots > 0:
            remaining_pool = [column for column in negative_pool if column not in sampled_negatives]
            sampled_negatives.extend(
                random.sample(
                    remaining_pool,
                    k=min(remaining_slots, len(remaining_pool)),
                )
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
    tasks = [
        {
            "source_csv": str(ROOT / "synthetic_context_needed_benchmark" / "context_needed_source.csv"),
            "target_csv": str(ROOT / "synthetic_context_needed_benchmark" / "context_needed_target.csv"),
            "gt_csv": str(ROOT / "synthetic_context_needed_benchmark" / "context_needed_ground_truth.csv"),
            "output_csv": str(ROOT / "synthetic_context_needed_benchmark" / "context_needed_triplets.csv"),
        },
        {
            "source_csv": str(ROOT / "synthetic_hard_context_benchmark" / "hard_source.csv"),
            "target_csv": str(ROOT / "synthetic_hard_context_benchmark" / "hard_target.csv"),
            "gt_csv": str(ROOT / "synthetic_hard_context_benchmark" / "hard_ground_truth.csv"),
            "output_csv": str(ROOT / "synthetic_hard_context_benchmark" / "hard_triplets.csv"),
        },
    ]

    for task in tasks:
        build_triplets_for_pair(**task, negatives_per_positive=4)


if __name__ == "__main__":
    main()
