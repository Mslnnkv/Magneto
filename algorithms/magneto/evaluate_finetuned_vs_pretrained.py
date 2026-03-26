from pathlib import Path

import pandas as pd
from magneto import Magneto


BENCHMARK_DIR = Path("synthetic_context_needed_benchmark")
SOURCE_PATH = BENCHMARK_DIR / "context_needed_source.csv"
TARGET_PATH = BENCHMARK_DIR / "context_needed_target.csv"
GT_PATH = BENCHMARK_DIR / "context_needed_ground_truth.csv"


def load_ground_truth(path: Path):
    gt = pd.read_csv(path)
    gt_map = {}
    for _, row in gt.iterrows():
        gt_map[row["source_column"]] = row["target_column"]
    return gt_map


def normalize_matches(matches):
    rows = []
    for pair, score in matches.items():
        source_part, target_part = pair
        source_col = source_part[1]
        target_col = target_part[1]
        rows.append((source_col, target_col, score))
    return rows


def topk_map_from_matches(matches, k=3):
    rows = normalize_matches(matches)
    by_source = {}

    for source_col, target_col, score in rows:
        by_source.setdefault(source_col, []).append((target_col, score))

    result = {}
    for source_col, items in by_source.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)
        result[source_col] = [target for target, _ in items[:k]]

    return result


def evaluate_topk(pred_map, gt_map, k):
    total = 0
    correct = 0
    wrong_examples = []

    for source_col, gt_target in gt_map.items():
        total += 1
        preds = pred_map.get(source_col, [])

        if gt_target in preds[:k]:
            correct += 1
        else:
            wrong_examples.append(
                {
                    "source_col": source_col,
                    "gt_target": gt_target,
                    "preds": preds[:k],
                }
            )

    acc = correct / total if total else 0.0
    return correct, total, acc, wrong_examples


def run_model(model_name, source_df, target_df):
    matcher = Magneto(
        embedding_model=model_name,
        encoding_mode="table_context_window_span",
        sampling_mode="mixed",
        sampling_size=5,
        topk=5,
        use_bp_reranker=False,
        include_equal_matches=False,
        max_context_columns=7,
    )
    return matcher.get_matches(source_df, target_df)


def main():
    source_df = pd.read_csv(SOURCE_PATH)
    target_df = pd.read_csv(TARGET_PATH)
    gt_map = load_ground_truth(GT_PATH)

    models = [
        ("pretrained_mpnet", "mpnet"),
        ("finetuned_span_mpnet", "finetuned_context_window_span_mpnet.pth"),
    ]

    all_results = []

    for label, model_name in models:
        print(f"\n{'=' * 90}")
        print(f"MODEL: {label} | {model_name}")
        print(f"{'=' * 90}")

        matches = run_model(model_name, source_df, target_df)

        pred_top1 = topk_map_from_matches(matches, k=1)
        pred_top3 = topk_map_from_matches(matches, k=3)

        c1, total, acc1, wrong1 = evaluate_topk(pred_top1, gt_map, 1)
        c3, _, acc3, wrong3 = evaluate_topk(pred_top3, gt_map, 3)

        print(f"Top-1: {c1}/{total} = {acc1:.4f}")
        print(f"Top-3: {c3}/{total} = {acc3:.4f}")

        print("\nTop-1 wrong examples:")
        for row in wrong1[:10]:
            print(row)

        all_results.append(
            {
                "label": label,
                "model_name": model_name,
                "top1_correct": c1,
                "n_gt": total,
                "top1_accuracy": acc1,
                "top3_correct": c3,
                "top3_accuracy": acc3,
            }
        )

    results_df = pd.DataFrame(all_results)
    print("\nFinal comparison:")
    print(results_df)

    results_df.to_csv(BENCHMARK_DIR / "finetuned_vs_pretrained_results.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "finetuned_vs_pretrained_results.csv").resolve())


if __name__ == "__main__":
    main()