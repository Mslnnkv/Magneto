from pathlib import Path

import pandas as pd
from magneto import Magneto


BENCHMARK_DIR = Path("synthetic_hard_context_benchmark")


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
    for source_col, gt_target in gt_map.items():
        total += 1
        preds = pred_map.get(source_col, [])
        if gt_target in preds[:k]:
            correct += 1
    return correct, total, correct / total if total else 0.0


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
    return matcher.get_matches(source_df, target_df)


def main():
    source_df = pd.read_csv(BENCHMARK_DIR / "hard_source.csv")
    target_df = pd.read_csv(BENCHMARK_DIR / "hard_target.csv")
    gt_map = load_ground_truth(BENCHMARK_DIR / "hard_ground_truth.csv")

    results = []

    for mode in ["header_values_verbose", "table_context_window_span"]:
        matches = run_mode(source_df, target_df, mode)
        pred_top1 = topk_map_from_matches(matches, k=1)
        pred_top3 = topk_map_from_matches(matches, k=3)

        c1, total, acc1 = evaluate_topk(pred_top1, gt_map, 1)
        c3, _, acc3 = evaluate_topk(pred_top3, gt_map, 3)

        results.append({
            "mode": mode,
            "n_gt": total,
            "top1_correct": c1,
            "top1_accuracy": acc1,
            "top3_correct": c3,
            "top3_accuracy": acc3,
        })

        print(f"\n=== {mode} ===")
        print("Top-1:", c1, "/", total, "=", acc1)
        print("Top-3:", c3, "/", total, "=", acc3)

        # show a few examples
        shown = 0
        for source_col, gt_target in gt_map.items():
            preds = pred_top3.get(source_col, [])
            print(f"{source_col:10s} | gt={gt_target:10s} | preds={preds}")
            shown += 1
            if shown >= 10:
                break

    results_df = pd.DataFrame(results)
    print("\nFinal results:")
    print(results_df)
    results_df.to_csv(BENCHMARK_DIR / "hard_evaluation_results.csv", index=False)
    print("\nSaved to:", (BENCHMARK_DIR / "hard_evaluation_results.csv").resolve())


if __name__ == "__main__":
    main()