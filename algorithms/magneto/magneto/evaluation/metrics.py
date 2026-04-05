from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


NormalizedMatchRow = Tuple[str, str, float]


def normalize_matches(matches: Dict[tuple, float]) -> List[NormalizedMatchRow]:
    """
    Convert Magneto/Valentine-style matches into tuples:
    (source_column_name, target_column_name, score).
    """
    rows: List[NormalizedMatchRow] = []
    for pair, score in matches.items():
        source_part, target_part = pair
        source_col = source_part[1]
        target_col = target_part[1]
        rows.append((source_col, target_col, float(score)))
    return rows



def ranking_map_from_matches(matches: Dict[tuple, float]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Group candidate target columns by source column and sort by descending score.
    """
    rows = normalize_matches(matches)
    by_source: Dict[str, List[Tuple[str, float]]] = {}

    for source_col, target_col, score in rows:
        by_source.setdefault(source_col, []).append((target_col, score))

    for source_col, candidates in by_source.items():
        by_source[source_col] = sorted(candidates, key=lambda item: item[1], reverse=True)

    return by_source



def pair_set_from_matches(matches: Dict[tuple, float]) -> Set[Tuple[str, str]]:
    return {(source_col, target_col) for source_col, target_col, _ in normalize_matches(matches)}



def compute_ranking_metrics(
    ranked_predictions: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, str],
    ks: Sequence[int] = (1, 3, 5),
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Metrics for source->single-target benchmarks.

    Returns
    -------
    metrics: dict
        top-k accuracies, MRR, mean rank, coverage.
    errors: list[dict]
        per-source debug rows.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    total = len(ground_truth)
    errors: List[Dict[str, object]] = []

    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    rank_sum = 0.0
    found_rank_count = 0
    covered = 0

    for source_col, gt_target in ground_truth.items():
        predictions = ranked_predictions.get(source_col, [])
        predicted_targets = [target for target, _ in predictions]

        rank = None
        if predicted_targets:
            covered += 1
            if gt_target in predicted_targets:
                rank = predicted_targets.index(gt_target) + 1
                reciprocal_rank_sum += 1.0 / rank
                rank_sum += rank
                found_rank_count += 1

        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1

        errors.append(
            {
                "source_column": source_col,
                "ground_truth_target": gt_target,
                "predicted_top1": predicted_targets[0] if predicted_targets else None,
                "predicted_top3": predicted_targets[:3],
                "predicted_top5": predicted_targets[:5],
                "ground_truth_rank": rank,
                "is_top1_correct": rank == 1,
                "is_top3_correct": rank is not None and rank <= 3,
                "is_found": rank is not None,
            }
        )

    metrics: Dict[str, float] = {
        "n_ground_truth": total,
        "coverage": covered / total if total else 0.0,
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "mean_rank": rank_sum / found_rank_count if found_rank_count else 0.0,
    }

    for k in ks:
        metrics[f"top{k}_correct"] = hits[k]
        metrics[f"top{k}_accuracy"] = hits[k] / total if total else 0.0

    return metrics, errors



def compute_pairset_metrics(
    predicted_pairs: Iterable[Tuple[str, str]],
    ground_truth_pairs: Iterable[Tuple[str, str]],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Metrics for many-to-many / pair-set benchmarks.
    """
    predicted = set(predicted_pairs)
    ground_truth = set(ground_truth_pairs)

    correct = predicted & ground_truth
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted

    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    errors: List[Dict[str, object]] = []
    for source_col, target_col in sorted(false_positives):
        errors.append(
            {
                "source_column": source_col,
                "target_column": target_col,
                "error_type": "false_positive",
            }
        )
    for source_col, target_col in sorted(false_negatives):
        errors.append(
            {
                "source_column": source_col,
                "target_column": target_col,
                "error_type": "false_negative",
            }
        )

    metrics = {
        "predicted": len(predicted),
        "ground_truth": len(ground_truth),
        "correct": len(correct),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, errors
