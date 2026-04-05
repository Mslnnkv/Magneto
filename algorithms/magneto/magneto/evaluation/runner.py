from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from magneto import Magneto
from magneto.evaluation.metrics import (
    compute_pairset_metrics,
    compute_ranking_metrics,
    pair_set_from_matches,
    ranking_map_from_matches,
)


@dataclass(frozen=True)
class ModelConfig:
    label: str
    embedding_model: str
    encoding_mode: str
    matcher_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    benchmark_type: str  # "ranking" or "pair_set"
    source_path: Path
    target_path: Path
    ground_truth_path: Path
    topk: int = 10
    ranking_ks: Sequence[int] = (1, 3, 5)


def _load_ranking_ground_truth(path: Path) -> Dict[str, str]:
    gt = pd.read_csv(path)
    return {row["source_column"]: row["target_column"] for _, row in gt.iterrows()}


def _load_pair_ground_truth(path: Path) -> set[Tuple[str, str]]:
    gt = pd.read_csv(path)
    return {(row["source_column"], row["target_column"]) for _, row in gt.iterrows()}


def _build_matcher(model: ModelConfig, benchmark: BenchmarkConfig) -> Magneto:
    matcher_params = {
        "embedding_model": model.embedding_model,
        "encoding_mode": model.encoding_mode,
        "sampling_mode": "mixed",
        "sampling_size": 5,
        "topk": benchmark.topk,
        "use_bp_reranker": False,
        "include_equal_matches": False,
        "max_context_columns": 7,
    }
    matcher_params.update(model.matcher_kwargs)
    return Magneto(**matcher_params)


def _run_matcher(
    matcher: Magneto,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
):
    return matcher.get_matches(source_df, target_df)


def evaluate_benchmark(
    benchmark: BenchmarkConfig,
    model: ModelConfig,
    matcher: Magneto,
):
    source_df = pd.read_csv(benchmark.source_path)
    target_df = pd.read_csv(benchmark.target_path)
    matches = _run_matcher(matcher, source_df, target_df)

    if benchmark.benchmark_type == "ranking":
        ground_truth = _load_ranking_ground_truth(benchmark.ground_truth_path)
        ranked_predictions = ranking_map_from_matches(matches)
        metrics, error_rows = compute_ranking_metrics(
            ranked_predictions=ranked_predictions,
            ground_truth=ground_truth,
            ks=benchmark.ranking_ks,
        )
    elif benchmark.benchmark_type == "pair_set":
        ground_truth = _load_pair_ground_truth(benchmark.ground_truth_path)
        predicted_pairs = pair_set_from_matches(matches)
        metrics, error_rows = compute_pairset_metrics(predicted_pairs, ground_truth)
    else:
        raise ValueError(f"Unsupported benchmark_type: {benchmark.benchmark_type}")

    result_row = {
        "benchmark": benchmark.name,
        "benchmark_type": benchmark.benchmark_type,
        "model_label": model.label,
        "embedding_model": model.embedding_model,
        "encoding_mode": model.encoding_mode,
        **metrics,
    }

    detailed_errors = []
    for row in error_rows:
        detailed_errors.append({"benchmark": benchmark.name, "model_label": model.label, **row})

    return result_row, detailed_errors


def evaluate_many(
    benchmarks: Iterable[BenchmarkConfig],
    models: Iterable[ModelConfig],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    benchmarks = list(benchmarks)
    models = list(models)

    if not benchmarks:
        return pd.DataFrame(), pd.DataFrame()

    for model in models:
        if sum(existing_model.label == model.label for existing_model in models) > 1:
            raise ValueError(f"Duplicate model label: {model.label}")

    for model in models:
        matcher_cache_for_model: Dict[int, Magneto] = {}

        for benchmark in benchmarks:
            if benchmark.topk not in matcher_cache_for_model:
                print(f"Building matcher: {model.label} (topk={benchmark.topk})")
                matcher_cache_for_model[benchmark.topk] = _build_matcher(model, benchmark)

            matcher = matcher_cache_for_model[benchmark.topk]
            result_row, detailed_errors = evaluate_benchmark(benchmark, model, matcher)
            summary_rows.append(result_row)
            error_rows.extend(detailed_errors)

        matcher_cache_for_model.clear()
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary_df = pd.DataFrame(summary_rows)
    errors_df = pd.DataFrame(error_rows)
    return summary_df, errors_df
