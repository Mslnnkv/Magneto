import os
from typing import Dict, Tuple, Any
import pandas as pd

from magneto.basic_matcher import get_str_similarity_candidates
from magneto.bp_reranker import arrange_bipartite_matches
from magneto.embedding_matcher import DEFAULT_MODELS, EmbeddingMatcher
from magneto.span_embedding_matcher_v2 import SpanEmbeddingMatcherV2
from magneto.span_embedding_matcher import SpanEmbeddingMatcher
from magneto.encoding_modes import SPAN_CONTEXTUAL_ENCODING_MODES
from magneto.embedding_matcher import DEFAULT_MODELS, EmbeddingMatcher
from magneto.span_embedding_matcher import SpanEmbeddingMatcher
from magneto.span_embedding_matcher_v2 import SpanEmbeddingMatcherV2
from magneto.starmie_embedding_matcher import StarmieEmbeddingMatcher
from magneto.llm_reranker import LLMReranker
from magneto.utils.utils import (
    clean_df,
    convert_to_valentine_format,
    get_samples,
    remove_invalid_characters,
)
from magneto.utils.dataframe_table import DataframeTable

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # это для HuggingFace

class Magneto:
    """
    Magneto is a class designed to match columns between two Pandas DataFrames.
    It provides multiple matching strategies, including:
      - String similarity
      - Embedding-based similarity
      - Exact name matching
    Additionally, the bipartite (BP) and GPT-based re-rankers can refine matches.
    """

    DEFAULT_PARAMS = {
        "embedding_model": "mpnet",
        "llm_model": "gpt-4o-mini",
        # Доп. параметры вызова LLM (температура и тд)
        "llm_model_kwargs": {},
        # Режим сериализации колонки в текст перед энкодером:
        "encoding_mode": "header_values_verbose",
        # Как выбирать примеры значений из столбца для сериализации.
        "sampling_mode": "mixed",
        "sampling_size": 10,
        "max_context_columns": 7,
        "include_target_type": True,
        "include_target_values": True,
        "target_start_marker": "<<TARGET_START>>",
        "target_end_marker": "<<TARGET_END>>",
        "target_block_start_marker": "<<TARGET_BLOCK_START>>",
        "target_block_end_marker": "<<TARGET_BLOCK_END>>",
        "col_marker": "<<COL>>",
        "include_header": True,
        "include_values": True,
        "span_max_length": 512,
        "topk": 20,
        # Включать ли строковый матчинг по именам колонок (fuzzy string similarity).
        "include_strsim_matches": False,
        # Включать ли эмбеддинг-матчинг (основной механизм retrieval на SLM).
        "include_embedding_matches": True,
        # Минимальный порог похожести (score) для кандидатов по эмбеддингам.
        "embedding_threshold": 0.1,
        # Включать ли точные совпадения имен колонок (после чистки) как совпадение со score=1.0.
        "include_equal_matches": True,
        # Включать ли BP reranker(bipartite matching): пытается сделать глобально согласованный
        # набор соответствий (ближе к 1 - 1), используя кандидатов из retrieval. (похоже на максимальное взвешенное паросочетание в двудольном графе)
        "use_bp_reranker": True,
        # Включать ли LLM reranker
        "use_gpt_reranker": False,
        # Если True — использовать только GPT reranker без предварительного retrieval.
        # Сейчас False.
        "gpt_only": False,
    }

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize Magneto with default or user-specified parameters.
        Конструктор, в kwargs передать нужные значения.
        Args:
            **kwargs: Overriding parameters for the matching process.

        Attributes:
            params (Dict[str, Any]): A dictionary of current settings 
                controlling Magneto’s behavior (e.g., which strategies to apply).
        """
        # Merge provided kwargs with defaults, use params in case you need more parameters: for ablation, etc
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

    def apply_strsim_matches(self) -> None:
        """
        If 'include_strsim_matches' is True, computes string-similarity 
        scores for each column pair (source, target) and updates self.input_sim_map.
        """
        if self.params["include_strsim_matches"]: #если включили в настройках
            strsim_candidates = get_str_similarity_candidates(
                self.df_source.columns, self.df_target.columns
            ) # передаем только названия. get_str_similarity_candidates возвращает :
            ''''
                ("birth_date", "date_of_birth"): 0.92,
                ("age", "patient_age"): 0.88,
                ...
            '''
            for (source_col, target_col), score in strsim_candidates.items(): #цикл по словарю: пара - ключ, score - значение
                self.input_sim_map[source_col][target_col] = score #словарь словарей, в итоге получили: self.input_sim_map = {
                                                                                                            #"A": {"B": 0.5, "C": 0.7},                                                                                                  #"D": {"E": 0.9}
                                                                                                                    #}
    #
    # def apply_embedding_matches(self) -> None:
    #     """
    #     If 'include_embedding_matches' is True, uses EmbeddingMatcher
    #     to compute similarity scores for column pairs, then updates self.input_sim_map.
    #     """
    #     if not self.params["include_embedding_matches"]:
    #         return
    #     #ЗАМЕНА
    #     embeddingMatcher = EmbeddingMatcher(params=self.params) # класс кот отвечает за то как сериализовать и тд
    #
    #     embedding_candidates = embeddingMatcher.get_embedding_similarity_candidates(
    #         self.df_source, self.df_target
    #     ) # берем схожие колонки
    #     for (col_source, col_target), score in embedding_candidates.items():
    #         self.input_sim_map[col_source][col_target] = score #сложить в общий контейнер

    def apply_embedding_matches(self) -> None:
        """
        If 'include_embedding_matches' is True, uses:
        - EmbeddingMatcher for legacy/simple modes
        - SpanEmbeddingMatcher for v3
        - SpanEmbeddingMatcherV2 for v4
        - StarmieEmbeddingMatcher for v5
        """
        if not self.params["include_embedding_matches"]:
            return

        encoding_mode = self.params["encoding_mode"]

        if encoding_mode == "table_context_window_span":
            embeddingMatcher = SpanEmbeddingMatcher(params=self.params)
        elif encoding_mode == "table_context_window_span_target_block":
            embeddingMatcher = SpanEmbeddingMatcherV2(params=self.params)
        elif encoding_mode == "table_context_window_starmie_marker":
            embeddingMatcher = StarmieEmbeddingMatcher(params=self.params)
        else:
            embeddingMatcher = EmbeddingMatcher(params=self.params)

        embedding_candidates = embeddingMatcher.get_embedding_similarity_candidates(
            self.df_source, self.df_target
        )

        for (col_source, col_target), score in embedding_candidates.items():
            self.input_sim_map[col_source][col_target] = score
            
    def apply_equal_matches(self) -> None:
        """
        If 'include_equal_matches' is True, marks columns with identical 
        cleaned names (lowercased, special chars removed) as perfect matches (score=1.0).
        """
        if self.params["include_equal_matches"]:
            source_cols_cleaned = {
                col: remove_invalid_characters(col.strip().lower())
                for col in self.df_source.columns
            }
            target_cols_cleaned = {
                col: remove_invalid_characters(col.strip().lower())
                for col in self.df_target.columns
            } #создаем 2 словаря с очищенными названиями колонок

            for source_col, cand_source in source_cols_cleaned.items():
                for target_col, cand_target in target_cols_cleaned.items():
                    if cand_source == cand_target:
                        self.input_sim_map[source_col][target_col] = 1.0 #если точное совпадение - ставим вероятность = 1

    def get_top_k_matches(self, col_matches: Dict[str, float]) -> list: # берем k лучших
        """
        Sorts column-to-score mappings and returns the top-K matches.

        Args:
            col_matches (Dict[str, float]): A dictionary of {column_name: similarity_score}.

        Returns:
            List of (column_name, score) tuples, up to the number specified by 'topk'.
        """
        sorted_matches = sorted(
            col_matches.items(), key=lambda item: item[1], reverse=True
        )
        top_k_matches = sorted_matches[: self.params["topk"]]
        return [(col, score) for col, score in top_k_matches]

    def call_llm_reranker(self, source_table: DataframeTable, target_table: DataframeTable, 
                          matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> dict:
        """
        If 'use_gpt_reranker' is True, performs an LLM-based reranking of the matched columns.

        Args:
            source_table (DataframeTable): The wrapper for the source DataFrame.
            target_table (DataframeTable): The wrapper for the target DataFrame.
            matches (Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]): 
                Valentine-style matches to be reranked.

        Returns:
            A dictionary of {source_col: [(target_col, score), ...]} after LLM reranking.
        """
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        reranker = LLMReranker(
            llm_model=self.params["llm_model"],
            **self.params.get("llm_model_kwargs", {})
        )

        source_values = {
            col: get_samples(source_table[col], 10) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col], 10) for col in target_table.columns
        }

        matched_columns = {}
        for entry, score in matches.items():
            source_col = entry[0][1]
            target_col = entry[1][1]
            if source_col not in matched_columns:
                matched_columns[source_col] = [(target_col, score)]
            else:
                matched_columns[source_col].append((target_col, score))

        matched_columns = reranker.rematch(
            source_table,
            target_table,
            source_values,
            target_values,
            matched_columns,
        )
        # на вход: таблицы, примеры значений, и список кандидатов из retrieval
        # на выход: обновленные оценки

        return matched_columns
    
    def apply_strategies_in_order(self, order: Dict[str, int]) -> None:
        # если одна и та же пара встретится в разных методах - то запишется результат последней, тут это можно контролировать
        """
        Applies match strategies in the user-defined order if provided, 
        ignoring any strategies assigned a value of -1.

        Args:
            order (Dict[str, int]): A dictionary mapping strategy names (e.g., 'strsim', 'embedding', 'equal')
                to their execution order (where -1 means "skip this strategy").
        """
        strategy_functions = {
            "strsim": self.apply_strsim_matches,
            "embedding": self.apply_embedding_matches,
            "equal": self.apply_equal_matches,
        }

        order = {k: v for k, v in order.items() if v != -1}
        sorted_strategies = sorted(order.items(), key=lambda item: item[1])

        for strategy, _ in sorted_strategies:
            strategy_functions[strategy]()

    def get_matches(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        """
        The main method for deriving matched columns between the source and target DataFrames.

        Args:
            source (pd.DataFrame): The source DataFrame to be matched.
            target (pd.DataFrame): The target DataFrame to be matched.

        Returns:
            A dictionary (Valentine format) mapping:
                ((source_table_name, source_col), (target_table_name, target_col)) -> score
        """
        source_table = DataframeTable(source, "source")
        target_table = DataframeTable(target, "target")
        self.df_source = clean_df(source)
        self.df_target = clean_df(target)

        if len(self.df_source.columns) == 0 or len(self.df_target.columns) == 0:
            return {}

        # store similarity scores between columns
        # we replace the (col_src: col_tgt:score) entries with scores from "stronger" matchers as we progress

        if self.params["gpt_only"]:
            self.input_sim_map = {col: [] for col in self.df_source.columns}
            # set self.input_sim_map to be all target columns for each source column
            for col in self.df_source.columns:
                self.input_sim_map[col] = [(tgt_col, 0.0) for tgt_col in self.df_target.columns]

            matches = convert_to_valentine_format(
                self.input_sim_map, source_table.name, target_table.name
            )
            matches = self.call_llm_reranker(source_table, target_table, matches) #call llm берет 10 значений
            matches = convert_to_valentine_format(
                matches, source_table.name, target_table.name
            )
        
        else:
            self.input_sim_map = {col: {} for col in self.df_source.columns}
            
            if "strategy_order" in self.params: # если пользователь задал порядок иначе - по умолчанию
                self.apply_strategies_in_order(self.params["strategy_order"])
            else:
                match_strategies = [
                    self.apply_strsim_matches,
                    self.apply_embedding_matches,
                    self.apply_equal_matches,
                ]

                for strategy in match_strategies:
                    strategy()  # runs the strategy and updates the input_sim_map

            # filter top-k matcher per column
            for col_source in self.input_sim_map:
                self.input_sim_map[col_source] = self.get_top_k_matches(
                    self.input_sim_map[col_source]
                )

            matches = convert_to_valentine_format(
                self.input_sim_map, source_table.name, target_table.name
            )
            #Valentine - формат: ключ = ((source_table_name, source_col),
            #                            (target_table_name, target_col)), значение = score.

            if self.params["use_bp_reranker"]: #венгерский алгоритм или что-то такое
                matches = arrange_bipartite_matches(
                    matches,
                    self.df_source,
                    source_table.name,
                    self.df_target,
                    target_table.name,
                )

            if self.params["use_gpt_reranker"]:# или гпт
                print("Applying LLM reranker")
                matches = self.call_llm_reranker(source_table, target_table, matches)
                matches = convert_to_valentine_format(
                    matches, source_table.name, target_table.name
                )

        return matches