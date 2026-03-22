
from itertools import product
import os
import sys
import json
import pandas as pd
import time
import datetime
from valentine import valentine_match
from valentine.algorithms import Coma
import valentine.algorithms.matcher_results as matcher_results

import warnings
warnings.simplefilter('ignore', FutureWarning)

project_path = os.getcwd()
sys.path.append(os.path.join(project_path))

import algorithms.magneto.magneto as mm
from experiments.benchmarks.benchmark_utils import compute_mean_ranking_reciprocal, create_result_file, record_result
from tqdm import tqdm


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings


def run_valentine_benchmark_one_level(BENCHMARK='valentine', DATASET='musicians', ROOT='./data/valentine/Wikidata/musicians'):

    # Define parameter grid
    encoding_modes = [
        "header_values_default",
        "header_values_prefix",
        "header_values_repeat",
        "header_values_verbose",
        "header_only",
        "header_values_verbose_notype",
        "header_values_columnvaluepair_notype",
        "header_header_values_repeat_notype",
        "header_values_default_notype"
    ]

    sampling_modes = ["random", "frequent", "mixed",
                      "weighted", "priority_sampling", "consistent_sampling"]
    # sampling_modes = [ "frequent"]

    sampling_sizes = [10, 30]

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'encoding_mode', 'sampling_mode', 'sampling_size',
        'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'encoding_mode', 'sampling_mode', 'sampling_size',
        'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'grid_search',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_grid_search_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    create_result_file(results_dir, result_file, HEADER)

    # for folder in os.listdir(ROOT): 
    folders = [folder for folder in os.listdir(ROOT) if folder not in ['.DS_Store', '.ipynb_checkpoints']]
    for folder in tqdm(folders, desc="Processing folders"):

        

        source_file = os.path.join(ROOT, folder, folder+'_source.csv')
        target_file = os.path.join(ROOT, folder, folder+'_target.csv')
        mapping_file = os.path.join(ROOT, folder, folder+'_mapping.json')

        df_source = pd.read_csv(source_file, low_memory=False)
        df_target = pd.read_csv(target_file, low_memory=False)
        ground_truth = extract_matchings(open(mapping_file).read())

        ncols_src = str(df_source.shape[1])
        ncols_tgt = str(df_target.shape[1])
        nrows_src = str(df_source.shape[0])
        nrows_tgt = str(df_target.shape[0])

        nmatches = len(ground_truth)

        # print(ground_truth)

        if len(ground_truth) == 0:
            continue

         # Grid search over all parameter combinations
        for encoding_mode, sampling_mode, sampling_size in product(
            encoding_modes, sampling_modes, sampling_sizes
        ):
            print(f"Testing configuration: {encoding_mode}, {sampling_mode}, {sampling_size}")

            # Initialize matcher with current parameter combination
            matcher = mm.Magneto(
                encoding_mode=encoding_mode,
                sampling_mode=sampling_mode,
                sampling_size=sampling_size,
                include_strsim_matches=False,
                include_embedding_matches=True,
                include_equal_matches=False,
                use_bp_reranker=False,
                use_gpt_reranker=False
            )

            # Run matching
            start_time = time.time()
            matches = valentine_match(df_source, df_target, matcher)
            runtime = time.time() - start_time

            # Calculate metrics
            mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
            all_metrics = matches.get_metrics(ground_truth)

            # Calculate one-to-one metrics
            one2one_matches = matches.one_to_one()
            one2one_metrics = one2one_matches.get_metrics(ground_truth)

            # Record result
            result = [
                BENCHMARK, DATASET, folder+'_source.csv', folder+'_target.csv',
                ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches,
                'Magneto', encoding_mode, sampling_mode, sampling_size,
                runtime, mrr_score,
                all_metrics['Precision'], all_metrics['F1Score'],
                all_metrics['Recall'], all_metrics['PrecisionTop10Percent'],
                all_metrics['RecallAtSizeofGroundTruth'],
                one2one_metrics['Precision'], one2one_metrics['F1Score'],
                one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'],
                one2one_metrics['RecallAtSizeofGroundTruth']
            ]

            record_result(result_file, result)

            print(f"MRR: {mrr_score:.4f}, RecallAtGT {all_metrics['RecallAtSizeofGroundTruth']:.4f} Runtime: {runtime:.2f}s")
        print("Done with ", folder, "\n")


def run_valentine_benchmark_three_levels(BENCHMARK='valentine', DATASET='OpenData', ROOT='./data/valentine/OpenData/'):
    # Define parameter grid
    encoding_modes = [
        "header_values_default",
        "header_values_prefix",
        "header_values_repeat",
        "header_values_verbose",
        "header_only",
        "header_values_verbose_notype",
        "header_values_columnvaluepair_notype",
        "header_header_values_repeat_notype",
        "header_values_default_notype"
    ]

    sampling_modes = ["random", "frequent", "mixed",
                      "weighted", "priority_sampling", "consistent_sampling"]
    # sampling_modes = [ "frequent"]

    sampling_sizes = [10, 30]

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'type', 'table_folder', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'encoding_mode', 'sampling_mode', 'sampling_size',
        'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'grid_search',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_grid_search_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    create_result_file(results_dir, result_file, HEADER)

    # for type in os.listdir(ROOT): 
    types = [type for type in os.listdir(ROOT) if type not in ['.DS_Store', '.ipynb_checkpoints']]
    for type in tqdm(types, desc="Processing types"):

        

        print("Type: ", type)
        table_folders = [folder for folder in os.listdir(os.path.join(ROOT, type)) if folder not in ['.DS_Store', '.ipynb_checkpoints']]

        table_folders = table_folders[:int(0.25 * len(table_folders))]

        for table_folder in tqdm(table_folders, desc=f"Processing table folders in {type}"):

            # if table_folder == '.DS_Store' or table_folder == '.ipynb_checkpoints':
            #     continue

            # print("Table: ", table_folder)

            source_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_source.csv')
            target_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_target.csv')
            mapping_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_mapping.json')

            ground_truth = extract_matchings(open(mapping_file).read())

            df_source = pd.read_csv(source_file, low_memory=False)
            df_target = pd.read_csv(target_file, low_memory=False)

            ncols_src = str(df_source.shape[1])
            ncols_tgt = str(df_target.shape[1])
            nrows_src = str(df_source.shape[0])
            nrows_tgt = str(df_target.shape[0])
            nmatches = len(ground_truth)

            if len(ground_truth) == 0:
                continue

            # Grid search over all parameter combinations
            for encoding_mode, sampling_mode, sampling_size in product(
                encoding_modes, sampling_modes, sampling_sizes
            ):
                print(f"Testing configuration: {encoding_mode}, {sampling_mode}, {sampling_size}")

                # Initialize matcher with current parameter combination
                matcher = mm.Magneto(
                    encoding_mode=encoding_mode,
                    sampling_mode=sampling_mode,
                    sampling_size=sampling_size,
                    include_strsim_matches=False,
                    include_embedding_matches=True,
                    include_equal_matches=False,
                    use_bp_reranker=False,
                    use_gpt_reranker=False
                )

                # Run matching
                start_time = time.time()
                matches = valentine_match(df_source, df_target, matcher)
                runtime = time.time() - start_time

                # Calculate metrics
                mrr_score = compute_mean_ranking_reciprocal(
                    matches, ground_truth)
                all_metrics = matches.get_metrics(ground_truth)

                # Calculate one-to-one metrics
                one2one_matches = matches.one_to_one()
                one2one_metrics = one2one_matches.get_metrics(ground_truth)

                # Record result
                result = [
                    BENCHMARK, DATASET, type, table_folder, table_folder +
                    '_source.csv', table_folder+'_target.csv',
                    ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches,
                    'Magneto', encoding_mode, sampling_mode, sampling_size,
                    runtime, mrr_score,
                    all_metrics['Precision'], all_metrics['F1Score'],
                    all_metrics['Recall'], all_metrics['PrecisionTop10Percent'],
                    all_metrics['RecallAtSizeofGroundTruth'],
                    one2one_metrics['Precision'], one2one_metrics['F1Score'],
                    one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'],
                    one2one_metrics['RecallAtSizeofGroundTruth']
                ]

                record_result(result_file, result)

                print(f"MRR: {mrr_score:.4f}, RecallAtGT {all_metrics['RecallAtSizeofGroundTruth']:.4f} Runtime: {runtime:.2f}s")
            print("Done with ", table_folder, "\n")


if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    # run_valentine_benchmark_one_level()

    # Magellan
    # DATASET = 'Magellan'
    # ROOT = './data/valentine/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # # OpenData
    # run_valentine_benchmark_three_levels()

    # # ChEMBLc
    DATASET = 'ChEMBL'
    ROOT = './data/valentine/ChEMBL/'
    run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # # TPC-DI
    # DATASET = 'TPC-DI'
    # ROOT = './data/valentine/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
