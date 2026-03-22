import pprint

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


from experiments.benchmarks.benchmark_utils import compute_mean_ranking_reciprocal_adjusted, compute_mean_ranking_reciprocal_detail, create_result_file, record_result
import algorithms.magneto.magneto as mm

pp = pprint.PrettyPrinter(indent=4)


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings


def run_valentine_benchmark_one_level(BENCHMARK='valentine', DATASET='musicians', ROOT='./data/valentine/Wikidata/Musicians'):



    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method','runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'use_bp_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_use_bp_reranker_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    create_result_file(results_dir, result_file, HEADER)

    for folder in os.listdir(ROOT):
        if folder == '.DS_Store' or folder == '.ipynb_checkpoints':
            continue

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
        for use_bp in [False, True]:
            
            method = 'Magneto'+('_BP' if use_bp else 'NonBP')
            print(f"Method: {method}")
            
            matcher = mm.Magneto(
                use_bp_reranker=use_bp,    
            )

            # Run matching
            start_time = time.time()
            matches = valentine_match(df_source, df_target, matcher)
            runtime = time.time() - start_time

            # Calculate metrics
            mrr_score = compute_mean_ranking_reciprocal_adjusted(matches, ground_truth)
            all_metrics = matches.get_metrics(ground_truth)

            # Calculate one-to-one metrics
            one2one_matches = matches.one_to_one()
            one2one_metrics = one2one_matches.get_metrics(ground_truth)



            # Record result
            result = [
                BENCHMARK, DATASET, folder+'_source.csv', folder+'_target.csv',
                ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches,
                method,runtime, mrr_score,
                all_metrics['Precision'], all_metrics['F1Score'],
                all_metrics['Recall'], all_metrics['PrecisionTop10Percent'],
                all_metrics['RecallAtSizeofGroundTruth'],
                one2one_metrics['Precision'], one2one_metrics['F1Score'],
                one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'],
                one2one_metrics['RecallAtSizeofGroundTruth']
            ]

            record_result(result_file, result)

            print(f"MRR: {mrr_score:.4f}, {all_metrics['RecallAtSizeofGroundTruth']:4f}, Runtime: {runtime:.2f}s")
            print("\n")


def run_valentine_benchmark_three_levels(BENCHMARK='valentine', DATASET='OpenData', ROOT='./data/valentine/OpenData/'):



   

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'type', 'table_folder', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'use_bp_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_use_bp_reranker_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    create_result_file(results_dir, result_file, HEADER)

    for type in os.listdir(ROOT):
        if type == '.DS_Store' or type == '.ipynb_checkpoints':
            continue

        print("Type: ", type)
        for table_folder in os.listdir(os.path.join(ROOT, type)):

            if table_folder == '.DS_Store' or table_folder == '.ipynb_checkpoints':
                continue

            # print("Table: ", table_folder)

            source_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_source.csv')
            target_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_target.csv')
            mapping_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_mapping.json')

            ground_truth = extract_matchings(open(mapping_file).read())

            # if len(ground_truth) < 2:
            #     continue

            df_source = pd.read_csv(source_file, low_memory=False)
            df_target = pd.read_csv(target_file, low_memory=False)

            # print("GroundTruth")
            # for gt in ground_truth:
            #     print(gt)
            # print("\n")

            # print(ground_truth)

            ncols_src = str(df_source.shape[1])
            ncols_tgt = str(df_target.shape[1])
            nrows_src = str(df_source.shape[0])
            nrows_tgt = str(df_target.shape[0])
            nmatches = len(ground_truth)

            if len(ground_truth) == 0:
                continue

            for use_bp in [False, True]:
            
                method = 'Magneto'+('_BP' if use_bp else 'NonBP')
                
                matcher = mm.Magneto(
                    use_bp_reranker=use_bp,    
                )

                # Run matching
                start_time = time.time()
                matches = valentine_match(df_source, df_target, matcher)
                runtime = time.time() - start_time

                # Calculate metrics
                mrr_score = compute_mean_ranking_reciprocal_adjusted(
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
                    method,
                    runtime, mrr_score,
                    all_metrics['Precision'], all_metrics['F1Score'],
                    all_metrics['Recall'], all_metrics['PrecisionTop10Percent'],
                    all_metrics['RecallAtSizeofGroundTruth'],
                    one2one_metrics['Precision'], one2one_metrics['F1Score'],
                    one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'],
                    one2one_metrics['RecallAtSizeofGroundTruth']
                ]

                record_result(result_file, result)

                print(f"Method: {method}, MRR: {mrr_score:.4f}, {all_metrics['RecallAtSizeofGroundTruth']:.4f} Runtime: {runtime:.2f}s")
            print("\n")


if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    # run_valentine_benchmark_one_level()

    # Magellan
    # DATASET='Magellan'
    # ROOT='./data/valentine/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # OpenData
    run_valentine_benchmark_three_levels()

    # # ChEMBLc
    # DATASET='ChEMBL'
    # ROOT='./data/valentine/ChEMBL/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='./data/valentine/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
