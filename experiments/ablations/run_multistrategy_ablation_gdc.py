import os
import sys
import pandas as pd
import time
import datetime

from valentine import valentine_match

project_path = os.getcwd()
sys.path.append(os.path.join(project_path))


from experiments.benchmarks.benchmark_utils import compute_mean_ranking_reciprocal, create_result_file, record_result
import algorithms.magneto.magneto as mm

def run_grid_search_experiment(BENCHMARK='gdc_studies', DATASET='gdc_studies', ROOT='./data/gdc'):


    encoding_mode = "header_values_repeat"
    sampling_mode = "mixed"
    sampling_size = 10

    strategies_orders = [
        # Single strategies
        {'strsim': 1, 'embedding': -1, 'equal': -1},  # Only strsim
        {'strsim': -1, 'embedding': 1, 'equal': -1},  # Only embedding
        {'strsim': -1, 'embedding': -1, 'equal': 1},  # Only equal

        # # Pairs of strategies
        {'strsim': 1, 'embedding': 2, 'equal': -1},  # strsim, then embedding
        {'strsim': 2, 'embedding': 1, 'equal': -1},  # embedding, then strsim
        {'strsim': 1, 'embedding': -1, 'equal': 2},  # strsim, then equal
        {'strsim': 2, 'embedding': -1, 'equal': 1},  # equal, then strsim
        {'strsim': -1, 'embedding': 1, 'equal': 2},  # embedding, then equal
        {'strsim': -1, 'embedding': 2, 'equal': 1},  # equal, then embedding

        # # All three strategies in different orders
        # strsim, then embedding, then equal
        {'strsim': 1, 'embedding': 2, 'equal': 3},
        # embedding, then strsim, then equal
        {'strsim': 2, 'embedding': 1, 'equal': 3},
        # embedding, then equal, then strsim
        {'strsim': 2, 'embedding': 3, 'equal': 1},
        # strsim, then equal, then embedding
        {'strsim': 1, 'embedding': 3, 'equal': 2},
        # equal, then strsim, then embedding
        {'strsim': 3, 'embedding': 1, 'equal': 2},
        # equal, then embedding, then strsim
        {'strsim': 3, 'embedding': 2, 'equal': 1}
    ]

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'encoding_mode', 'sampling_mode', 'sampling_size',
        'strsim', 'emebedding', 'equal',
        'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'grid_search_strategies',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_grid_search_strategies_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    )
    create_result_file(results_dir, result_file, HEADER)

    # Load target data
    target_file = os.path.join(
        ROOT, 'target-tables', 'gdc_unique_columns_concat_values.csv')
    df_target = pd.read_csv(target_file, low_memory=False)

    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    # Process each ground truth file
    for gt_file in os.listdir(gt_path):
        if not gt_file.endswith('.csv'):
            continue

        # if gt_file != 'Krug.csv':
        #     continue

        print(f"\nProcessing {gt_file}")

        # Load source data and ground truth
        source_file = os.path.join(studies_path, gt_file)
        df_source = pd.read_csv(source_file)

        gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
        gt_df.dropna(inplace=True)
        ground_truth = list(gt_df.itertuples(index=False, name=None))

        # Get dataset dimensions
        ncols_src = str(df_source.shape[1])
        ncols_tgt = str(df_target.shape[1])
        nrows_src = str(df_source.shape[0])
        nrows_tgt = str(df_target.shape[0])
        nmatches = len(ground_truth)

        # Grid search over all parameter combinations
        for strategy_order in strategies_orders:
            

            # Initialize matcher with current parameter combination
            matcher = mm.Magneto(
                encoding_mode=encoding_mode,
                sampling_mode=sampling_mode,
                sampling_size=sampling_size,
                include_strsim_matches=True,
                include_embedding_matches=True,
                include_equal_matches=True,
                use_bp_reranker=False,
                use_gpt_reranker=False,
                strategy_order=strategy_order
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

            strsim = str(strategy_order['strsim'] ) 
            emebedding= str(strategy_order['embedding'] )
            equal= str(strategy_order['equal'])

            print(f"Testing strsim: {strsim}, embedding: {emebedding}, equal: {equal}")

            # Record result
            result = [
                BENCHMARK, DATASET, 'gdc_table', gt_file,
                ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches,
                'Magneto', encoding_mode, sampling_mode, sampling_size,
                strsim, emebedding, equal,
                runtime, mrr_score,
                all_metrics['Precision'], all_metrics['F1Score'],
                all_metrics['Recall'], all_metrics['PrecisionTop10Percent'],
                all_metrics['RecallAtSizeofGroundTruth'],
                one2one_metrics['Precision'], one2one_metrics['F1Score'],
                one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'],
                one2one_metrics['RecallAtSizeofGroundTruth']
            ]

            record_result(result_file, result)

            print(f"MRR: {mrr_score:.4f}, Runtime: {runtime:.2f}s")


if __name__ == '__main__':
    run_grid_search_experiment()
