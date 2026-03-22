
import pprint

from tqdm import tqdm
import os
import sys
import json
import pandas as pd
import time
import datetime
from valentine import valentine_match
from valentine.algorithms import Coma
import valentine.algorithms.matcher_results as matcher_results
import random

import warnings
warnings.simplefilter('ignore', FutureWarning)


project_path = os.getcwd()
sys.path.append(os.path.join(project_path))


import algorithms.gpt_matcher.gpt_matcher as gpt_matcher
import algorithms.magneto.magneto as mm
from experiments.benchmarks.benchmark_utils import compute_mean_ranking_reciprocal, create_result_file, record_result
from tqdm import tqdm


pp = pprint.PrettyPrinter(indent=4)


def extract_matchings(json_data):
    data = json.loads(json_data)
    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings


def get_gpt_method(method):
    if method == 'GPTMatcherSchemaOrder':
        return gpt_matcher.GPTMatcher()
    elif method == 'GPTMatcherRandomOrder':
        return gpt_matcher.GPTMatcher(random_order=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_magneto_matcher(method):
    if method=="Magneto":
        return mm.Magneto()
    elif method=="MagnetoBP":
        return mm.Magneto(use_bp_reranker=True)
    elif method.startswith('MagnetoGPT'):
        print("Method: ", method)
        topk = int(method.split('_')[1])
        return mm.Magneto(use_gpt_reranker=True, topk=topk)

def get_matcher(method):
    if method.startswith('GPT'):
        return get_gpt_method(method)
    elif method.startswith('Magneto'):
        return get_magneto_matcher(method)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_valentine_benchmark_one_level(BENCHMARK='valentine', DATASET='musicians', ROOT='./data/valentine/Wikidata/Musicians'):

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'gpt_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_gpt_reranker_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    )
    create_result_file(results_dir, result_file, HEADER)

    # for folder in os.listdir(ROOT):
    folders = [folder for folder in os.listdir(ROOT) if folder not in ['.DS_Store', '.ipynb_checkpoints']]
    for folder in tqdm(folders, desc="Processing folders"):

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

        # matchers = ["GPTMatcher", "GPTMatcherExample", "Magneto"]
        # matchers = ["GPTMatcher", "GPTMatcherExample", "Magneto", "MagnetoBP", "MagnetoGPT_5", "MagnetoGPT_10, MagnetoGPT_20"]
        # matchers = ["Magneto", "MagnetoBP"]
        # matchers = ["MagnetoGPT_5"]
        # matchers = [ "Magneto", "MagnetoBP", "MagnetoGPT_10", "MagnetoGPT_20", "GPTMatcher", "GPTMatcherExample", "GPTMatcher"]

        

        matchers = [ "MagnetoGPT_5"]

        # gptFull = "MagnetoGPT_"+str(ncols_tgt)
        # matchers = ["MagnetoBP", "MagnetoGPT_3","MagnetoGPT_5","MagnetoGPT_10","MagnetoGPT_20",  gptFull, "GPTMatcherSchemaOrder",  "GPTMatcherRandomOrder"]


        for matcher in matchers:

            print("Running matcher: ", matcher)

            method_name = matcher
            matcher = get_matcher(matcher)

            start_time = time.time()

            try:
                matches = valentine_match(df_source, df_target, matcher)
            except Exception as e:
                print(f"Not able to run the matcher because of exception: {e}")
                matches = matcher_results.MatcherResults({})
            # matches = valentine_match(df_source, df_target, matcher)

            end_time = time.time()
            runtime = end_time - start_time

            mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

            all_metrics = matches.get_metrics(ground_truth)

            recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

            print(method_name, " with MRR Score: ", mrr_score,
                  " and RecallAtGT: ", recallAtGT, " and runtime: ", runtime)

            matches = matches.one_to_one()
            one2one_metrics = matches.get_metrics(ground_truth)

            source_file = source_file.split('/')[-1]
            target_file = target_file.split('/')[-1]

            result = [BENCHMARK, DATASET, source_file, target_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                      one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]
            
            record_result(result_file, result)

        # break
def run_valentine_benchmark_three_levels(BENCHMARK='valentine', DATASET='OpenData', ROOT='data/valentine/OpenData/'):
    '''
    Run the valentine benchmark for datasets split on Unionable, View-Unionable, Joinable, Semantically-Joinable
    '''

    HEADER = ['benchmark', 'dataset', 'type', 'source_table', 'target_table', 'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches', 'method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'gpt_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_gpt_reranker_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    )
    create_result_file(results_dir, result_file, HEADER)

   

    for type in os.listdir(ROOT):
        if type == '.DS_Store':
            continue

        # large_table_count=0

        
        print("Type: ", type)
        table_folders = [folder for folder in os.listdir(os.path.join(ROOT, type)) if folder not in ['.DS_Store', '.ipynb_checkpoints']]

        # table_folders = table_folders[:int(0.2 * len(table_folders))]
        table_folders = random.sample(table_folders, int(0.11 * len(table_folders)))
        print("Table Folders: ", len(table_folders))
        

        for table_folder in tqdm(table_folders, desc=f"Processing table folders in {type}"):

            if table_folder == '.DS_Store':
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

            # print(ncols_src, ncols_tgt)

            # if int(ncols_src) > 15 or int(ncols_tgt) > 15:
            #     large_table_count+=1
            


            # matchers = ["GPTMatcherExample"]
            # matchers = [ "Magneto", "MagnetoBP", "MagnetoGPT_5", "MagnetoGPT_10", "MagnetoGPT_20", "GPTMatcher", "GPTMatcherExample"]

            # gptFull = "MagnetoGPT_"+str(ncols_tgt)
            # matchers = [ "MagnetoBP", "MagnetoGPT_20",gptFull, "GPTMatcherExample", "GPTMatcher"]

            gptFull = "MagnetoGPT_"+str(ncols_tgt)
            matchers = ["MagnetoBP", "MagnetoGPT_20",  gptFull, "GPTMatcherSchemaOrder",  "GPTMatcherRandomOrder"]

            for matcher in matchers:
                print("Running matcher: ", matcher)

                method_name = matcher
                matcher = get_matcher(matcher)

                start_time = time.time()

                try:
                    matches = valentine_match(df_source, df_target, matcher)
                except Exception as e:
                    print(
                        f"Not able to run the matcher because of exception: {e}")
                    matches = matcher_results.MatcherResults({})
                # matches = valentine_match(df_source, df_target, matcher)

                end_time = time.time()
                runtime = end_time - start_time
                
                mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
                
                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

                print(method_name, " with MRR Score: ",
                      mrr_score, " and RecallAtGT: ", recallAtGT, " and runtime: ", runtime)

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                source_file = source_file.split('/')[-1]
                target_file = target_file.split('/')[-1]

                result = [BENCHMARK, DATASET, type, source_file, target_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                          one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)
        # print("Large Table Count for ", type, ' is:', large_table_count)

if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    run_valentine_benchmark_one_level()

    # Magellan
    # DATASET='Magellan'
    # ROOT='data/valentine/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # OpenData
    # run_valentine_benchmark_three_levels()

    # # ChEMBLc
    # DATASET='ChEMBL'
    # ROOT='./data/valentine/ChEMBL/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # # # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='./data/valentine/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
