import os
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Set, List
import logging
import argparse
import ast 

def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension and 'evaluation' not in filename:
                searched_list.append(filename)

    if searched_list:
        return searched_list
    else:
        return False
    
def search_walk_path(info):
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    if searched_list:
        return searched_list
    else:
        return False



def load_data(pkl_path: str, csv_path: str, generated_file: str, 
              test_data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load pickle file
    # print("pkl_path: ", pkl_path)
    # print("csv_path: ", csv_path)
    # print("generated_file: ", generated_file)
    # print("test_data_file: ", test_data_file)
    # exit(1)
    pkl_file_path = f'{pkl_path}/{generated_file}'
    gen_list = []
    all_pkl_file = search_walk({'path': pkl_file_path, 'extension': '.pkl'})
    for pkl_file in all_pkl_file:
        pkl_path = f'{pkl_file_path}/{pkl_file}'
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
            gen_list.append(pkl_data)
            
    test_file = os.path.join(csv_path, test_data_file)
    df_csv = pd.read_csv(test_file)
    
    return all_pkl_file, gen_list, df_csv

def get_patient_ids(df: pd.DataFrame) -> Set[str]:
    """Extract patient IDs from DataFrame"""
    if df.index.name:
        return set(df.index)
    return set(df.iloc[:, 0])

def calculate_id_metrics(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[Dict, List[str]]:
    """Calculate precision, recall, and F1 score for patient ID matching"""
    common_ids = set(list(true_ids.intersection(pred_ids)))
    # print("1o2oeh12hd1290h1209dh")
    # print("common_ids: ", len(common_ids))
    # print("true_ids: ", len(true_ids))
    # print("pred_ids: ", len(pred_ids))
    # exit(1)
    tp = len(common_ids)
    fp = len(pred_ids - common_ids)
    fn = len(true_ids - common_ids)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        'id_precision': prec * 100,  # Convert to percentage
        'id_recall': rec * 100,
        'id_f1': f1 * 100,
        'id_true_positives': tp,
        'id_false_positives': fp,
        'id_false_negatives': fn
    }, common_ids

def calculate_feature_similarity(df_pred: pd.DataFrame, 
                               df_true: pd.DataFrame,
                               common_ids: list,
                               patient_id_column_name: str) -> Dict:
    print("df_pred shape:", df_pred.shape)
    print("df_true shape:", df_true.shape)
    df_pred_common = df_pred[df_pred[patient_id_column_name].isin(common_ids)].sort_values(by=patient_id_column_name).reset_index(drop=True)
    df_true_common = df_true[df_true[patient_id_column_name].isin(common_ids)].sort_values(by=patient_id_column_name).reset_index(drop=True)
    df_pred_common = df_pred_common.drop_duplicates(subset=[patient_id_column_name]).reset_index(drop=True)
    print("df_pred_common shape:", df_pred_common.shape)
    print("df_true_common shape:", df_true_common.shape)
    print("common_ids length:", len(common_ids))

    # Align columns
    common_cols = list(set(df_pred_common.columns) & set(df_true_common.columns))
    if not common_cols:
        return {'feature_similarity': {}}
    else:
        df_pred = df_pred_common[common_cols]
        df_true = df_true_common[common_cols]
        feature_metrics = {}
        total_cells = len(df_pred)  # number of rows (patients)
        for col in common_cols:
            # Special handling for LOS/los and age/AGE/AgeOnAdmission columns
            if col.lower() == 'los' or col == 'LOS':
                pred_vals = df_pred[col].astype(float).round(2)
                true_vals = df_true[col].astype(float).round(2)
            elif col.lower() == 'age' or col == 'AGE' or col == 'AgeOnAdmission':
                pred_vals = np.floor(df_pred[col].astype(float)).astype(int)
                true_vals = np.floor(df_true[col].astype(float)).astype(int)
            else:
                pred_vals = df_pred[col].astype(str)
                true_vals = df_true[col].astype(str)
            # Convert to string for consistent comparison
            pred_vals = pred_vals.astype(str)
            true_vals = true_vals.astype(str)
            # Calculate exact matches
            # exact_matches = (pred_vals == true_vals).sum()
            pred_arr = np.atleast_1d(np.asarray(pred_vals))
            true_arr = np.atleast_1d(np.asarray(true_vals))
            if pred_arr.shape == true_arr.shape:
                exact_matches = (pred_arr == true_arr).sum()
                
            match_percentage = (exact_matches / total_cells) * 100
            # Get examples of mismatches
            mismatches = []
            mismatch_mask = (pred_vals != true_vals)
            if any(mismatch_mask):
                mismatch_examples = pd.DataFrame({
                    'patient_id': df_pred[patient_id_column_name][mismatch_mask].values,
                    'predicted': pred_vals[mismatch_mask],
                    'actual': true_vals[mismatch_mask]
                })
                mismatches = mismatch_examples.head(5).to_dict('records')
            feature_metrics[col] = {
                'exact_match_count': int(exact_matches),
                'total_count': total_cells,
                'exact_match_percentage': match_percentage,
                'mismatch_examples': mismatches
            }
        
        # Calculate overall exact match percentage
        total_matches = sum(m['exact_match_count'] for m in feature_metrics.values())
        total_cells = sum(m['total_count'] for m in feature_metrics.values())
        overall_match_percentage = (total_matches / total_cells) * 100 if total_cells > 0 else 0
        
        return {
            'feature_similarity': feature_metrics,
            'overall_exact_match_percentage': overall_match_percentage
        }

def evaluate_cohorts(gen_data, df_csv):
    all_metrics = []
    obs_number = 0
    api_number = 0
    duration = 0

    current_metrics = {}
    
    # Check if cohort_feature exists
    if 'cohort_feature' in gen_data and gen_data['cohort_feature']:
        # Convert pickle data to DataFrame
        df_pkl = pd.DataFrame(gen_data['cohort_feature'])
        column_set= ast.literal_eval(args.column_set)
        patient_id_column_name = column_set[0]
        df_pkl.columns = column_set
        df_csv = pd.DataFrame(df_csv)
        # Basic count comparison
        current_metrics['pkl_count'] = len(df_pkl)
        current_metrics['csv_count'] = len(df_csv)
        
        # Get patient IDs from both datasets
        pkl_ids = set(df_pkl[patient_id_column_name])
        csv_ids = set(df_csv[patient_id_column_name])

        # Calculate ID-based metrics
        id_metrics, common_ids = calculate_id_metrics(csv_ids, pkl_ids)
        current_metrics.update(id_metrics)
        feature_metrics = calculate_feature_similarity(df_pkl, df_csv, common_ids, patient_id_column_name)
        current_metrics.update(feature_metrics)
    
    else:
        # Set all metrics to 0 for failed cases
        current_metrics.update({
            'pkl_count': 0,
            'csv_count': len(df_csv),
            'id_precision': 0,
            'id_recall': 0,
            'id_f1': 0,
            'id_true_positives': 0,
            'id_false_positives': 0,
            'id_false_negatives': len(df_csv),
            'overall_exact_match_percentage': 0,
            'feature_similarity': {
                'gender': {'exact_match_count': 0, 'total_count': len(df_csv), 'exact_match_percentage': 0, 'mismatch_examples': []},
                'age': {'exact_match_count': 0, 'total_count': len(df_csv), 'exact_match_percentage': 0, 'mismatch_examples': []},
                'mortality': {'exact_match_count': 0, 'total_count': len(df_csv), 'exact_match_percentage': 0, 'mismatch_examples': []},
                'los': {'exact_match_count': 0, 'total_count': len(df_csv), 'exact_match_percentage': 0, 'mismatch_examples': []}
            }
        })

    obs_number += gen_data.get('number_of_observation', 0)
    api_number += gen_data.get('api_run_count', 0)
    duration += gen_data.get('duration', 0)
    all_metrics.append(current_metrics)
    
    # Calculate average metrics
    final_metrics = {}
    if all_metrics:
        # Average numeric metrics
        numeric_keys = ['pkl_count', 'csv_count', 'id_precision', 'id_recall', 
                       'id_f1', 'id_true_positives', 'id_false_positives', 
                       'id_false_negatives', 'overall_exact_match_percentage']
        
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                final_metrics[key] = sum(values) / len(values)
                final_metrics[key + "_std"] = np.std(values)
        
        # Average feature similarity metrics
        feature_similarity = {}
        for metric in all_metrics:
            if 'feature_similarity' in metric:
                for feat, feat_metrics in metric['feature_similarity'].items():
                    if feat not in feature_similarity:
                        feature_similarity[feat] = []
                    feature_similarity[feat].append(feat_metrics)
        
        final_feature_similarity = {}
        for feat, feat_metrics_list in feature_similarity.items():
            final_feature_similarity[feat] = {
                'exact_match_count': sum(m['exact_match_count'] 
                    for m in feat_metrics_list) / len(feat_metrics_list),
                'total_count': sum(m['total_count'] 
                    for m in feat_metrics_list) / len(feat_metrics_list),
                'exact_match_percentage': sum(m['exact_match_percentage'] 
                    for m in feat_metrics_list) / len(feat_metrics_list),
                'mismatch_examples': feat_metrics_list[0]['mismatch_examples']
            }
        
        final_metrics['feature_similarity'] = final_feature_similarity
    
    final_metrics['total_observation_count'] = obs_number
    final_metrics['total_api_count'] = api_number
    final_metrics['total_duration'] = duration
    
    return final_metrics


def main(args):
    # Load data
    predicted_data, gen_list, eval_data = load_data(
        args.generated_cohort_path,
        args.test_data_path,
        args.generated_cohort_feature,
        args.test_data_file)
    
    final_results = {}
    for idx, predicted_single_file in enumerate(predicted_data):
        # if "result_" in predicted_single_file:
        #     print(f"Skipping {predicted_single_file} because it already exists")
        #     continue
        metrics = evaluate_cohorts(gen_list[idx], eval_data)
        print("\n\n============================================================")
        print(f"Predicted Files: {predicted_single_file} for {args.test_data_file}")
        print("============================================================")
    
        # Log results
        print("=== Patient ID Matching Metrics ===")
        print(
            f"Total records - Generated: {metrics['pkl_count']}, "
            f"Test Data: {metrics['csv_count']}"
        )
        print(f"Precision: {metrics['id_precision']:.2f}")
        print(f"Recall: {metrics['id_recall']:.2f}")
        print(f"F1 Score: {metrics['id_f1']:.2f}")
        # print(metrics)
        # continue
        print(f"True Positives: {metrics['id_true_positives']}")
        print(f"False Positives: {metrics['id_false_positives']}")
        print(f"False Negatives: {metrics['id_false_negatives']}")
        final_results["Generated_IDs"] = metrics['pkl_count']
        final_results["Test_IDs"] = metrics['csv_count']
        final_results["Precision"] = metrics['id_precision']
        final_results["Recall"] = metrics['id_recall']
        final_results["F1_Score"] = metrics['id_f1']
        final_results["True_Positives"] = metrics['id_true_positives']
        final_results["False_Positives"] = metrics['id_false_positives']
        final_results["False_Negatives"] = metrics['id_false_negatives']
        final_results['total_observation_count'] = metrics['total_observation_count']
        final_results['total_api_count'] = metrics['total_api_count']
        final_results['total_duration'] = metrics['total_duration']
        
        print("\n=== Feature-wise Exact Matching ===")
        print(f"Overall Exact Match: "f"{metrics['overall_exact_match_percentage']:.2f}")
        final_results['overall_exact_match_percentage'] = metrics['overall_exact_match_percentage']
        
        for feat, feat_metrics in metrics['feature_similarity'].items():
            print(f"Feature: {feat}")
            print(
                f"  Exact Matches: {feat_metrics['exact_match_count']:.1f} / "
                f"{feat_metrics['total_count']:.1f} "
                f"({feat_metrics['exact_match_percentage']:.2f}%)"
            )
            
            if feat_metrics['mismatch_examples']:
                print("  Mismatch Examples (up to 5):")
                for ex in feat_metrics['mismatch_examples']:
                    print(
                        f"Patient {ex['patient_id']}: "
                        f"predicted='{ex['predicted']}' vs "
                        f"actual='{ex['actual']}'"
                    )    
            final_results[feat] = feat_metrics['exact_match_percentage']
        
        pkl_file_path = f'{args.generated_cohort_path}/{args.generated_cohort_feature}'
        if "result_" not in predicted_single_file:
            with open(f'{pkl_file_path}/result_{predicted_single_file}', 'wb') as f:
                pickle.dump(final_results, f)
        else:
            with open(f'{pkl_file_path}/{predicted_single_file}', 'wb') as f:
                pickle.dump(final_results, f)
            
def average_per_db(args):
    all_pkl_files = search_walk_path({'path': args.generated_cohort_path, 'extension': '.pkl'})
    db_pkl_files = list(sorted([i for i in all_pkl_files if f"{args.db_name}" in i and "result_" in i]))
    total_sample_num = args.expected_sample_num
    # Lists to accumulate metrics
    pkl_counts = []
    csv_counts = []
    precisions = []
    recalls = []
    f1_scores = []
    true_positives_list = []
    false_positives_list = []
    false_negatives_list = []
    total_observation_counts = []
    total_api_counts = []
    total_durations = []
    overall_exact_match_percentages = []
    for pkl_file in db_pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        pkl_counts.append(data['Generated_IDs'])
        csv_counts.append(data['Test_IDs'])
        precisions.append(data['Precision'])
        recalls.append(data['Recall'])
        f1_scores.append(data['F1_Score'])
        true_positives_list.append(data['True_Positives'])
        false_positives_list.append(data['False_Positives'])
        false_negatives_list.append(data['False_Negatives'])
        total_observation_counts.append(data['total_observation_count'])
        total_api_counts.append(data['total_api_count'])
        total_durations.append(data['total_duration'])
        overall_exact_match_percentages.append(data['overall_exact_match_percentage'])

    def print_avg_std(name, values):
        arr = np.array(values)
        # Use mean for specific metrics, otherwise use the original logic
        print(f"\n{name}: avg={arr.mean()/100:.4f} ({(arr.std()/8.366)/100:.4f})")
        print("Single score: ", values)

    def print_int_avg_std(name, values):
        arr = np.array(values)
        # Use mean for specific metrics, otherwise use the original logic
        # 8.366 = sqrt(70)
        print(f"\n{name}: avg={arr.mean():.4f} ({(arr.std()/8.366):.4f})")
        print("Single score: ", values)

    # 만약 Generated output 파일 개수가 total_sample_num보다 낮으면 0점으로 채워두기
    def pad_with_zeros(lst):
        if len(lst) < total_sample_num:
            return lst + [0] * (total_sample_num - len(lst))
        return lst

    print("\n=== Average and Std for All Metrics ===")
    print_int_avg_std('Generated_IDs', pkl_counts)
    print_int_avg_std('Test_IDs', csv_counts)
    print_avg_std('precision', pad_with_zeros(precisions))
    print_avg_std('recall', pad_with_zeros(recalls))
    print_avg_std('f1_score', pad_with_zeros(f1_scores))
    # print_int_avg_std('true_positives', true_positives_list)
    # print_int_avg_std('false_positives', false_positives_list)
    # print_int_avg_std('false_negatives', false_negatives_list)
    print_avg_std('overall_exact_match_percentage', pad_with_zeros(overall_exact_match_percentages))
    print_int_avg_std('total_observation_count', total_observation_counts)
    print_int_avg_std('total_api_count', total_api_counts)
    print_int_avg_std('total_duration', total_durations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cohort generation')
    parser.add_argument('--generated-cohort-path', type=str, help='Path to generated cohort features')
    parser.add_argument('--generated-cohort-feature', type=str, help='Name of generated cohort feature')
    parser.add_argument('--test-data-path', type=str, help='Path to test data')
    parser.add_argument('--test-data-file', type=str, help='Name of test data file')
    parser.add_argument('--column-set', type=str, help='Name of column set')
    
    
    parser.add_argument('--db-name', type=str, default='None', choices=['None','eicu', 'mimic3', 'sicdb'])
    parser.add_argument('--expected-sample-num', type=int, help='Expected sample number')
    args = parser.parse_args()
    
    if args.db_name == 'None':
        main(args)
    else:
        average_per_db(args)
