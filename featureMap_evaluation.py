import re
import pickle
import json
import argparse
import os
import numpy as np

# If eicu, use only feature name as answer
# If mimic3 and sicdb, use idx and feature name as answer
DB_IDX = {
    "mimic3": "code and name",
    "eicu": "name_only",
    "sicdb": "code and name"
}

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

def compute_feature_level_prf_answer_once(answer_dict, prediction_dict, db='sicdb', trial=2):
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    balanced_accuracy_list = []

    for i in range(trial):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        correct = 0
        
        for feature, preds_trials in prediction_dict.items():
            if feature not in answer_dict[db]:
                continue

            raw = answer_dict[db][feature]
            if isinstance(raw, list) and all(isinstance(x, list) for x in raw):
                raw = [item for sublist in raw for item in sublist]

            gold = list(set(raw))
            preds = preds_trials[i] if i < len(preds_trials) else None
            if len(preds) == 0:
                preds = [None]
            if preds[0] is None and gold[0] is None:
                tn += 1
                continue
            for pred in preds:
                if isinstance(pred, str):
                    pred = pred.lower()
                if isinstance(pred, tuple):
                    if DB_IDX[db] == "name_only":
                        pred = pred[1]
                if pred in gold:
                    tp += 1
                    gold.remove(pred)
                    correct += 1
                else: # pred not in gold:
                    fp += 1
            if gold:
                fn += len(gold)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        total = tp + fp + fn
        accuracy = correct / total if total else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = (tpr + tnr) / 2

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        balanced_accuracy_list.append(balanced_accuracy)
    print(tp, fp, fn)

    return {
        "precision_mean": np.round(np.mean(precision_list), 4),
        "precision_std": np.round(np.std(precision_list), 4),
        "recall_mean": np.round(np.mean(recall_list), 4),
        "recall_std": np.round(np.std(recall_list), 4),
        "f1_mean": np.round(np.mean(f1_list), 4),
        "f1_std": np.round(np.std(f1_list), 4),
        "accuracy_mean": np.round(np.mean(accuracy_list), 4),
        "accuracy_std": np.round(np.std(accuracy_list), 4),
        "balanced_accuracy_mean": np.round(np.mean(balanced_accuracy_list), 4),
        "balanced_accuracy_std": np.round(np.std(balanced_accuracy_list), 4),
    }


def main(args):
    target_database = args.target_database
    all_pkl_files = search_walk_path({
        'path': os.path.join(args.generated_result_path, args.generated_mapping),
        'extension': '.pkl'
    })
    test_data_path = args.test_data_path
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    target_features = list(test_data.keys())
    duration_list = []
    api_run_count_list = []
    mini_api_run_count_list = []
    number_of_observation_list = []
    observation_sql_count_list = []
    prediction_dict = {}
    
    ### Building prediction_dict
    for target_feature in target_features:
        prediction_dict[target_feature] = [[None] for i in range(args.trial_num)]
    for pkl_file in all_pkl_files:
        with open(pkl_file, "rb") as f:
            loaded_data = pickle.load(f)
        number_of_observation_list.append(loaded_data["number_of_observation"])
        observation_sql_count_list.append(loaded_data["observation_sql_count"])
        api_run_count_list.append(loaded_data["api_run_count"])
        mini_api_run_count_list.append(loaded_data["mini_api_run_count"])
        duration_list.append(loaded_data["duration"])
        trial_number = loaded_data["trial_number"]
        feature_name = loaded_data["requested_features"]
        feature_name = feature_name.split(" (Lab-Test)")[0] if "(Lab-Test)" in loaded_data["requested_features"] else loaded_data["requested_features"].split(" (Vital-Sign)")[0]
        prediction = loaded_data["feature_mapping_result"]
        if trial_number < args.trial_num:
            prediction_dict[feature_name][trial_number] = list(prediction)
    ### Building answer_dict
    answer_dict = {"mimic3": {}, "eicu": {}, "sicdb": {}}

    for feature, info in test_data.items():
        for db in ["mimic3", "eicu", "sicdb"]:
            if db not in info["sources"]:
                answer_dict[db][feature] = [[None]]
                continue

            entry = info["sources"][db][0]

            if isinstance(entry.get("idx"), list) and isinstance(entry.get("raw_text"), list):
                if all(isinstance(i, str) for i in entry["idx"]):
                    items = [rt for rt in entry["raw_text"]]
                else:
                    items = [(i, rt) for i, rt in zip(entry["idx"], entry["raw_text"])]
            elif isinstance(entry.get("idx"), list):
                if all(isinstance(i, str) for i in entry["idx"]):
                    items = [entry["raw_text"]]
                else:
                    items = [(i, entry["raw_text"]) for i in entry["idx"]]
            elif isinstance(entry.get("idx"), str):
                items = [entry["raw_text"]]
            elif isinstance(entry.get("idx"), (int, float)):
                items = [(entry["idx"], entry["raw_text"])]
            elif "column" in entry and "table" in entry:
                items = [f"{entry['table']}.{entry['column']}"]
            else:
                items = [[None]]

            answer_dict[db][feature] = items
    def flatten_items(d, db_name):
        result = []
        for v in d[db_name].values():
            # mimic3, sicdb: value는 list of tuple or [None] or tuple
            # eicu: value는 list of str or [None]
            for item in v:
                if item is None:
                    continue
                if isinstance(item, list) and item == [None]:
                    continue
                if isinstance(item, tuple) or isinstance(item, str):
                    result.append(item)
                elif isinstance(item, list):
                    for subitem in item:
                        if subitem is not None:
                            result.append(subitem)
        return result

    mimic3_flat = flatten_items(answer_dict, 'mimic3')
    eicu_flat = flatten_items(answer_dict, 'eicu')
    sicdb_flat = flatten_items(answer_dict, 'sicdb')
    print(len(mimic3_flat))
    print(len(eicu_flat))
    print(len(sicdb_flat))

    result = compute_feature_level_prf_answer_once(answer_dict, prediction_dict, db=target_database, trial=args.trial_num)
    print(result)
    print("Accuracy: ", result["accuracy_mean"], "+-", result["accuracy_std"])
    print("Balanced Accuracy: ", result["balanced_accuracy_mean"], "+-", result["balanced_accuracy_std"])
    print("Duration: ", np.mean(duration_list))
    print("Observation SQL Count: ", np.mean(observation_sql_count_list))
    print("API Run Count: ", np.mean(api_run_count_list))
    print("API Run Count for Matching Process Only: ", np.mean(mini_api_run_count_list))
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-database', type=str, default="eicu", choices = ["mimic3","eicu","sicdb"])
    parser.add_argument('--generated-result-path', type=str, default="/home/destin/LLM4EMR_DB/results")
    parser.add_argument('--test-data-path', type=str, default="/home/destin/LLM4EMR_DB/test_data/mapping_dictionary.json")
    parser.add_argument('--generated-mapping', type=str, default="feature_map_DBeicu_Piordb_and_manual_and_prior_LLMgpt-4o_sgTrue_obsFalse_fbTrue")
    parser.add_argument('--trial-num', type=int, default=2)
    args = parser.parse_args()
    print(f"\n######### Database : {args.target_database}")
    print(f"######### Experiment : {args.generated_mapping}")
    main(args)
