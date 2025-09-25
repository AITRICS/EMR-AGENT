#!/bin/bash
Generated_Mapping_Codes=(
    feature_map_DBmimic3_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBeicu_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBmimic3_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBeicu_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBsicdb_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbTrue
    feature_map_DBmimic3_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbFalse
    feature_map_DBeicu_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbFalse
    feature_map_DBsicdb_Piordb_only_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbFalse
    feature_map_DBmimic3_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbFalse
    feature_map_DBeicu_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbFalse
    feature_map_DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgTrue_obsFalse_fbFalse
    feature_map_DBmimic3_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbTrue
    feature_map_DBeicu_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbTrue
    feature_map_DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620_sgFalse_obsFalse_fbTrue2
    feature_map_DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-haiku-latest_sgTrue_obsFalse_fbTrue
)

DATABASES=(
    "mimic3"
    "eicu"
    "sicdb"
    "mimic3"
    "eicu"
    "sicdb"
    "mimic3"
    "eicu"
    "sicdb"
    "mimic3"
    "eicu"
    "sicdb"
    "mimic3"
    "eicu"
    "sicdb"
    "sicdb"
)

Generated_Result_Path="./results"
Test_Data_Path="./test_data/mapping_dictionary.json"
TRIAL_NUM=3
for i in "${!Generated_Mapping_Codes[@]}"; do
    Generated_Mapping="${Generated_Mapping_Codes[$i]}"
    Database_Name="${DATABASES[$i]}"
    python ./featureMap_evaluation.py \
        --generated-mapping "$Generated_Mapping" \
        --target-database "$Database_Name" \
        --generated-result-path "$Generated_Result_Path" \
        --test-data-path "$Test_Data_Path" \
        --trial-num ${TRIAL_NUM}
done
