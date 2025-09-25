#!/bin/bash
Generated_Mapping_Codes=(
    BS_REACT__DBmimic3_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620
    BS_REACT__DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620
    BS_SQL__DBeicu_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620
    BS_SQL__DBmimic3_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620
    BS_SQL__DBsicdb_Piordb_and_manual_and_prior_LLMclaude-3-5-sonnet-20240620
    )

DATABASES=(
    "mimic3"
    "sicdb"
    "eicu"
    "mimic3"
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
