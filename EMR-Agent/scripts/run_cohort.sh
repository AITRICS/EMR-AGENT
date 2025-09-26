#!/bin/bash

COHORT_SELECTIONS=(
"Include only ICU stays from patients aged 19 to 29 and Include only Male patients and Exclude ICU stays with missing discharge time"
"Include only ICU stays from patients aged 61 to 69 and Include only Female patients and Include only ICU stays with at least 30 hours duration"
"Include only ICU stays from patients aged 70 to 89 and Include only Male patients and Include only patients who have only single admission in their lifetime."
"Include only ICU stays from patients aged 20 to 30 and Exclude patient with missing gender information and Include both Female and Male patients"
"Include only ICU stays from patients aged 40 to 55 and include ICU stays which contains at least one clinical recrod of 'Hemoglobin [Mass/volume] in Arterial blood (lab.labname, row:(Hgb))'"
"Include only ICU stays from patients aged 19 to 30 and Include only Male patients and include stays which contains at least 15 clinical recrod of 'Bicarbonate [Moles/volume] in Arterial blood (lab.labname, rows:(bicarbonate, HCO3)'"
"Include only ICU stays from patients aged 55 to 70 and include ICU stays which contains at least one clinical recrod of 'Lactate [Mass/volume] in Arterial blood (lab.labname, row:(lactate)) or Methemoglobin/Hemoglobin.total in Arterial blood (lab.labname, row:(Methemoglobin))'"
"Include only ICU stays from patients aged 19 to 29 and Include only Male patients and Exclude ICU stays with missing discharge time"
"Include only ICU stays from patients aged 61 to 69 and Include only Female patients and Include only ICU stays with at least 30 hours duration"
"Include only ICU stays from patients aged 70 to 89 and Include only Male patients and Include only patients who have only single admission in their lifetime."
"Include only ICU stays from patients aged 20 to 30 and Exclude patient with missing gender information and Include both Female and Male patients"
"Include only ICU stays from patients aged 40 to 55 and include ICU stays which contains at least one clinical recrod of 'Hemoglobin [Mass/volume] in Arterial blood (labevents.itemid, rows:((50811, Hemoglobin), (50855, Absolute Hemoglobin), (51222, Hemoglobin))'"
"Include only ICU stays from patients aged 19 to 30 and Include only Male patients and include stays which contains at least 15 clinical recrod of 'Bicarbonate [Moles/volume] in Arterial blood (labevents.itemid, rows:((50803, Calculated Bicarbonate, Whole Blood), (50882, Bicarbonate))'"
"Include only ICU stays from patients aged 55 to 70 and include ICU stays which contains at least one clinical recrod of 'Lactate [Mass/volume] in Arterial blood (labevents.itemid, row:(50813, Lactate)) or Methemoglobin/Hemoglobin.total in Arterial blood (labevents.itemid, row:(50814, Methemoglobin))'"
"Include only ICU stays from patients aged 19 to 29 and Include only Male patients and Exclude ICU stays with missing discharge time"
"Include only ICU stays from patients aged 61 to 69 and Include only Female patients and Include only ICU stays with at least 30 hours duration"
"Include only ICU stays from patients aged 70 to 89 and Include only Male patients and Include only patients who have only single admission in their lifetime."
"Include only ICU stays from patients aged 20 to 30 and Exclude patient with missing gender information and Include both Female and Male patients"
"Include only ICU stays from patients aged 40 to 55 and include ICU stays which contains at least one clinical recrod of 'Hemoglobin [Mass/volume] in Arterial blood (laboratory.laboratoryid, rows:((288, Hämoglobin (BG) (ZL)), (658, Hämoglobin (BGA)), (289, Hämoglobin (ZL)))'"
"Include only ICU stays from patients aged 19 to 30 and Include only Male patients and include stays which contains at least 15 clinical recrod of 'Bicarbonate [Moles/volume] in Arterial blood (laboratory.laboratoryid, rows:((451, Akt. Bicarbonat (ZL)), (456, Bicarbonat (ZL)), (666, HCO3 act (BGA))))"
"Include only ICU stays from patients aged 55 to 70 and include ICU stays which contains at least one clinical recrod of 'Lactate [Mass/volume] in Arterial blood (laboratory.laboratoryid, row:(454, Lactat (BG) (ZL)), (657, Lactat (BGA)), (465, Lactat (ZL))) or Methemoglobin/Hemoglobin.total in Arterial blood (laboratory.laboratoryid, row:(181, Frakt.Met-Hämoglobin (ZL)), (661, Met-Hämoglobin (BGA)))'"
)
REQUESTED_FEATURES=(
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'length of stay (hours, rounded to 4 decimals in float format)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
        "ICU_stay_id, 'gender (Male/Female/Unknown)', age (integer), 'mortality status (Dead/Alive/Unknown)'"
)
DATABASES=("eicu" "eicu" "eicu" "eicu" "eicu" "eicu" "eicu" "mimic3" "mimic3" "mimic3" "mimic3" "mimic3" "mimic3" "mimic3" "sicdb" "sicdb" "sicdb" "sicdb" "sicdb" "sicdb" "sicdb" ) 

CASE_NAMES=(
    eicu_criteria1
    eicu_criteria2
    eicu_criteria3
    eicu_criteria4_1
    eicu_criteria4_2
    eicu_criteria4_3
    eicu_criteria4_4
    mimic3_criteria1
    mimic3_criteria2
    mimic3_criteria3
    mimic3_criteria4_1
    mimic3_criteria4_2
    mimic3_criteria4_3
    mimic3_criteria4_4
    sicdb_criteria1
    sicdb_criteria2
    sicdb_criteria3
    sicdb_criteria4_1
    sicdb_criteria4_2
    sicdb_criteria4_3
    sicdb_criteria4_4
)
        # claude-3-5-sonnet-20240620
        # claude-3-7-sonnet-20250219
        # claude-3-5-haiku-latest
LLM_MODEL="claude-3-5-sonnet-20240620" 
TRIAL_NUM=10
for trial_idx in $(seq 1 $TRIAL_NUM); do
    for i in "${!COHORT_SELECTIONS[@]}"; do
        COHORT="${COHORT_SELECTIONS[$i]}"
        FEATURES="${REQUESTED_FEATURES[$i]}"
        CASE_NAME="${CASE_NAMES[$i]}"
        DB="${DATABASES[$i]}"
        python ./models/EMR-SQL_V10.py \
            --target-database "$DB" \
            --database-knowledge db_and_manual_and_prior \
            --cohort-selection "$COHORT" \
            --requested-features "$FEATURES" \
            --llm-model "$LLM_MODEL" \
            --trial-num "$trial_idx" \
            --agent-part cohort --schema-guideline --sql-observation --sql-feedback \
            --case-name "$CASE_NAME"
    done
done
