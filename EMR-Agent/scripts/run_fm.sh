#!/bin/bash
DATABASES=("eicu" "mimic3" "sicdb")
# DATABASES=("sicdb")

TARGET_FEATURES=(
    "Heart rate (Vital-Sign)"
    "Core body temperature (Vital-Sign)"
    "Invasive systolic arterial pressure (Vital-Sign)"
    "Invasive diastolic arterial pressure (Vital-Sign)"
    "Invasive mean arterial pressure (Vital-Sign)"
    "Non-invasive systolic arterial pressure (Vital-Sign)"
    "Non-invasive diastolic arterial pressure (Vital-Sign)"
    "Non-invasive mean arterial pressure (Vital-Sign)"
    "Respiratory rate (Vital-Sign)"
    "Base excess in Arterial blood by calculation (Lab-Test)"
    "Hemoglobin [Mass/volume] in Arterial blood (Lab-Test)"
    "Bicarbonate [Moles/volume] in Arterial blood (Lab-Test)"
    "Lactate [Mass/volume] in Arterial blood (Lab-Test)"
    "Methemoglobin/Hemoglobin.total in Arterial blood (Lab-Test)"
    "pH of Arterial blood (Lab-Test)"
    "Carbon dioxide [Partial pressure] in Arterial blood (Lab-Test)"
    "Oxygen [Partial pressure] in Arterial blood (Lab-Test)"
    "Oxygen saturation in Arterial blood (Lab-Test)"
    "Troponin T.cardiac [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Creatine kinase.MB [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Potassium [Moles/volume] in Blood (Lab-Test)"
    "Sodium [Moles/volume] in Blood (Lab-Test)"
    "Chloride [Moles/volume] in Blood (Lab-Test)"
    "Calcium.ionized [Moles/volume] in Blood (Lab-Test)"
    "Calcium [Moles/volume] in Blood (Lab-Test)" 
    "Phosphate [Moles/volume] in Blood (Lab-Test)"
    "Magnesium [Moles/volume] in Blood (Lab-Test)"
    "Urea [Moles/volume] in Venous blood (Lab-Test)"
    "Creatinine [Moles/volume] in Blood (Lab-Test)"
    "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma (Lab-Test)"
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma (Lab-Test)"
    "Bilirubin.total [Moles/volume] in Serum or Plasma (Lab-Test)"
    "Bilirubin.direct [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Alkaline phosphatase [Enzymatic activity/volume] in Blood (Lab-Test)"
    "aPTT in Blood by Coagulation assay (Lab-Test)"
    "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay (Lab-Test)"
    "INR in Blood by Coagulation assay (Lab-Test)"
    "Albumin [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Glucose [Moles/volume] in Serum or Plasma (Lab-Test)"
    "C reactive protein [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Lymphocytes [#/volume] in Blood (Lab-Test)"
    "Neutrophils/100 leukocytes in Blood (Lab-Test)"
    "Band form neutrophils/100 leukocytes in Blood (Lab-Test)"
    "Leukocytes [#/volume] in Blood (Lab-Test)"
    "Platelets [#/volume] in Blood (Lab-Test)"
    "Urea nitrogen [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Cholesterol [Mass/volume] in Serum or Plasma (Lab-Test)"
    "Hematocrit [Volume Fraction] of Blood (Lab-Test)" 
    "Oxygen measurement, partial pressure, arterial (Lab-Test)"
    "Troponin I measurement (Lab-Test)"
    "Partial thromboplastin time ratio (Lab-Test)"
    "Creatine kinase [Mass/volume] in Blood (Lab-Test)"
    "Creatine kinase.MB [Mass/volume] in Blood (Lab-Test)"
    "MCH - Mean corpuscular haemoglobin (Lab-Test)"
    "MCHC [Mass/volume] (Lab-Test)"
    "MCV [Entitic volume] (Lab-Test)"
    )
LLM_MODEL="claude-3-5-sonnet-20240620"
# LLM_MODEL="claude-3-5-haiku-latest"
API_KEY="NONE"
TRIAL_NUM=3
for DB in "${DATABASES[@]}"; do
    for TARGET_FEATURE in "${TARGET_FEATURES[@]}"; do
        for trial_idx in $(seq 0 $TRIAL_NUM); do
            CASE_NUM=0
            python ./models/EMR-SQL_V10.py \
                --api-key "$API_KEY" \
                --target-database "$DB" \
                --database-knowledge db_and_manual_and_prior \
                --llm-model "$LLM_MODEL" \
                --trial-num "$trial_idx" \
                --max-retries 3 \
                --timeseries-feature "$TARGET_FEATURE" \
                --agent-part feature_mapping --schema-guideline --sql-feedback
        done
    done
done
