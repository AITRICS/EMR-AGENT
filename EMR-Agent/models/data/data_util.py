unit_dict = {
    'Heart rate': '/minute',
    'Core body temperature': '℃',
    'Invasive systolic arterial pressure': 'mmHg',
    'Invasive diastolic arterial pressure': 'mmHg',
    'Invasive mean arterial pressure': 'mmHg',
    'Non-invasive systolic arterial pressure': 'mmHg',
    'Non-invasive diastolic arterial pressure': 'mmHg',
    'Non-invasive mean arterial pressure': 'mmHg',
    'End tidal carbon dioxide concentration': 'mmHg',
    'Respiratory rate': '/min',
    'Inspired oxygen concentration': '%',
    'Positive end expiratory pressure setting': 'cmH2O',
    'Expiratory tidal volume': 'ml',
    'Tidal volume setting': 'ml',
    'Plateau pressure': 'cmH2O',
    'Hourly urine volume': 'ml/h',
    'Glasgow Coma Score verbal response subscore': 'Ordinal Score',
    'Glasgow Coma Score motor response subscore': 'Ordinal Score',
    'Glasgow Coma Score eye opening subscore': 'Ordinal Score',
    'Body weight': 'kg',
    'Body height measure': 'cm',
    'Base excess in Arterial blood by calculation': 'mmol/l',
    'Hemoglobin [Mass/volume] in Arterial blood': 'g/L',
    'Bicarbonate [Moles/volume] in Arterial blood': 'mmol/l',
    'Lactate [Mass/volume] in Arterial blood': 'mmol/l',
    'Methemoglobin/Hemoglobin.total in Arterial blood': '%',
    'pH of Arterial blood': '',
    'Carbon dioxide [Partial pressure] in Arterial blood': 'mmHg',
    'Oxygen [Partial pressure] in Arterial blood': 'mmHg',
    'Oxygen saturation in Arterial blood': '%',
    'Troponin T.cardiac [Mass/volume] in Serum or Plasma': 'ng/l',
    'Creatine kinase.MB [Mass/volume] in Serum or Plasma': 'U/l',
    'Potassium [Moles/volume] in Blood': 'mmol/l',
    'Sodium [Moles/volume] in Blood': 'mmol/l',
    'Chloride [Moles/volume] in Blood': 'mmol/l',
    'Calcium.ionized [Moles/volume] in Blood': 'mmol/l',
    'Calcium [Moles/volume] in Blood': 'mmol/l',
    'Phosphate [Moles/volume] in Blood': 'mmol/l',
    'Magnesium [Moles/volume] in Blood': 'mmol/l',
    'Urea [Moles/volume] in Venous blood': 'mmol/l',
    'Creatinine [Moles/volume] in Blood': 'umol/l',
    'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma': 'U/l',
    'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma': 'U/l',
    'Bilirubin.total [Moles/volume] in Serum or Plasma': 'umol/l',
    'Bilirubin.direct [Mass/volume] in Serum or Plasma': 'umol/l',
    'Alkaline phosphatase [Enzymatic activity/volume] in Blood': 'U/l',
    'aPTT in Blood by Coagulation assay': 'Sek',
    'Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay': 'g/L',
    'INR in Blood by Coagulation assay': '',
    'Albumin [Mass/volume] in Serum or Plasma': 'g/L',
    'Glucose [Moles/volume] in Serum or Plasma': 'mmol/l',
    'C reactive protein [Mass/volume] in Serum or Plasma': 'mg/l',
    'Lymphocytes [#/volume] in Blood': 'G/l',
    'Neutrophils/100 leukocytes in Blood': '%',
    'Band form neutrophils/100 leukocytes in Blood': '%',
    'Leukocytes [#/volume] in Blood': 'G/l',
    'Platelets [#/volume] in Blood': 'G/l',
    'Capillary refill': '',
    'Inspired oxygen concentration': '%',
    'Glasgow coma score total': 'Ordinal Score',
    'Urea nitrogen [Mass/volume] in Serum or Plasma': 'mg/dL',
    'Cholesterol [Mass/volume] in Serum or Plasma': 'mg/dL',
    'Hematocrit [Volume Fraction] of Blood': '%',
    'Oxygen measurement, partial pressure, arterial': 'mmHg',
    'Mechanical ventilation response': 'Categorical (0, 1)',
    'Troponin I measurement': 'μg/L',
    'Partial thromboplastin time ratio': 'sec',
    'Creatine kinase [Mass/volume] in Blood': 'IU/L',
    'Creatine kinase.MB [Mass/volume] in Blood': 'ng/mL',
    'MCH - Mean corpuscular haemoglobin': 'pg',
    'MCHC [Mass/volume]': '%',
    'MCV [Entitic volume]': 'fL'
}

all_ts_features=['Heart rate', 'Core body temperature', 'Invasive systolic arterial pressure', 'Invasive diastolic arterial pressure', 'Invasive mean arterial pressure', 'Non-invasive systolic arterial pressure', 'Non-invasive diastolic arterial pressure', 'Non-invasive mean arterial pressure', 'End tidal carbon dioxide concentration', 'Respiratory rate', 'Inspired oxygen concentration', 'Positive end expiratory pressure setting', 'Expiratory tidal volume', 'Tidal volume setting', 'Plateau pressure', 'Hourly urine volume', 'Glasgow Coma Score verbal response subscore', 'Glasgow Coma Score motor response subscore', 'Glasgow Coma Score eye opening subscore', 'Body weight', 'Body height measure', 'Base excess in Arterial blood by calculation', 'Hemoglobin [Mass/volume] in Arterial blood', 'Bicarbonate [Moles/volume] in Arterial blood', 'Lactate [Mass/volume] in Arterial blood', 'Methemoglobin/Hemoglobin.total in Arterial blood', 'pH of Arterial blood', 'Carbon dioxide [Partial pressure] in Arterial blood', 'Oxygen [Partial pressure] in Arterial blood', 'Oxygen saturation in Arterial blood', 'Troponin T.cardiac [Mass/volume] in Serum or Plasma', 'Creatine kinase.MB [Mass/volume] in Serum or Plasma', 'Potassium [Moles/volume] in Blood', 'Sodium [Moles/volume] in Blood', 'Chloride [Moles/volume] in Blood', 'Calcium.ionized [Moles/volume] in Blood', 'Calcium [Moles/volume] in Blood', 'Phosphate [Moles/volume] in Blood', 'Magnesium [Moles/volume] in Blood', 'Urea [Moles/volume] in Venous blood', 'Creatinine [Moles/volume] in Blood', 'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma', 'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma', 'Bilirubin.total [Moles/volume] in Serum or Plasma', 'Bilirubin.direct [Mass/volume] in Serum or Plasma', 'Alkaline phosphatase [Enzymatic activity/volume] in Blood', 'aPTT in Blood by Coagulation assay', 'Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay', 'INR in Blood by Coagulation assay', 'Albumin [Mass/volume] in Serum or Plasma', 'Glucose [Moles/volume] in Serum or Plasma', 'C reactive protein [Mass/volume] in Serum or Plasma', 'Lymphocytes [#/volume] in Blood', 'Neutrophils/100 leukocytes in Blood', 'Band form neutrophils/100 leukocytes in Blood', 'Leukocytes [#/volume] in Blood', 'Platelets [#/volume] in Blood', 'Capillary refill', 'Inspired oxygen concentration', 'Glasgow coma score total', 'Urea nitrogen [Mass/volume] in Serum or Plasma', 'Cholesterol [Mass/volume] in Serum or Plasma', 'Hematocrit [Volume Fraction] of Blood', 'Oxygen measurement, partial pressure, arterial', 'Mechanical ventilation response', 'Troponin I measurement', 'Partial thromboplastin time ratio', 'Creatine kinase [Mass/volume] in Blood', 'Creatine kinase.MB [Mass/volume] in Blood', 'MCH - Mean corpuscular haemoglobin', 'MCHC [Mass/volume]', 'MCV [Entitic volume]']


def make_overall_knowledge(db_knowledge, overview_knowledge):
    overall_instruction = """We want to make overall knowledge of database. So, make best knowledge that fits details between [DB knowledge] and [Overview knowledge]. 
    Follow the format of each knowledge.
    Details :\n"""
    prompt = overall_instruction + static_instruction + target_instruction + timeseries_instruction + mapping_instruction +relation_instruction + "\n[DB knowledge]:\n" + db_knowledge + "\n[Overview knowledge]:\n" + overview_knowledge
    return prompt

def separate_knowledge(knowledge):
    knowledge_set = knowledge.split("###")
    static_info = knowledge_set[1]
    target_info = knowledge_set[2]
    timeseries_info = knowledge_set[3]
    map_info = knowledge_set[4]
    relation_info = knowledge_set[5]
    return static_info, target_info, timeseries_info, map_info, relation_info