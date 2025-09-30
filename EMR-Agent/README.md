# EMR-Agent 

An AI agent system for patient cohort generation and clinical feature mapping in EMR databases.

## 🏗️ Project Structure

```
EMR-Agent/
├── models/                    # AI model implementations
│   ├── EMR_AGENT.py          # Main EMR agent
│   ├── react.py              # ReAct-style agent
│   └── ehr_sql_pluq_style.py # EHR SQL style agent
├── scripts/                  # Experiment execution scripts
│   ├── run_*.sh             # Experiment execution scripts
│   └── eval_*.sh            # Evaluation scripts
├── test_data/               # Test data
├── cohort_evaluation.py     # Cohort evaluation module
└── featureMap_evaluation.py # Feature mapping evaluation module
```

## 🚀 Key Features

### 1. CFSA (Cohort and Feature Selection Agent)  
- **Purpose**: Extract patient cohorts and feature from medical data
- **Input**: Cohort selection criteria, requested features
- **Output**: Generated cohort with feature

### 2. CMA (Code Mapping Agent)
- **Purpose**: Mapping and selection of clinical features
- **Input**: vital-sign or laboratory features
- **Output**: Feature mapping results

## ⚙️ Environment Setup

### Required Environment Variables

```bash
# Results file storage path (optional, default: ./results)
export RESULTS_BASE_PATH="/path/to/your/results"

# API key (if needed)
export API_KEY="your-api-key-here"
```

## 🗄️ Database Environment Setup

### Supported EMR Datasets

EMR-Agent uses three publicly available EMR datasets for evaluation and operation:

| Dataset | Version | Description |
|---------|---------|-------------|
| **MIMIC-III** | v1.4 | Medical Information Mart for Intensive Care III |
| **eICU** | v2.0 | eICU Collaborative Research Database |
| **SICdb** | v1.0.8 | Salzburg Intensive Care database |

### Database Setup Instructions

The datasets are configured using official open-source scripts to ensure consistent data processing and loading into PostgreSQL environments while preserving the original schema:

#### 1. MIMIC-III Setup
```bash
# Clone the official MIMIC-III build scripts
git clone https://github.com/MIT-LCP/mimic-code.git
cd mimic-code/mimic-iii/buildmimic/postgres

# Follow the official setup instructions to load MIMIC-III data into PostgreSQL
# Official repository: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres
```

#### 2. eICU Setup
```bash
# Clone the official eICU build scripts
git clone https://github.com/MIT-LCP/eicu-code.git
cd eicu-code/build-db/postgres

# Follow the official setup instructions to load eICU data into PostgreSQL
# Official repository: https://github.com/MIT-LCP/eicu-code/tree/main/build-db/postgres
```

#### 3. SICdb Setup
```bash
# For SICdb, manually convert the provided CSV files into PostgreSQL
# Download SICdb v1.0.8 CSV files from the official source
# Create PostgreSQL database and import CSV files using standard PostgreSQL import tools
```

The resulting database environments are used to generate evaluation sets and support EMR-AGENT's operations with consistent data processing across all supported datasets.

## 🎯 Quick Start

### 1. Basic Experiment Execution

```bash
# CMA experiment
./scripts/run_cma.sh

# CFSA experiment
./scripts/run_cfsa.sh
```

### 2. Results Evaluation

```bash
# CMA results evaluation
./scripts/eval_cma.sh

# CFSA results evaluation
./scripts/eval_cfsa.sh
```

### 3. Baseline Comparison

```bash
# Baseline experiment execution
./scripts/run_cma_baselines.sh
./scripts/run_cfsa_baselines.sh

# Baseline results evaluation
./scripts/eval_cma_baselines.sh
./scripts/eval_cfsa_baselines.sh
```

## 🔧 Key Parameters

### Common Parameters
```bash
--target-database          # Target database (eicu|mimic3|sicdb)
--database-knowledge       # external knowledge type (db_and_manual_and_prior|db_only)
--llm-model               # Language model to use (claude-3-5-sonnet-20240620|claude-3-5-haiku-latest)
--trial-num               # Number of experiment repetitions (integer)
--agent-part              # Agent part (cohort|feature_mapping)
--api-key                 # API key (if required)
--max-retries             # Maximum number of retries (default: 3)
```

## 🧩 Ablation Study Modules
EMR-Agent supports the following ablation modules:
1. (`--schema-guideline`): Schema Guideline Module in both "CFSA" and "CMA" 
2. (`--sql-observation`): SQL Observation Module in "CFSA" and Candidates Matching in "CMA"
3. (`--sql-feedback`): SQL Feedback Module in "CFSA" 
