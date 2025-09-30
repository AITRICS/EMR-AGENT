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

### 1. CMA (Code Mapping Agent)
- **Purpose**: Mapping and selection of clinical features
- **Input**: vital-sign or laboratory features
- **Output**: Feature mapping results

## 🗃️ Supported Databases

- **eICU**: eICU Collaborative Research Database
- **MIMIC-III**: Medical Information Mart for Intensive Care III
- **SICdb**: São Paulo ICU Database

## ⚙️ Environment Setup

### Required Environment Variables

```bash
# Results file storage path (optional, default: ./results)
export RESULTS_BASE_PATH="/path/to/your/results"

# API key (if needed)
export API_KEY="your-api-key-here"
```

### Dependencies Installation

```bash
# Install Python packages
pip install -r requirements.txt

# Grant execution permissions to scripts
chmod +x scripts/*.sh
```

## 🎯 Quick Start

### 1. Basic Experiment Execution

```bash
# CMA experiment (cohort generation)
./scripts/run_cma.sh

# CFSA experiment (feature mapping)
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

### 🎯 Schema Guideline Module (`--schema-guideline`)
**Purpose**: Provide database schema guidelines  
**Features**: 
- Detailed schema structure descriptions
- Table relationships and constraint guidance
- Schema rule compliance support for SQL query writing

**Usage**:
```bash
python ./models/EMR_AGENT.py --schema-guideline [other-args]
```

### 👁️ SQL Observation Module (`--sql-observation`)
**Purpose**: Observe and analyze SQL query execution process  
**Features**:
- Monitor generated SQL query execution process
- Query result and performance analysis
- Intermediate step result review and debugging

**Usage**:
```bash
python ./models/EMR_AGENT.py --sql-observation [other-args]
```

### 🔄 SQL Feedback Module (`--sql-feedback`)
**Purpose**: Provide feedback for SQL query improvement  
**Features**:
- Analysis of executed SQL query results
- Automatic correction suggestions when errors occur
- Query optimization and performance improvement feedback

**Usage**:
```bash
python ./models/EMR_AGENT.py --sql-feedback [other-args]
```

### 🔬 Ablation Study Combinations

Various combinations for ablation study:

```bash
# All modules enabled
python ./models/EMR_AGENT.py --schema-guideline --sql-observation --sql-feedback [other-args]

# Only Schema Guideline + SQL Feedback
python ./models/EMR_AGENT.py --schema-guideline --sql-feedback [other-args]

# Only SQL Observation
python ./models/EMR_AGENT.py --sql-observation [other-args]

# All ablation modules disabled (base model)
python ./models/EMR_AGENT.py [other-args]
```

### 📊 Ablation Results Analysis

Metrics measured to analyze the effect of each module:

- **Accuracy Improvement**: Impact of each module on query accuracy
- **Performance Enhancement**: SQL query execution time and efficiency improvement
- **Error Reduction**: Reduced frequency of incorrect query generation
- **User Satisfaction**: Quality improvement of generated results

## 📋 Experimental Scenarios

### Standard Workflow
```bash
# 1. Main experiment execution (all ablation modules enabled)
./scripts/run_cma.sh     # CMA with --schema-guideline --sql-observation --sql-feedback
./scripts/run_cfsa.sh    # CFSA with --schema-guideline --sql-feedback

# 2. Results evaluation  
./scripts/eval_cma.sh
./scripts/eval_cfsa.sh

# 3. Baseline comparison (ReAct, EHR-SQL, etc.)
./scripts/run_cma_baselines.sh    # ReAct, EHR-SQL baselines
./scripts/run_cfsa_baselines.sh   # ReAct baseline
./scripts/eval_cma_baselines.sh
./scripts/eval_cfsa_baselines.sh
```

### Ablation Study Workflow
Individually measure the effect of each ablation module:

```bash
# 1. Full model with all modules enabled
python ./models/EMR_AGENT.py --target-database mimic3 --agent-part cohort \
    --schema-guideline --sql-observation --sql-feedback [other-args]

# 2. Remove Schema Guideline
python ./models/EMR_AGENT.py --target-database mimic3 --agent-part cohort \
    --sql-observation --sql-feedback [other-args]

# 3. Remove SQL Observation
python ./models/EMR_AGENT.py --target-database mimic3 --agent-part cohort \
    --schema-guideline --sql-feedback [other-args]

# 4. Remove SQL Feedback
python ./models/EMR_AGENT.py --target-database mimic3 --agent-part cohort \
    --schema-guideline --sql-observation [other-args]

# 5. Base model (all ablation modules disabled)
python ./models/EMR_AGENT.py --target-database mimic3 --agent-part cohort [other-args]
```
