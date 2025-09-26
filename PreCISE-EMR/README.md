# EMR Preprocessing Benchmark: *PreCISE-EMR*

This repository contains reproduction code for generating evaluation datasets from PhysioNet databases (MIMIC-III, eICU, SICdb) for EMR-Agent research. Due to PhysioNet's data use agreements, raw data cannot be shared publicly, but this code enables researchers with approved access to reproduce the evaluation results.  


## 📊 Information about supported datasets
|  | [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) | [eICU](https://physionet.org/content/eicu-crd/2.0/) | [SICdb](https://physionet.org/content/sicdb/1.0.8/) |
|---------|---------|---------|---------|
| **Version** | v1.4 | v2.0 | v1.0.8 |
| **Published** | Sept 2016 | Apr 2019 | Sept 2024 |
| **Origin** | USA | USA | Austria |


## ⚠️ Important Notice
**PhysioNet Data Access Requirements**

You cannot run this repository without approved access to PhysioNet datasets. To obtain access, you must first create an account on PhysioNet and complete the required credentialing process, including agreeing to the data use agreement.

**No Raw Data Included**

This repository contains only preprocessing code. No clinical data is provided due to privacy and licensing restrictions.  

**Execution Method**

Unlike some other benchmarks that provide a single `run.sh` script, this project requires running Python scripts directly. Each dataset has its own evaluation script (e.g., `python evaluation_mimic3.py`).  


## 🏗️ Project Structure
```
📦evaluation-benchmark  
 ┣ 📁func/                          # Core processing functions
 ┃ ┣ 📜mimic3.py                    # MIMIC-III specific functions
 ┃ ┣ 📜eicu.py                      # eICU specific functions  
 ┃ └ 📜sicdb.py                     # SICdb specific functions
 ┣ 📁mapping/                       # Mapping Dictionary
 ┃ └ 📜mapping_dictionary.json      # Code mapping dictionary
 ┣ 📜evaluation_mimic3.py           # MIMIC-III evaluation set generator
 ┣ 📜evaluation_eicu.py             # eICU evaluation set generator
 ┣ 📜evaluation_sicdb.py            # SICdb evaluation set generator
 ┣ 📜path.json                      # Configuration file for data paths
 ┣ 📜requirements.txt               # Python packages dependencies
 └ 📜README.md                      # Description
```

## 🚀 Quick Start
### Prerequisites
- Python 3.8+
- Required packages: `pandas`, `numpy`
- Approved access to PhysioNet datasets

### Setup Instructions
1. **Prepare Data Directories**

   Create local directories for each dataset:
   ```bash
   mkdir mimic3 eicu sicdb
   ```

2. **Download PhysioNet Data**

   Place your downloaded files into the corresponding folders:
   - MIMIC-III CSV files (*.csv.gz) → `mimic3/`
   - eICU CSV files (*.csv.gz) → `eicu/`  
   - SICdb CSV files (*.csv.gz) → `sicdb/`

3. **Configure Paths**

   Edit `path.json` with your data directories:
   ```json
   {
       "path_source_mimic3": "/path/to/mimic3",
       "path_source_eicu": "/path/to/eicu", 
       "path_source_sicdb": "/path/to/sicdb",
       "path_evaluation": "/path/to/output"
   }
   ```
   ⚠️ **Important:** If `path.json` is not updated, the scripts will fail with a `FileNotFoundError`.

     The `path_evaluation` directory will serve as the root for storing results. Inside this folder, an `evaluation_set/` directory will be created automatically.

5. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

6. **Generate Evaluation Sets**
   Run the scripts individually:
   ```bash
   python evaluation_mimic3.py
   python evaluation_eicu.py
   python evaluation_sicdb.py
   ```


## 📊 Generated Outputs
Each script generates evaluation datasets inside:
   ```bash
   /path/to/output/evaluation_set/
   ```

### Reliability Evaluation Files
- `reliability_mimic3_cohort[1-3].csv` - MIMIC-III cohort (3 files)
- `reliability_eicu_cohort[1-2].csv` - eICU cohort 1 (2 files)

### Harmonizability Evaluation Files
- `harmonizability_mimic3_criteria[1-7].csv` - MIMIC-III criteria (7 files)
- `harmonizability_eicu_criteria[1-7].csv` - eICU criteria (7 files)
- `harmonizability_sicdb_criteria[1-7].csv` - SICdb criteria (7 files)


## ✅ Notes
Always verify that your local dataset paths are consistent with `path.json`.
