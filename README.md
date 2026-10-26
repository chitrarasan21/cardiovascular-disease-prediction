
# Cardiovascular Disease Prediction

A student-style machine-learning project using the cardiovascular disease dataset. It covers data preprocessing, exploratory visualization, feature correlations, comparison of five classifiers, and saving a selected model.

## Project contents

- `outputs/Cardiovascular_Disease_Prediction.ipynb` — notebook walkthrough
- `outputs/Project_Report.md` — written project report and findings
- `outputs/*.png` — plots and model evaluation figures
- `outputs/model_comparison.csv` — accuracy, ROC-AUC, and cross-validation scores
- `outputs/correlation_matrix.csv` — numeric correlation matrix
- `outputs/cardiovascular_model.joblib` — selected fitted SVM pipeline
- `work/cardio_analysis.py` — script to reproduce analysis outputs
- `requirements.txt` — Python dependencies

## Run the analysis

Install Python 3.9 or newer, then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python work/cardio_analysis.py --data "/path/to/cardio_train (1).csv" --out outputs
```

The CSV is not included. Download it separately and pass its local path to the script. The notebook currently uses the original local path in its first data-loading cell; update `DATA_PATH` if your copy is elsewhere.

## Results

With the supplied dataset and the settings in the notebook, SVM achieved 73.55% test accuracy. Random Forest produced the highest ROC-AUC at 0.798. The selected SVM achieved 68% recall on the positive disease class, so the model misses some positive cases. See the report for details and limitations.

This is a coursework demonstration, not a clinical diagnostic tool
