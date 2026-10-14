# HeartWise | Heart-disease ML exploration

HeartWise evolves the original notebook into a small, demonstrable ML product while preserving the original exploratory notebook. It is designed for learning and project demos, not for clinical use.

## What's included

- `ML_01_Heart_Disease.ipynb`: original EDA retained; model section updated to use a leakage-safe scikit-learn pipeline.
- `app.py`: interactive Streamlit application.
- `src/heartwise.py`: reusable validation, preprocessing, and training logic.
- `tests/test_heartwise.py`: core regression tests.
- `requirements.txt`: reproducible core dependencies.
- `.gitignore`: excludes local datasets, virtual environments, and model artifacts.

## Signature feature: What-if lab

A user creates a hypothetical record from the uploaded dataset's feature ranges, receives the selected model's positive-class estimate, then changes a numeric feature and compares the output. The UI explicitly describes this as model sensitivity exploration, not clinical risk assessment.

## Product improvements

1. **Model arena:** compares Logistic Regression, Random Forest, and Gradient Boosting on the same stratified holdout split.
2. **Leakage-safe preprocessing:** imputation, scaling, and one-hot encoding are fitted inside a pipeline using training data only.
3. **Data audit:** reports missing values, duplicates removed, unique values, and a preview.
4. **Permutation importance:** ranks features by their effect on held-out F1, without implying causality.
5. **Useful exports:** download model metrics, cleaned data, and hypothetical inputs.
6. **Graceful validation:** detects missing target columns, unsupported target values, single-class targets, and small datasets.
7. **Responsible framing:** no hardcoded sample predictions and a visible medical disclaimer.

## Run locally

Python 3.10+ is recommended.

```bash
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

# In another terminal, run the core tests
python -m unittest discover -s tests
```

Upload a CSV with the target column named `HeartDisease` and encoded as `0/1` (the app also accepts common yes/no labels). The familiar Heart Failure Prediction dataset usually has columns such as `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, and `HeartDisease`.

The uploaded ZIP did **not** include `heart.csv`, so the app intentionally asks for a dataset instead of showing invented predictions. Do not upload identifiable patient data.

## Architecture

```text
CSV upload
   ↓
Validation + conservative cleaning
   ↓
Stratified train/test split
   ↓
Per-model sklearn Pipeline
   ├── numeric: median imputation → scaling
   ├── categorical: mode imputation → one-hot encoding
   └── classifier
   ↓
Held-out metrics + permutation importance
   ↓
What-if scenario comparison + CSV exports
```

## Demo flow (about 3 minutes)

1. Upload `heart.csv`.
2. Show class balance and dataset audit.
3. Compare F1, recall, precision, and ROC-AUC in **Model arena**.
4. Explain the top permutation-importance features and clarify that importance is not causation.
5. In **What-if lab**, evaluate a baseline hypothetical row, vary a numeric input, and show the changed model estimate.
6. Download the scorecard to demonstrate a practical output.

## Evaluation criteria

- **Data quality:** no crash on missing feature cells; invalid target values produce a clear message.
- **Modeling hygiene:** preprocessing is trained only on the training split.
- **Usefulness:** user can compare models and explore sensitivity without editing Python.
- **Transparency:** held-out metrics and limitations are visible.
- **Reliability:** the app explains missing data/unsupported formats rather than claiming success.

## Limitations and next steps

- The default evaluation is a single split, not nested cross-validation or external validation.
- Predicted probabilities are not calibrated and must not be interpreted as individual medical risk.
- Dataset bias, label quality, subgroup performance, and clinical validity have not been assessed.
- Before any real-world medical use, require expert oversight, appropriate governance, external validation, calibration, subgroup analysis, security review, and regulatory assessment.
