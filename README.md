# AI-Based Loan Approval System

This is a beginner-friendly Python project that learns to classify loan applications using the Kaggle Loan Prediction training CSV. It reads a CSV already on your computer; it does not download a dataset. The default file location is `data/raw/train.csv`. You can also pass the location of your own copy with `--data`.

## 1. Project layout

```text
loan prediction problem/
├── data/raw/                 # Put your training CSV here (not committed to Git)
├── models/                   # Saved trained model
├── reports/                  # Metrics, feature analysis table, and chart
├── src/train.py              # Data preparation, training, evaluation, tuning
├── .gitignore                # Prevents local data/model outputs from Git commits
└── requirements.txt          # Python packages this project needs
```

## 2. Dataset required

Use the Kaggle training CSV, which has a `Loan_Status` column containing `Y` (approved) and `N` (not approved). The Kaggle `test.csv` usually has no `Loan_Status`, so it cannot be used to train or score a supervised classifier. This first version creates its own validation holdout from the labeled training data.

To use the default path, place your training CSV in `data/raw/`. The script uses `train.csv` by default; if that exact filename is absent and there is only one `train*.csv` file in that folder (including a file named `train.csv.csv`), it uses that file. Alternatively, supply a path explicitly with `--data`. The code does not edit your input file.

## 3. Set up in VS Code

1. Open this project folder in VS Code.
2. Install a current Python 3 version if you do not already have one. In VS Code, install/select the Python extension and choose the Python interpreter you will use for this project.
3. Open **Terminal → New Terminal**. Make sure the terminal is at the project root (the folder containing `requirements.txt`).
4. Create an isolated environment:

   ```powershell
   py -m venv .venv
   ```

   An environment keeps this project's Python packages separate from other projects.

5. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use VS Code's interpreter picker to select `.venv`, or run the script with `.venv\Scripts\python.exe` explicitly.

6. Install the listed libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

## 4. Run the project

With your CSV at `data/raw/train.csv`:

```powershell
python src/train.py
```

If the CSV is somewhere else, pass its path. For example:

```powershell
python src/train.py --data "C:\Users\YourName\Downloads\loan-train.csv"
```

The program prints data types, missing-value counts, approval class counts, model metrics, confusion matrices, and the best tuning parameters. It writes its outputs to `reports/` and the selected model to `models/loan_approval_model.joblib`.

## 5. What the code does, in order

### Imports and paths

An **import** makes code from a Python library available. `pandas` reads and works with table-shaped data. Scikit-learn provides preprocessing, classification models, data splitting, and evaluation. `Path` represents file paths safely across operating systems. The constants near the top of `src/train.py` keep the input, output, target column, and random seed in one place.

### `load_data`

This function checks that the CSV exists, reads it with `pandas.read_csv`, checks for the required `Loan_Status` target, and removes rows with no known outcome. The **target** is the answer the model is learning to predict. It also validates that the target contains only `Y` and `N`.

### `make_preprocessor`

The function separates numeric and categorical columns based on their actual values in the CSV. A **numeric feature** is a quantity such as applicant income. A **categorical feature** is a label such as property area or employment status.

- A **missing value** is a blank cell. The median imputer fills a numeric blank with that column's middle value; the most-frequent imputer fills a category blank with its most common value.
- **Scaling** puts numeric values on comparable ranges. It particularly helps logistic regression learn reliably when income and credit history have very different numerical scales.
- **One-hot encoding** turns category labels into binary columns that algorithms can use. For example, a property-area column becomes indicators for its possible categories.
- `handle_unknown="ignore"` lets prediction handle a category not seen during training without crashing.

These steps are in a scikit-learn `Pipeline` and `ColumnTransformer`. A pipeline is a sequence of transformations followed by a model. A column transformer applies different sequences to different column groups. Fitting the full pipeline only on training folds prevents **data leakage**: accidental use of validation information during training.

### Split into training and holdout data

`train_test_split` reserves 20% of labeled rows as a **holdout set**, which is kept aside to estimate performance on unseen examples. The remaining 80% is training data. `stratify=y` preserves approximately the same approved/not-approved proportions in both sets. `random_state=42` makes the random split repeatable.

### Models and evaluation

- **Logistic regression** is a relatively simple classification model. Despite its name, it predicts class probabilities and is often a useful baseline.
- **Random forest** combines many decision trees. Each tree makes rule-based splits; the forest combines their predictions.
- **Accuracy** is the fraction of all predictions that are correct. It can hide poor performance on a smaller class.
- **Precision for approved** asks: of the applications predicted approved, how many were actually approved in the recorded labels?
- **Recall for approved** asks: of the applications actually approved in the recorded labels, how many did the model identify?
- **F1** combines precision and recall as their harmonic mean. Here tuning uses F1 for the approved class, but a real lending system should choose metrics based on explicit business and consumer-protection requirements.
- **ROC-AUC** measures how well scores rank positives above negatives across thresholds; it is not a guarantee of a good operational decision threshold.
- A **confusion matrix** counts correct and incorrect predictions by actual and predicted class. Rows are actual `N`, `Y`; columns are predicted `N`, `Y`.

### Hyperparameter tuning

A **hyperparameter** is a model setting chosen before training, such as the number of trees or the maximum tree depth. `GridSearchCV` tries the listed combinations. **Cross-validation** splits training data into folds, trains on some folds, and scores on the remaining fold repeatedly. `StratifiedKFold` keeps class proportions similar in each fold. The holdout set is not used to choose settings, so it remains a final check.

The current grid is intentionally small enough for a beginner project. Tuning uses **macro F1**, which calculates F1 for each class and averages them equally; this avoids treating the string class label as numeric `1` and gives the less common class equal weight. Larger grids take longer and should be tried only after seeing baseline results.

### Feature signals

The tuned random forest's `feature_importances_` is saved as a CSV and chart. It summarizes how useful a feature was for the forest's splits. These values do not show whether a feature causes approval, and they can be misleading when features are correlated. The script also supports signed logistic-regression coefficients if you choose to inspect that model: positive coefficients push predictions toward `Y`, negative toward `N`, under the fitted encoding. Neither is a fairness test or causal explanation.

## 6. Outputs

- `reports/model_comparison.csv`: holdout metrics for the baseline and tuned models.
- `reports/random_forest_importance.csv`: feature-importance values for the selected forest.
- `reports/random_forest_importance.png`: chart of the strongest feature signals.
- `models/loan_approval_model.joblib`: the fitted preprocessing-plus-model pipeline.

## 7. Improvements to consider after the first run

1. **Check the dataset and outputs together.** Confirm the column types and class counts make sense. A high score on this small dataset is not evidence that the system is ready to make real lending decisions.
2. **Choose metrics with care.** A false approval and a false rejection have different effects. Discuss the costs and requirements before selecting a threshold or optimizing one metric.
3. **Evaluate fairness and privacy.** Some columns may be sensitive or act as proxies for protected traits. Compare errors and calibration across legally and ethically appropriate groups, with expert review and local legal guidance. Do not deploy this educational model for real applicants.
4. **Use a real external test set only if it has labels.** Kaggle's unlabeled `test.csv` is useful for generating a submission, but it cannot provide an accuracy score.
5. **Add better explanations and monitoring.** Feature importance is only a global summary. For a real decision-support tool, add case-level explanations, drift checks, versioning, human review, access controls, and a documented appeal process.

## 8. Run the website

The Streamlit website (`app.py`) has six menu sections: Overview, Application analysis, Model performance, Influencing factors, Dataset insights, and About. It uses a dark navy and mint theme, charts the local training data and saved reports, and provides a form for one example application. It does not save form entries.

1. Install/update project packages: `python -m pip install -r requirements.txt`.
2. Train the model first if `models/loan_approval_model.joblib` is missing: `python src/train.py`.
3. Start the site from the project folder: `python -m streamlit run app.py`.
4. Streamlit prints a local address (usually `http://localhost:8501`) and opens it in your browser. Press `Ctrl+C` in the terminal to stop the local site.

For online hosting, the project needs to be pushed to a GitHub repository and connected to a hosting provider such as Streamlit Community Cloud. The model file must be included in the deployed project, or the app must have a reliable way to train/load it. Do not commit personal applicant data or secrets. The app is an educational demo and must not be used to make real lending decisions.

## 9. Deploy this app on Render

The project includes a `render.yaml` Blueprint for a free Singapore-region web service. Render needs a Git repository as its source. The local raw CSV remains excluded from Git; before deploying, run `python src/train.py` so it creates the saved model, model reports, and aggregate `reports/dataset_summary.json`. The deployed app can then show its dashboard using aggregate data without the row-level CSV.

1. Create a **private** GitHub repository for this project and upload the project files. Include `render.yaml`, `.python-version`, `app.py`, `src/train.py`, `requirements.txt`, `models/loan_approval_model.joblib`, `reports/model_comparison.csv`, `reports/random_forest_importance.csv`, and `reports/dataset_summary.json`. The `.gitignore` is set to omit CSV data files and report images while allowing these deployment artifacts.
2. In Render, connect the GitHub account/repository and create a Blueprint from this repository. Render reads `render.yaml` to install dependencies and start Streamlit on Render's assigned port.
3. When deployment finishes, Render shows the public URL assigned to the service. The free service can spin down after 15 minutes without traffic, so the first visit after a quiet period may take longer.

The Render web service is publicly reachable. Keep the repository private if you do not want the source and saved model to be public. Do not commit applicant data, API keys, or passwords. Render's service setup and Git provider connection are managed in the Render dashboard.

## Important limitation

This is an educational classification exercise built from historical labels. It reproduces patterns in those labels, which may encode past policy or bias. A model score is not a credit decision, a causal explanation, or a recommendation to approve/deny a real person.
