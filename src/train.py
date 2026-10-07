"""Train and compare loan-approval classifiers using a local CSV file."""

from pathlib import Path
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Paths are relative to the project root, so this script works in VS Code
# regardless of which folder is selected as the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET = "Loan_Status"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Read the local training CSV and perform basic structural checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put the Kaggle training CSV at data/raw/train.csv, then run again."
        )

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("The CSV has headers but contains no data rows.")
    if TARGET not in data.columns:
        raise ValueError(
            f"Required target column {TARGET!r} was not found. "
            f"Columns in your CSV: {list(data.columns)}"
        )

    # Rows without a known outcome cannot teach a supervised classifier.
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(str).str.strip().str.upper()
    unexpected = sorted(set(data[TARGET].unique()) - {"Y", "N"})
    if unexpected:
        raise ValueError(
            f"Expected Loan_Status values Y and N; found unexpected values: {unexpected}"
        )
    return data


def save_dashboard_summary(data: pd.DataFrame) -> None:
    """Save aggregate charts so the hosted app does not need the raw CSV."""
    summary = {
        "total_rows": int(len(data)),
        "applicant_field_count": int(len([c for c in data.columns if c not in {"Loan_ID", TARGET}])),
        "approval_counts": {str(key): int(value) for key, value in data[TARGET].value_counts().items()},
        "missing_counts": {str(key): int(value) for key, value in data.isna().sum().items() if value > 0},
        "property_area_outcomes": {},
        "numeric_distributions": {},
    }
    if "Property_Area" in data:
        area_table = pd.crosstab(data["Property_Area"], data[TARGET]).reindex(columns=["Y", "N"], fill_value=0)
        area_table = area_table.rename(columns={"Y": "Approved", "N": "Not approved"})
        summary["property_area_outcomes"] = {
            str(area): {str(label): int(count) for label, count in row.items()}
            for area, row in area_table.to_dict(orient="index").items()
        }

    for column in data.select_dtypes(include="number").columns:
        values = data[column].dropna()
        if len(values) > 1:
            bins = pd.cut(values, bins=12).value_counts().sort_index()
            summary["numeric_distributions"][str(column)] = {
                f"{int(interval.left):,}–{int(interval.right):,}": int(count)
                for interval, count in bins.items()
            }

    summary_path = REPORTS_DIR / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build separate missing-value and encoding steps for each column type."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def make_model_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    """Keep preprocessing and the classifier together to prevent data leakage."""
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            ("classifier", classifier),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """Print common binary-classification metrics and return them as a dictionary."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_approved": precision_score(y_test, predictions, pos_label="Y", zero_division=0),
        "recall_approved": recall_score(y_test, predictions, pos_label="Y", zero_division=0),
        "f1_approved": f1_score(y_test, predictions, pos_label="Y", zero_division=0),
        "roc_auc": roc_auc_score((y_test == "Y").astype(int), probabilities),
    }
    print(f"\n--- {name} ---")
    print(pd.Series(metrics).to_string())
    print("\nConfusion matrix (rows=actual N,Y; columns=predicted N,Y):")
    print(confusion_matrix(y_test, predictions, labels=["N", "Y"]))
    print("\nPer-class details:")
    print(classification_report(y_test, predictions, labels=["N", "Y"], zero_division=0))
    return metrics


def save_feature_analysis(model: Pipeline) -> None:
    """Save model-based feature signals for a first-pass interpretation."""
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        analysis_name = "random_forest_importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        analysis_name = "logistic_regression_coefficient"
    else:
        print("This model does not expose a simple feature-importance measure.")
        return

    analysis = pd.DataFrame({"feature": feature_names, "value": values})
    analysis["absolute_value"] = analysis["value"].abs()
    analysis = analysis.sort_values("absolute_value", ascending=False)
    analysis.to_csv(REPORTS_DIR / f"{analysis_name}.csv", index=False)

    top = analysis.head(15).sort_values("value")
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["value"])
    plt.title(f"Top feature signals: {analysis_name.replace('_', ' ')}")
    plt.xlabel("Importance (tree) or signed coefficient (logistic regression)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{analysis_name}.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train loan approval classifiers from your local Kaggle training CSV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the training CSV (default: data/raw/train.csv).",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    # Kaggle downloads may have a longer name, or Windows may leave a second
    # .csv extension (for example train.csv.csv). If the default filename is
    # absent, accept the only train*.csv file in data/raw automatically.
    if args.data == DATA_PATH and not data_path.exists():
        training_files = sorted((PROJECT_ROOT / "data" / "raw").glob("train*.csv"))
        if len(training_files) == 1:
            data_path = training_files[0]
        elif len(training_files) > 1:
            raise ValueError(
                "Found multiple training CSV files in data/raw. "
                "Pass the one you want explicitly with --data."
            )
    data = load_data(data_path)
    save_dashboard_summary(data)
    print(f"Loaded {len(data)} labeled rows and {len(data.columns)} columns from {data_path}")
    print("\nColumn types and missing values:")
    print(pd.DataFrame({"dtype": data.dtypes.astype(str), "missing": data.isna().sum()}))
    print("\nApproval class counts:")
    print(data[TARGET].value_counts())

    # Loan_ID is an identifier, not an applicant characteristic. Excluding it
    # avoids teaching the model arbitrary patterns tied to row identifiers.
    drop_columns = [TARGET]
    if "Loan_ID" in data.columns:
        drop_columns.append("Loan_ID")
    X = data.drop(columns=drop_columns)
    y = data[TARGET]

    # Keep a final holdout set untouched while comparing and tuning models.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": make_model_pipeline(
            X_train, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "random_forest": make_model_pipeline(
            X_train,
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))

    # Tune Random Forest settings using cross-validation on training data only.
    # GridSearchCV tries each listed combination and selects the best macro F1.
    # Macro F1 averages both classes equally and works with string labels.
    forest_pipeline = candidates["random_forest"]
    parameter_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 3, 5],
    }
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print("\nBest Random Forest settings:")
    print(search.best_params_)
    print(f"Best cross-validation macro F1: {search.best_score_:.3f}")
    results.append(evaluate_model("tuned_random_forest", search.best_estimator_, X_test, y_test))

    results_frame = pd.DataFrame(results).sort_values("f1_approved", ascending=False)
    results_frame.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (sorted by approval F1):")
    print(results_frame.to_string(index=False))

    best_model = search.best_estimator_
    joblib.dump(best_model, MODELS_DIR / "loan_approval_model.joblib")
    save_feature_analysis(best_model)
    print(f"\nSaved tuned model to {MODELS_DIR / 'loan_approval_model.joblib'}")
    print(f"Saved reports and feature analysis to {REPORTS_DIR}")
    print("\nInterpretation reminder: feature signals show association in this dataset, not causation or fairness.")


if __name__ == "__main__":
    main()
