"""Multi-page Streamlit dashboard for the educational loan project."""

from pathlib import Path
import re
import json

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "reports"
SUMMARY_PATH = REPORTS_PATH / "dataset_summary.json"

st.set_page_config(
    page_title="Loan_Scope| Decision Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --grass: #2f855a; --navy: #17324d; --panel: #ffffff; --muted: #607286; }
      .stApp { background: #f4f7fa; }
      [data-testid="stSidebar"] { background: #edf2f6; border-right: 1px solid #d8e1e9; }
      [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
      .brand { display:flex; align-items:center; gap:.75rem; margin:.4rem .2rem 1.4rem; }
      .brand-mark { display:grid; place-items:center; width:2.7rem; height:2.7rem; border-radius:.8rem; background:var(--grass); color:#ffffff; font-size:1.35rem; font-weight:900; }
      .brand-title { color:#17324d; font-size:1.2rem; font-weight:800; line-height:1.1; }
      .brand-subtitle { color:#64788c; font-size:.65rem; letter-spacing:.16em; margin-top:.25rem; }
      .eyebrow { color:var(--grass); font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
      .hero-title { font-size:2.35rem; line-height:1.1; font-weight:800; color:#17324d; margin:.25rem 0 .55rem; }
      .hero-copy { color:#52677c; font-size:1rem; max-width:54rem; }
      .academic-note { background:#f0f7f2; border:1px solid #d5e6da; border-left:3px solid var(--grass); border-radius:.6rem; padding:.8rem .9rem; color:#40584b; font-size:.82rem; }
      div[data-testid="stMetric"] { background:#ffffff; border:1px solid #d8e1e9; padding:1rem; border-radius:.8rem; box-shadow:0 2px 8px rgba(23,50,77,.04); }
      div[data-testid="stMetricLabel"] { color:#607286; }
      div[data-testid="stMetricValue"] { color:#17324d; }
      div[data-testid="stForm"] { border:1px solid #d8e1e9; border-radius:.9rem; background:#ffffff; padding:1.2rem; box-shadow:0 2px 10px rgba(23,50,77,.04); }
      .stButton > button[kind="primaryFormSubmit"], .stFormSubmitButton > button { background:#2f855a; color:#ffffff; border:0; font-weight:800; }
      .stButton > button[kind="primaryFormSubmit"]:hover, .stFormSubmitButton > button:hover { background:#246b48; color:#ffffff; }
      .sidebar-footer { border:1px solid #d8e1e9; background:#ffffff; border-radius:.75rem; padding:.85rem; margin-top:2rem; color:#607286; font-size:.76rem; }
      .sidebar-footer strong { color:#17324d; display:block; margin-bottom:.2rem; }
      h1,h2,h3 { color:#17324d !important; }
      p, label, [data-testid="stCaptionContainer"] { color:#52677c; }
      hr { border-color:#d8e1e9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    """Cache the local Kaggle training CSV while the app is open."""
    return pd.read_csv(path)


@st.cache_data
def load_dataset_summary(path: str) -> dict:
    """Read aggregate dataset charts for hosting without the raw applicant CSV."""
    with open(path, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


@st.cache_data
def load_report(path: str) -> pd.DataFrame:
    """Load a project report CSV if it has already been generated."""
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    """Load the trained preprocessing-and-classification pipeline once."""
    return joblib.load(path)


def section_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div>'
        f'<div class="hero-copy">{description}</div>',
        unsafe_allow_html=True,
    )
    st.write("")


def friendly_feature_name(raw_name: str) -> str:
    """Turn scikit-learn's encoded feature names into readable chart labels."""
    name = raw_name.replace("categorical__", "").replace("numeric__", "")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.replace("_", " ").strip().title()


st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark">L</div><div>'
    '<div class="brand-title">Loan_Scope</div><div class="brand-subtitle">DECISION INTELLIGENCE</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

pages = [
    "▦  Overview",
    "⌁  Application analysis",
    "▤  Model performance",
    "✳  Influencing factors",
    "◷  Dataset insights",
    "ⓘ  About",
]
selected_page = st.sidebar.radio("Menu", pages, label_visibility="collapsed")
page = selected_page.split("  ", 1)[1]
st.sidebar.markdown(
    '<div class="sidebar-footer"><strong>● &nbsp; Academic project</strong>'
    'Historical outcomes are not lending decisions.</div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists() and not SUMMARY_PATH.exists():
    st.error("Training data summary not found. Run `python src/train.py` from the project folder first.")
    st.stop()

data = None
summary = None
if DATA_PATH.exists():
    try:
        data = load_dataset(str(DATA_PATH))
    except Exception as error:
        st.error(f"Could not read the training CSV: {error}")
        st.stop()
if SUMMARY_PATH.exists():
    try:
        summary = load_dataset_summary(str(SUMMARY_PATH))
    except Exception as error:
        st.error(f"Could not read the aggregate dataset summary: {error}")
        st.stop()

if data is not None and "Loan_Status" not in data.columns:
    st.error("The training CSV must include a Loan_Status column.")
    st.stop()

if data is not None:
    total_applications = len(data)
    approved_count = int((data["Loan_Status"] == "Y").sum())
    rejected_count = int((data["Loan_Status"] == "N").sum())
else:
    total_applications = int(summary.get("total_rows", 0))
    approval_counts = summary.get("approval_counts", {})
    approved_count = int(approval_counts.get("Y", 0))
    rejected_count = int(approval_counts.get("N", 0))
approval_rate = approved_count / total_applications if total_applications else 0

if page == "Overview":
    section_heading(
        "Portfolio snapshot",
        "Loan approval overview",
        "A clear snapshot of the historical Kaggle dataset and this educational model project.",
    )
    st.write("")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Total applications", f"{total_applications:,}")
    metric_cols[1].metric("Historical approval rate", f"{approval_rate:.1%}")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Historically approved", f"{approved_count:,}")
    metric_cols[1].metric("Historically not approved", f"{rejected_count:,}")

    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("Recorded outcomes")
        outcome_counts = pd.DataFrame(
            {"Applications": [approved_count, rejected_count]},
            index=["Approved", "Not approved"],
        )
        st.bar_chart(outcome_counts, color="#2f855a", horizontal=True)
        st.caption("Counts in the historical training data, not future predictions.")
    with right:
        st.subheader("Outcomes by property area")
        if data is not None and "Property_Area" in data:
            area_counts = pd.crosstab(data["Property_Area"], data["Loan_Status"])
            area_counts = area_counts.reindex(columns=["Y", "N"], fill_value=0)
            area_counts = area_counts.rename(columns={"Y": "Approved", "N": "Not approved"})
            st.bar_chart(area_counts, color=["#2f855a", "#9aabc0"], stack=True)
        elif summary and summary.get("property_area_outcomes"):
            area_counts = pd.DataFrame.from_dict(summary["property_area_outcomes"], orient="index")
            st.bar_chart(area_counts, color=["#2f855a", "#9aabc0"], stack=True)
        st.caption("A descriptive comparison; differences do not show cause or fairness.")

    st.markdown(
        '<div class="academic-note">This dashboard summarizes a small historical dataset. '
        'It is designed for learning and demonstration, not real credit decisions.</div>',
        unsafe_allow_html=True,
    )

elif page == "Application analysis":
    section_heading(
        "Interactive demo",
        "Explore an example application",
        "Enter sample values to see how the trained model classifies one application.",
    )
    if not MODEL_PATH.exists():
        st.warning("The trained model is missing. Run `python src/train.py` from the project folder first.")
        st.stop()
    model = load_model(str(MODEL_PATH))
    st.markdown(
        '<div class="academic-note">Use fictional/example details only. This score is not a lending '
        'decision and the project does not save form submissions.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with st.form("application_form"):
        st.subheader("Applicant profile")
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        with right:
            applicant_income = st.number_input("Applicant income (dataset units)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant income (dataset units)", min_value=0.0, value=0.0, step=500.0)
            loan_amount = st.number_input("Loan amount (dataset units)", min_value=1.0, value=120.0, step=10.0)
            loan_term = st.number_input("Loan term (months)", min_value=1.0, value=360.0, step=12.0)
            credit_history = st.selectbox(
                "Credit history meets guidelines?", [1.0, 0.0],
                format_func=lambda value: "Yes (1)" if value == 1.0 else "No (0)",
            )
        submitted = st.form_submit_button("Estimate model outcome", type="primary", width="stretch")

    if submitted:
        applicant = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }])
        prediction = model.predict(applicant)[0]
        probability_by_label = dict(zip(model.classes_, model.predict_proba(applicant)[0]))
        approval_score = float(probability_by_label.get("Y", 0.0))
        st.divider()
        st.subheader("Model estimate")
        result_cols = st.columns([1, 1])
        with result_cols[0]:
            if prediction == "Y":
                st.success("Model classification: APPROVED")
            else:
                st.warning("Model classification: NOT APPROVED")
        with result_cols[1]:
            st.metric("Estimated approval score", f"{approval_score:.1%}")
        st.progress(approval_score)
        score_chart = pd.DataFrame(
            {"Model score": [approval_score, float(probability_by_label.get("N", 0.0))]},
            index=["Approved", "Not approved"],
        )
        st.bar_chart(score_chart, color="#2f855a", horizontal=True)
        st.caption("The score reflects this model's output. It is not a guarantee, explanation, or actual lender decision.")

elif page == "Model performance":
    section_heading(
        "Evaluation",
        "Model performance",
        "Compare the models on the held-out portion of the labeled dataset.",
    )
    comparison_path = REPORTS_PATH / "model_comparison.csv"
    if not comparison_path.exists():
        st.warning("No model comparison report yet. Run `python src/train.py` first.")
        st.stop()
    comparison = load_report(str(comparison_path))
    display_names = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "tuned_random_forest": "Tuned random forest",
    }
    comparison["Model"] = comparison["model"].map(display_names).fillna(comparison["model"])
    st.subheader("Score comparison")
    chart_metrics = comparison.set_index("Model")[
        ["accuracy", "f1_approved", "roc_auc"]
    ].rename(columns={"accuracy": "Accuracy", "f1_approved": "F1 for approved", "roc_auc": "ROC-AUC"})
    st.bar_chart(chart_metrics, color=["#2f855a", "#426b8a", "#95a94b"], stack=False)
    st.caption("Scores range from 0 to 1. Higher is better for these summary metrics; each metric measures a different thing.")
    table = comparison[["Model", "accuracy", "precision_approved", "recall_approved", "f1_approved", "roc_auc"]].rename(
        columns={
            "accuracy": "Accuracy",
            "precision_approved": "Precision (approved)",
            "recall_approved": "Recall (approved)",
            "f1_approved": "F1 (approved)",
            "roc_auc": "ROC-AUC",
        }
    )
    st.dataframe(table.style.format({column: "{:.1%}" for column in table.columns if column != "Model"}), hide_index=True, use_container_width=True)
    st.info("In this run, logistic regression scored highest on the holdout set. The saved app model is the tuned random forest; use this dashboard to compare the results, not to claim a final production model.")

elif page == "Influencing factors":
    section_heading(
        "Model interpretation",
        "Influencing factors",
        "See which encoded inputs the saved random forest used most when making its splits.",
    )
    importance_path = REPORTS_PATH / "random_forest_importance.csv"
    if not importance_path.exists():
        st.warning("No feature analysis report yet. Run `python src/train.py` first.")
        st.stop()
    importance = load_report(str(importance_path))
    top_features = importance.nlargest(12, "value").sort_values("value").copy()
    top_features["Readable feature"] = top_features["feature"].map(friendly_feature_name)
    chart = top_features.set_index("Readable feature")[["value"]].rename(columns={"value": "Relative importance"})
    st.bar_chart(chart, color="#2f855a", horizontal=True)
    st.caption("Higher means the forest relied more on that encoded feature for its splits. Importance does not show whether a feature causes an outcome.")
    with st.expander("Why do some categories have names like `Education_Graduate`?"):
        st.write("The preprocessing code uses one-hot encoding: it turns each category into a yes/no indicator so the model can work with it.")

elif page == "Dataset insights":
    section_heading(
        "Data quality",
        "Dataset insights",
        "Explore the size, outcome balance, missing values, and selected numeric distributions.",
    )
    overview_cols = st.columns(3)
    if data is not None:
        row_count = len(data)
        applicant_field_count = len([column for column in data.columns if column not in ["Loan_ID", "Loan_Status"]])
        missing_counts = data.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        numeric_columns = data.select_dtypes(include="number").columns.tolist()
    else:
        row_count = int(summary.get("total_rows", 0))
        applicant_field_count = int(summary.get("applicant_field_count", 0))
        missing_counts = pd.Series(summary.get("missing_counts", {}), dtype="int64")
        numeric_columns = list(summary.get("numeric_distributions", {}).keys())
    overview_cols[0].metric("Rows", f"{row_count:,}")
    overview_cols[1].metric("Applicant fields", f"{applicant_field_count}")
    overview_cols[2].metric("Columns with missing values", f"{len(missing_counts)}")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values by column")
        missing = missing_counts.sort_values(ascending=False)
        missing.index = [friendly_feature_name(column) for column in missing.index]
        missing = missing.rename("Blank cells").to_frame()
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.bar_chart(missing, color="#6b8ca7", horizontal=True)
    with right:
        st.subheader("Historical approval balance")
        outcomes = pd.Series({"Approved": approved_count, "Not approved": rejected_count}).to_frame("Applications")
        st.bar_chart(outcomes, color="#2f855a", horizontal=True)

    if numeric_columns:
        chosen_column = st.selectbox(
            "View a numeric field",
            numeric_columns,
            index=numeric_columns.index("ApplicantIncome") if "ApplicantIncome" in numeric_columns else 0,
            format_func=friendly_feature_name,
        )
        if data is not None:
            values = data[chosen_column].dropna()
        else:
            values = None
        if values is not None and len(values) > 1:
            histogram = pd.cut(values, bins=12).value_counts().sort_index()
            histogram.index = [f"{int(bucket.left):,}–{int(bucket.right):,}" for bucket in histogram.index]
            st.subheader(f"Distribution of {friendly_feature_name(chosen_column)}")
            st.bar_chart(histogram.rename("Applications"), color="#2f855a")
        elif summary:
            histogram = pd.Series(summary.get("numeric_distributions", {}).get(chosen_column, {}), dtype="int64")
            st.subheader(f"Distribution of {friendly_feature_name(chosen_column)}")
            st.bar_chart(histogram.rename("Applications"), color="#2f855a")

elif page == "About":
    section_heading(
        "Project notes",
        "About this project",
        "A classroom demonstration of a machine-learning workflow for a financial dataset.",
    )
    st.subheader("What this project demonstrates")
    st.markdown(
        """
        - **Data preparation:** missing values are filled and categories are encoded.
        - **Classification:** logistic regression and random forest models predict the historical `Loan_Status` label.
        - **Evaluation:** a holdout set and cross-validation provide performance estimates.
        - **Interpretation:** feature importance provides a first-pass summary of model signals.
        - **Interactive demo:** the application page accepts fictional example values and displays the model output.
        """
    )
    st.subheader("Important limitations")
    st.markdown(
        """
        - Historical labels may reflect past policy choices and bias.
        - This small dataset is not sufficient evidence for real lending use.
        - Metrics can change with a different train/validation split.
        - Feature importance shows association within this model, not cause or fairness.
        - The model output is not a real credit decision, financial advice, or a guarantee.
        """
    )
    st.markdown(
        '<div class="academic-note"><strong>Academic project</strong><br>Historical outcomes are not lending decisions.</div>',
        unsafe_allow_html=True,
    )