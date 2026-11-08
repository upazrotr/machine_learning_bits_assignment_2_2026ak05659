"""Evaluate the five notebook-trained Bank Marketing pipelines on an uploaded CSV."""

from pathlib import Path
import csv

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


st.set_page_config(page_title="Bank Marketing | ML Assignment 2", page_icon="🏦", layout="wide")

NUMERIC_FEATURES = [
    "age", "campaign", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome", "pdays",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "SVM": "svm.joblib",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(Path("model") / filename)


st.caption("Name:   Tony Baby \n BITSID: 2026ak05659")
st.title("🏦 Bank Marketing Explorer")
st.write("Can we identify term-deposit subscribers? Compare five classifiers on the same held-out calls.")

with st.sidebar:
    st.header("Choose your experiment")
    selected_model = st.selectbox("Classification model", list(MODEL_FILES), index=4)
    st.caption("SVM was selected using training cross-validation macro F1.")
    st.markdown("**Experiment at a glance**")
    st.write("19 predictors · 5 classifiers · 6 metrics")
    st.write("Target: no = 0, yes = 1")
    st.caption("Source: UCI Bank Marketing. Call duration is excluded because it is known only after the call.")
    st.link_button("Original UCI dataset", "https://archive.ics.uci.edu/dataset/222/bank+marketing")
    st.link_button("View notebook and README", "https://github.com/upazrotr/machine_learning_bits_assignment_2_2026ak05659")

with st.container(border=True):
    st.subheader("Evaluate held-out calls")
    st.info("Download the test CSV below, upload it here, then switch models in the sidebar to compare results.")
    sample_path = Path("test_data.csv")
    if sample_path.exists():
        st.download_button("Download test_data.csv", sample_path.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    else:
        st.warning("test_data.csv is unavailable. Run the notebook to create it.")
    uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"],
                                    help="Include all 19 predictors and y. Comma or semicolon delimiters are accepted.")

if uploaded_file is None:
    st.stop()

try:
    # Read pdays as text so its categories match those learned in the notebook.
    data = pd.read_csv(uploaded_file, sep=None, engine="python",
                       dtype={column: str for column in CATEGORICAL_FEATURES + ["y"]})
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError, csv.Error):
    st.error("Could not read the CSV. Use a UTF-8 CSV with a header and comma or semicolon separators.")
    st.stop()

missing = [column for column in FEATURES + ["y"] if column not in data.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

extra = [column for column in data.columns if column not in FEATURES + ["y"]]
if extra:
    st.error("Unexpected columns: " + ", ".join(extra) + ". Upload exactly the 19 predictors and y; exclude duration and index columns.")
    st.stop()

if data.empty:
    st.error("The CSV contains no data rows. Upload at least one labeled test record.")
    st.stop()

target_text = data["y"].str.strip().str.lower()
y_true = target_text.map({"no": 0, "yes": 1, "0": 0, "1": 1})
if y_true.isna().any():
    st.error("Invalid target y: every row must contain no/yes or 0/1, with no missing targets.")
    st.stop()
y_true = y_true.astype(int)

X = data[FEATURES].copy()
for column in NUMERIC_FEATURES:
    try:
        X[column] = pd.to_numeric(X[column], errors="raise")
    except (ValueError, TypeError):
        st.error(f"Column '{column}' must contain numbers or empty cells for imputation.")
        st.stop()
    if np.isinf(X[column].to_numpy(dtype=float)).any():
        st.error(f"Column '{column}' contains infinity. Use finite numbers or empty cells.")
        st.stop()

try:
    model = load_model(MODEL_FILES[selected_model])
    y_pred = model.predict(X)
    positive_index = list(model.classes_).index(1)
    y_probability = model.predict_proba(X)[:, positive_index]
except (OSError, ImportError, ValueError) as error:
    st.error(f"Could not load or evaluate the saved pipeline: {error}")
    st.info("Run the notebook from the repository root and use the versions in requirements.txt.")
    st.stop()

# Keep both classes in macro metrics even when an upload contains only one class.
auc = roc_auc_score(y_true, y_probability) if y_true.nunique() == 2 else None
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "ROC AUC": auc,
    "Precision (macro)": precision_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
    "Recall (macro)": recall_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
    "F1 (macro)": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Results · {selected_model}")
st.caption(f"{len(data):,} test records · {int(y_true.sum()):,} actual subscribers · default model decision rule")
for column, (label, value) in zip(st.columns(6), metrics.items()):
    column.metric(label, "N/A" if value is None else f"{value:.4f}")
if auc is None:
    st.warning("ROC AUC is undefined because the uploaded target contains only one class. Include both no and yes to calculate it.")
    st.caption("Macro metrics retain both classes; an undefined class score is set to zero. MCC is zero when a class margin is constant.")

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
st.info(f"Subscriber detection: {tp} of {tp + fn} found · {fn} missed · {fp} false positives.")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1],
        display_labels=["no (0)", "yes (1)"], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(selected_model)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Rows: actual outcome. Columns: predicted outcome.")
with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, labels=[0, 1],
        target_names=["no", "yes"], output_dict=True, zero_division=0)
    # Only the comparable per-class and average rows belong in this table.
    report_table = pd.DataFrame({key: report[key] for key in
        ["no", "yes", "macro avg", "weighted avg"]}).T
    st.dataframe(report_table.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
        "f1-score": "{:.4f}", "support": "{:.0f}"}), width="stretch")
    st.caption("The yes row measures subscriber detection. Macro averages give the two classes equal weight.")

with st.expander("Inspect uploaded data"):
    st.dataframe(data.head(10), width="stretch", hide_index=True)

st.caption("Accuracy can hide missed subscribers: compare macro F1, MCC, and the yes-class recall. Results on other uploads may differ from the notebook.")
