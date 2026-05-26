import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            padding: 1.2rem;
            border-radius: 18px;
            background: white;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
            border: 1px solid #eef1f5;
        }
        .success-box {
            padding: 1.3rem;
            border-radius: 18px;
            background: #eafaf1;
            border: 1px solid #c6f6d5;
            color: #14532d;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .danger-box {
            padding: 1.3rem;
            border-radius: 18px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            color: #881337;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .info-box {
            padding: 1rem;
            border-radius: 14px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
MODEL_FILE_NAME = "hr_attrition_model.pkl"


def find_model_file() -> Path | None:
    """Find the model file in the same folder as this app."""
    app_dir = Path(__file__).resolve().parent
    candidate = app_dir / MODEL_FILE_NAME
    if candidate.exists():
        return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_artifact_from_path(path: str):
    """Load the saved pickle/joblib artifact."""
    return joblib.load(path)


def load_uploaded_artifact(uploaded_file):
    """Load model from an uploaded pkl file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        artifact = joblib.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return artifact


def validate_artifact(artifact):
    """Return normalized model metadata from either dict artifact or direct model."""
    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns", [])
        feature_schema = artifact.get("feature_schema", {})
        numeric_features = artifact.get("numeric_features", [])
        categorical_features = artifact.get("categorical_features", [])
        prediction_labels = artifact.get("prediction_labels", {0: "No", 1: "Yes"})
        model_name = artifact.get("model_name", type(model).__name__)
        metrics = artifact.get("best_metrics", {})
    else:
        model = artifact
        feature_columns = []
        feature_schema = {}
        numeric_features = []
        categorical_features = []
        prediction_labels = {0: "No", 1: "Yes"}
        model_name = type(model).__name__
        metrics = {}

    if model is None:
        raise ValueError("The pickle file does not contain a valid model.")

    if not feature_columns:
        raise ValueError(
            "Feature columns were not found in the pickle file. "
            "Please save the model along with feature_columns or update the app manually."
        )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "feature_schema": feature_schema,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "prediction_labels": prediction_labels,
        "model_name": model_name,
        "metrics": metrics,
    }


def get_positive_probability(model, row_df, positive_class=1):
    """Safely get probability of attrition class."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(row_df)[0]

    try:
        classes = list(model.classes_)
    except AttributeError:
        try:
            classes = list(model.named_steps["model"].classes_)
        except Exception:
            classes = list(range(len(probabilities)))

    if positive_class in classes:
        positive_index = classes.index(positive_class)
    elif "Yes" in classes:
        positive_index = classes.index("Yes")
    else:
        positive_index = min(1, len(probabilities) - 1)

    return float(probabilities[positive_index])


def format_label(prediction, prediction_labels):
    """Map prediction output to readable label."""
    try:
        key = int(prediction)
        return prediction_labels.get(key, str(prediction))
    except Exception:
        return prediction_labels.get(prediction, str(prediction))


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed object columns to strings so Streamlit/PyArrow can display them safely."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].apply(
                lambda value: ", ".join(map(str, value))
                if isinstance(value, (list, tuple, set))
                else ("" if pd.isna(value) else str(value))
            )
    return safe_df


def make_input_widget(feature, schema):
    """Create a Streamlit widget based on schema metadata."""
    feature_info = schema.get(feature, {})
    feature_type = feature_info.get("type")

    if feature_type == "categorical":
        options = feature_info.get("options", [])
        default = feature_info.get("default", options[0] if options else "")
        index = options.index(default) if default in options else 0
        return st.selectbox(feature, options=options, index=index)

    min_value = feature_info.get("min", 0)
    max_value = feature_info.get("max", 100)
    default = feature_info.get("default", min_value)

    # Use integer inputs where values are integer-like.
    is_int_like = all(float(v).is_integer() for v in [min_value, max_value, default])
    if is_int_like:
        return st.number_input(
            feature,
            min_value=int(min_value),
            max_value=int(max_value),
            value=int(default),
            step=1,
        )

    return st.number_input(
        feature,
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(default),
        step=0.5,
    )


def predict_dataframe(model, df, prediction_labels):
    """Predict labels and probabilities for a dataframe."""
    predictions = model.predict(df)
    result_df = df.copy()
    result_df["Predicted Attrition"] = [
        format_label(pred, prediction_labels) for pred in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)
        try:
            classes = list(model.classes_)
        except AttributeError:
            try:
                classes = list(model.named_steps["model"].classes_)
            except Exception:
                classes = list(range(probabilities.shape[1]))

        positive_index = classes.index(1) if 1 in classes else min(1, probabilities.shape[1] - 1)
        result_df["Attrition Probability (%)"] = np.round(probabilities[:, positive_index] * 100, 2)

    return result_df


# -----------------------------
# Header
# -----------------------------
st.title("👥 HR Employee Attrition Prediction")
st.caption("Predict whether an employee is likely to leave the company using the trained ML model.")

with st.sidebar:
    st.header("📦 Model Loader")
    st.write("Place `hr_attrition_model.pkl` in the same folder as this app, or upload it below.")

    uploaded_model = st.file_uploader("Upload PKL model file", type=["pkl"])

    st.divider()
    st.markdown("### Run command")
    st.code("streamlit run streamlit_hr_attrition_app.py", language="bash")


# -----------------------------
# Load model
# -----------------------------
try:
    if uploaded_model is not None:
        raw_artifact = load_uploaded_artifact(uploaded_model)
        model_source = "Uploaded PKL file"
    else:
        model_path = find_model_file()
        if model_path is None:
            st.warning(
                "Model file not found. Please keep `hr_attrition_model.pkl` in the same folder "
                "as this Streamlit file, or upload it from the sidebar."
            )
            st.stop()

        raw_artifact = load_artifact_from_path(str(model_path))
        model_source = str(model_path.name)

    meta = validate_artifact(raw_artifact)

except Exception as e:
    st.error("Unable to load the model file.")
    st.write("Error details:")
    st.code(str(e))

    st.info(
        "If you see an error like `MT19937 is not a known BitGenerator module`, "
        "it is usually caused by a NumPy / scikit-learn version mismatch. "
        "Create a fresh environment and reinstall the requirements."
    )

    st.code(
        """
pip install --upgrade pip
pip install streamlit pandas numpy scipy scikit-learn joblib
streamlit run streamlit_hr_attrition_app.py
        """.strip(),
        language="bash",
    )
    st.stop()


model = meta["model"]
feature_columns = meta["feature_columns"]
feature_schema = meta["feature_schema"]
prediction_labels = meta["prediction_labels"]
model_name = meta["model_name"]
metrics = meta["metrics"]


# -----------------------------
# Model Overview
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", model_name)

with col2:
    acc = metrics.get("accuracy")
    st.metric("Accuracy", f"{acc * 100:.2f}%" if acc is not None else "N/A")

with col3:
    roc_auc = metrics.get("roc_auc")
    st.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")

with col4:
    st.metric("Model Source", model_source)


tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "ℹ️ Model Details"])


# -----------------------------
# Single Prediction
# -----------------------------
with tab1:
    st.subheader("Enter Employee Details")

    input_values = {}

    # Group features into manageable sections.
    personal_features = [
        "Age",
        "Gender",
        "MaritalStatus",
        "Education",
        "EducationField",
        "DistanceFromHome",
    ]

    job_features = [
        "Department",
        "JobRole",
        "JobLevel",
        "BusinessTravel",
        "OverTime",
        "MonthlyIncome",
        "DailyRate",
        "HourlyRate",
        "MonthlyRate",
        "PercentSalaryHike",
        "PerformanceRating",
        "StockOptionLevel",
    ]

    satisfaction_features = [
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]

    experience_features = [
        "NumCompaniesWorked",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    sections = [
        ("Personal Information", personal_features),
        ("Job Information", job_features),
        ("Satisfaction Ratings", satisfaction_features),
        ("Experience Details", experience_features),
    ]

    for section_name, section_features in sections:
        with st.expander(section_name, expanded=(section_name == "Personal Information")):
            cols = st.columns(3)
            for i, feature in enumerate(section_features):
                if feature in feature_columns:
                    with cols[i % 3]:
                        input_values[feature] = make_input_widget(feature, feature_schema)

    # Ensure every required feature exists even if not included in sections.
    remaining_features = [f for f in feature_columns if f not in input_values]
    if remaining_features:
        with st.expander("Other Features"):
            cols = st.columns(3)
            for i, feature in enumerate(remaining_features):
                with cols[i % 3]:
                    input_values[feature] = make_input_widget(feature, feature_schema)

    input_df = pd.DataFrame([[input_values[col] for col in feature_columns]], columns=feature_columns)

    st.markdown("### Preview Input")
    st.dataframe(input_df, width="stretch")

    predict_btn = st.button("Predict Attrition", type="primary", width="stretch")

    if predict_btn:
        prediction = model.predict(input_df)[0]
        label = format_label(prediction, prediction_labels)
        probability = get_positive_probability(model, input_df)

        if label.lower() == "yes":
            st.markdown(
                "<div class='danger-box'>⚠️ Prediction: Employee is likely to leave.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='success-box'>✅ Prediction: Employee is not likely to leave.</div>",
                unsafe_allow_html=True,
            )

        if probability is not None:
            st.progress(min(max(probability, 0.0), 1.0))
            st.metric("Attrition Probability", f"{probability * 100:.2f}%")

        st.markdown(
            """
            <div class='info-box'>
            Note: This prediction is based on historical HR data and should be used as a decision-support tool, not as the only decision factor.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Batch Prediction
# -----------------------------
with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    st.write("Your CSV must contain these columns:")
    st.code(", ".join(feature_columns))

    csv_file = st.file_uploader("Upload employee CSV", type=["csv"], key="batch_csv")

    if csv_file is not None:
        try:
            batch_df = pd.read_csv(csv_file)

            missing_cols = [col for col in feature_columns if col not in batch_df.columns]
            if missing_cols:
                st.error("Missing required columns:")
                st.code(", ".join(missing_cols))
            else:
                clean_df = batch_df[feature_columns].copy()
                predictions_df = predict_dataframe(model, clean_df, prediction_labels)

                st.success("Batch prediction completed.")
                st.dataframe(predictions_df, width="stretch")

                csv_output = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Prediction CSV",
                    data=csv_output,
                    file_name="hr_attrition_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as e:
            st.error("Could not process the uploaded CSV.")
            st.code(str(e))


# -----------------------------
# Model Details
# -----------------------------
with tab3:
    st.subheader("Model Details")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Required Features")
        st.write(feature_columns)

    with col_b:
        st.markdown("#### Best Metrics")
        if metrics:
            metrics_df = pd.DataFrame([metrics]).T.reset_index()
            metrics_df.columns = ["Metric", "Value"]
            st.dataframe(make_arrow_safe(metrics_df), width="stretch")
        else:
            st.write("No metrics found in the pickle file.")

    st.markdown("#### Feature Schema")
    if feature_schema:
        schema_df = pd.DataFrame(feature_schema).T.reset_index().rename(columns={"index": "Feature"})
        st.dataframe(make_arrow_safe(schema_df), width="stretch")
    else:
        st.write("No feature schema found.")
