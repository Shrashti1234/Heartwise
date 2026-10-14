"""HeartWise Streamlit user interface."""
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from src.heartwise import TARGET, EXPECTED_COLUMNS, prepare_dataset, train_models, make_patient_row

st.set_page_config(page_title="HeartWise | ML Lab", page_icon="🫀", layout="wide")
st.markdown("""
<style>
:root { color-scheme: light; }
.block-container {max-width: 1240px; padding-top: 1.7rem; padding-bottom: 3rem;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid #dce5e8; border-radius: 18px;
       background: linear-gradient(115deg,#f3faf9 0%,#f8f7ff 100%); margin-bottom: 1.2rem;}
.eyebrow {text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:750;color:#087e80;}
.hero h1 {font-size:2.35rem;letter-spacing:-.04em;margin:.25rem 0;color:#142d3a;}
.hero p {color:#4b626d;margin:0;max-width:780px;}
.disclaimer {padding:.85rem 1rem;border-left:4px solid #d49a35;background:#fff8e8;border-radius:8px;color:#60491e;}
div[data-testid="stMetric"] {border:1px solid #e2e8ec;padding:1rem;border-radius:14px;background:#fff;}
</style>
<div class="hero">
  <div class="eyebrow">Machine learning • Model transparency • Scenario lab</div>
  <h1>HeartWise</h1>
  <p>Explore a heart-disease dataset, compare models fairly, and see how changing inputs changes a model's estimate.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="disclaimer"><b>Important:</b> HeartWise is an educational ML prototype, not a diagnosis, medical device, or substitute for a qualified clinician. Predictions are only as reliable as the uploaded data and have not been clinically validated.</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Start your analysis")
    upload = st.file_uploader("Upload heart.csv", type=["csv"], help="Expected target column: HeartDisease (0/1 or yes/no).")
    st.caption("Your dataset is processed in this app session. Avoid uploading identifiable patient records.")
    st.divider()
    st.caption("Built around the Heart Failure Prediction dataset schema.")

if upload is None:
    st.subheader("Your dataset, your evidence")
    a, b, c = st.columns(3)
    a.metric("Model comparison", "3 pipelines")
    b.metric("Data leakage", "Prevented")
    c.metric("Explainability", "Feature impact")
    st.markdown("### What you can do")
    st.markdown("- **Audit the data:** class balance, missing values, duplicates, and feature distributions.\n- **Compare models:** Logistic Regression, Random Forest, and Gradient Boosting on a held-out split.\n- **Run a what-if:** change patient features and compare the model estimate against a baseline.\n- **Inspect drivers:** permutation importance shows which columns mattered most to held-out performance.")
    st.info("Upload your dataset to unlock the analysis. No sample predictions are shown as if they were real.")
    st.stop()

try:
    raw = pd.read_csv(upload)
    X, y, feature_cols = prepare_dataset(raw)
except Exception as exc:
    st.error(f"Could not prepare this dataset: {exc}")
    st.stop()

if not EXPECTED_COLUMNS.issubset(set(raw.columns)):
    missing = sorted(EXPECTED_COLUMNS - set(raw.columns))
    st.warning("This file differs from the common heart-disease schema. The app will use the available columns. Missing expected columns: " + ", ".join(missing))

with st.spinner("Training and evaluating models on a held-out test set…"):
    try:
        models, metrics, X_test, y_test = train_models(X, y)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.stop()

best_name = metrics.iloc[0]["Model"]
best_model = models[best_name]
tab_overview, tab_models, tab_scenario, tab_data = st.tabs(
    ["Overview", "Model arena", "What-if lab", "Data audit"]
)

with tab_overview:
    st.subheader("Dataset snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Usable records", f"{len(X):,}")
    m2.metric("Features", len(feature_cols))
    m3.metric("Positive class", f"{y.mean():.1%}")
    m4.metric("Best held-out F1", f"{metrics.iloc[0]['F1']:.3f}")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Target distribution")
        dist = y.value_counts().sort_index().rename(index={0: "No disease (0)", 1: "Disease (1)"})
        st.bar_chart(dist)
    with right:
        st.markdown("#### Model scorecard")
        st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
    st.caption("Scores come from one stratified 80/20 split. They are educational estimates, not evidence of clinical performance.")

with tab_models:
    st.subheader("Compare models on the same held-out records")
    st.dataframe(metrics.style.format({c: "{:.3f}" for c in metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
    st.markdown("#### Which features influence predictive performance?")
    with st.spinner("Calculating permutation importance…"):
        try:
            perm = permutation_importance(best_model, X_test, y_test, n_repeats=5,
                                          random_state=42, scoring="f1", n_jobs=-1)
            importance = pd.DataFrame({"Feature": X_test.columns, "F1 drop when shuffled": perm.importances_mean})
            importance = importance.sort_values("F1 drop when shuffled", ascending=False).head(12).set_index("Feature")
            st.bar_chart(importance)
            st.caption("Permutation importance measures the change in held-out F1 after a feature is shuffled. It is not causal evidence.")
        except Exception as exc:
            st.warning(f"Importance could not be calculated for this dataset: {exc}")
    st.download_button("Download model scorecard (CSV)", metrics.to_csv(index=False).encode("utf-8"),
                       file_name="heartwise_model_scorecard.csv", mime="text/csv")

with tab_scenario:
    st.subheader("Signature feature · What-if lab")
    st.write("Create a baseline record, then change one or more inputs to explore how the selected model responds. This is a model sensitivity demo, not a personal medical risk calculator.")
    st.caption(f"Selected automatically by held-out F1: **{best_name}**")
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    defaults = {}
    with st.form("patient_form"):
        cols = st.columns(3)
        for i, col in enumerate(numeric_cols):
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            if series.empty:
                continue
            default = float(series.median())
            min_v, max_v = float(series.min()), float(series.max())
            # Avoid invalid slider ranges on constant or unusual data.
            if min_v == max_v:
                defaults[col] = cols[i % 3].number_input(col, value=default, key=f"base_{col}")
            else:
                defaults[col] = cols[i % 3].slider(col, min_value=min_v, max_value=max_v, value=default, key=f"base_{col}")
        for j, col in enumerate(categorical_cols):
            options = sorted(X[col].dropna().astype(str).unique().tolist())
            if options:
                defaults[col] = cols[j % 3].selectbox(col, options, index=0, key=f"base_{col}")
        submitted = st.form_submit_button("Evaluate baseline", type="primary")
    if submitted or "baseline_values" in st.session_state:
        if submitted:
            st.session_state["baseline_values"] = defaults
        baseline_values = st.session_state["baseline_values"]
        baseline_row = make_patient_row(X, baseline_values)
        baseline_prob = float(best_model.predict_proba(baseline_row)[0, 1])
        st.markdown("#### Baseline estimate")
        st.metric("Model-estimated positive-class probability", f"{baseline_prob:.1%}")
        st.caption("This is a model output for a hypothetical row, not an individual's clinically validated probability.")
        st.markdown("#### Change one input")
        editable = [c for c in numeric_cols if c in baseline_values]
        if editable:
            chosen = st.selectbox("Feature to vary", editable)
            series = pd.to_numeric(X[chosen], errors="coerce").dropna()
            low, high = float(series.min()), float(series.max())
            if low == high:
                st.info(f"{chosen} has only one value in this dataset, so there is no meaningful range to explore.")
            else:
                changed_value = st.slider(f"New {chosen}", low, high, float(baseline_values[chosen]), key="scenario_slider")
                changed_values = dict(baseline_values)
                changed_values[chosen] = changed_value
                changed_row = make_patient_row(X, changed_values)
                changed_prob = float(best_model.predict_proba(changed_row)[0, 1])
                delta = changed_prob - baseline_prob
                x1, x2, x3 = st.columns(3)
                x1.metric("Baseline", f"{baseline_prob:.1%}")
                x2.metric("Scenario", f"{changed_prob:.1%}")
                x3.metric("Difference", f"{delta:+.1%}")
                st.caption("The change reflects this model and dataset only. Correlated features and real-world clinical context are not controlled.")
        st.download_button("Export hypothetical input (CSV)", baseline_row.to_csv(index=False).encode("utf-8"),
                           file_name="heartwise_hypothetical_input.csv", mime="text/csv")

with tab_data:
    st.subheader("Data quality audit")
    d1, d2, d3 = st.columns(3)
    d1.metric("Duplicate rows removed", int(len(raw) - len(X)))
    d2.metric("Missing feature cells", int(X.isna().sum().sum()))
    d3.metric("Columns", X.shape[1])
    audit = pd.DataFrame({
        "Column": X.columns,
        "Type": [str(X[c].dtype) for c in X.columns],
        "Missing": [int(X[c].isna().sum()) for c in X.columns],
        "Unique values": [int(X[c].nunique(dropna=True)) for c in X.columns]
    })
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.markdown("#### Preview")
    st.dataframe(X.head(20).assign(HeartDisease=y.head(20).values), use_container_width=True)
    st.download_button("Download cleaned feature data (CSV)", X.assign(HeartDisease=y).to_csv(index=False).encode("utf-8"),
                       file_name="heartwise_cleaned_data.csv", mime="text/csv")
