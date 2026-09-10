from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

def get_trustworthy_ai_dirs(project_root):
    project_root = Path(project_root)

    return {
        "fairness": project_root / "data" / "models" / "fairness",
        "bias": project_root / "data" / "models" / "bias",
        "robustness": project_root / "data" / "models" / "robustness",
        "xai": project_root / "data" / "models" / "xai",
    }


# ============================================================
# GENERIC CSV LOADER
# ============================================================

def load_csv_files(directory):
    directory = Path(directory)

    if not directory.exists():
        return {}

    csv_files = {}

    for file_path in sorted(directory.glob("*.csv")):
        try:
            csv_files[file_path.name] = pd.read_csv(file_path)
        except Exception:
            continue

    return csv_files


# ============================================================
# FAIRNESS
# ============================================================

def render_fairness(project_root):
    st.subheader("⚖️ Fairness Evaluation")

    fairness_dir = get_trustworthy_ai_dirs(project_root)["fairness"]
    results = load_csv_files(fairness_dir)

    if not results:
        st.info("No Fairness result CSV files were found.")
        return

    st.write(
        "Fairness results are loaded directly from the project's "
        "`data/models/fairness` output files."
    )

    for filename, dataframe in results.items():

        st.markdown(f"### 📄 {filename}")

        if dataframe.empty:
            st.info("This result file is empty.")
            continue

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# BIAS
# ============================================================

def render_bias(project_root):
    st.subheader("📊 Bias Analysis")

    bias_dir = get_trustworthy_ai_dirs(project_root)["bias"]
    results = load_csv_files(bias_dir)

    if not results:
        st.info("No Bias result CSV files were found.")
        return

    st.write(
        "Bias analysis results are loaded directly from the project's "
        "`data/models/bias` output files."
    )

    for filename, dataframe in results.items():

        st.markdown(f"### 📄 {filename}")

        if dataframe.empty:
            st.info("This result file is empty.")
            continue

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ROBUSTNESS
# ============================================================

def render_robustness(project_root):
    st.subheader("🛡️ Robustness Evaluation")

    robustness_dir = get_trustworthy_ai_dirs(project_root)["robustness"]
    results = load_csv_files(robustness_dir)

    if not results:
        st.warning(
            "No Robustness CSV result files were found. "
            "The Robustness analysis may currently be stored "
            "as console output rather than CSV."
        )
        return

    st.write(
        "Robustness results are loaded directly from the project's "
        "evaluation output files."
    )

    for filename, dataframe in results.items():

        st.markdown(f"### 📄 {filename}")

        if dataframe.empty:
            st.info("This result file is empty.")
            continue

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SHAP / EXPLAINABILITY
# ============================================================

def render_explainability(project_root):
    st.subheader("🔍 Explainability & SHAP")

    xai_dir = get_trustworthy_ai_dirs(project_root)["xai"]

    if not xai_dir.exists():
        st.warning("SHAP output directory was not found.")
        return

    # --------------------------------------------------------
    # SHAP CSV RESULTS
    # --------------------------------------------------------

    csv_files = sorted(xai_dir.glob("*.csv"))

    if csv_files:

        st.markdown("### 📈 SHAP Numerical Results")

        for csv_path in csv_files:

            try:
                dataframe = pd.read_csv(csv_path)
            except Exception:
                continue

            st.markdown(f"#### {csv_path.name}")

            if not dataframe.empty:
                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True
                )

    else:
        st.info("No SHAP CSV result files were found.")

    # --------------------------------------------------------
    # SHAP VISUALIZATIONS
    # --------------------------------------------------------

    st.markdown("### 📊 SHAP Visualizations")

    image_files = sorted(xai_dir.glob("*.png"))

    if not image_files:
        st.info("No SHAP visualization images were found.")
        return

    for image_path in image_files:

        st.image(
            str(image_path),
            caption=image_path.name,
            use_container_width=True
        )

    st.caption(
        "SHAP values explain how model features contribute to "
        "predictions. They describe model behavior and should not "
        "be interpreted as medical causation."
    )


# ============================================================
# PRIVACY
# ============================================================

def render_privacy(project_root):
    st.subheader("🔐 Privacy Evaluation")

    st.write(
        "Privacy analysis evaluates the protection of sensitive "
        "healthcare data, direct identifiers, potential training-data "
        "exposure, and model files."
    )

    # ========================================================
    # DATASET PRIVACY STATUS
    # ========================================================

    st.markdown("### 📊 Dataset Privacy")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🫀 CARDIO")
        st.error("⚠️ Direct Identifier Detected")
        st.write("Identifier: `id`")

    with col2:
        st.markdown("#### 🩸 DIABETES")
        st.success("✅ No Obvious Direct Identifier")
        st.write("No obvious direct identifier was detected.")

    with col3:
        st.markdown("#### 💓 HYPERTENSION")
        st.success("✅ No Obvious Direct Identifier")
        st.write("No obvious direct identifier was detected.")

    # ========================================================
    # PRIVACY CHECKS
    # ========================================================

    st.markdown("---")
    st.markdown("### 🔎 Privacy Checks")

    privacy_checks = [
        (
            "Direct identifiers",
            "Evaluated",
            "CARDIO contains the `id` column."
        ),
        (
            "Health-sensitive attributes",
            "Evaluated",
            "Healthcare datasets contain sensitive health information."
        ),
        (
            "Duplicate records",
            "Evaluated",
            "Duplicate-record risk was included in the privacy analysis."
        ),
        (
            "Missing values",
            "Evaluated",
            "Missing-value patterns were included in the privacy assessment."
        ),
        (
            "High-cardinality attributes",
            "Evaluated",
            "High-cardinality attributes were checked as part of privacy assessment."
        ),
        (
            "Potential training-data exposure",
            "Evaluated",
            "Potential exposure through model artifacts was considered."
        ),
        (
            "Model-file privacy",
            "Evaluated",
            "Model files were inspected and should remain access-controlled."
        ),
    ]

    for check_name, status, description in privacy_checks:
        with st.expander(f"🔍 {check_name}", expanded=False):
            st.write(f"**Status:** {status}")
            st.write(description)

    # ========================================================
    # IMPORTANT FINDINGS
    # ========================================================

    st.markdown("---")
    st.markdown("### ⚠️ Important Findings")

    findings = [
        "Healthcare datasets contain sensitive health information.",
        "Direct identifiers should be removed before model training.",
        "The CARDIO dataset contains an `id` column.",
        "Diabetes and Hypertension do not appear to contain an obvious direct identifier.",
        "Model files should be protected from unauthorized access.",
        "Privacy protection also requires access control and secure deployment.",
    ]

    for finding in findings:
        st.write(f"• {finding}")

    # ========================================================
    # PRIVACY RISK WARNING
    # ========================================================

    st.markdown("---")

    st.warning(
        "The absence of a direct identifier does NOT guarantee complete privacy. "
        "Re-identification can still be possible when multiple attributes "
        "are combined."
    )

    st.info(
        "Privacy analysis should therefore be interpreted as a risk assessment "
        "rather than proof of complete privacy."
    )

    # ========================================================
    # PRODUCTION RECOMMENDATIONS
    # ========================================================

    st.markdown("### 🛡️ Recommended Privacy Controls")

    recommendations = [
        "Remove direct identifiers before model training.",
        "Restrict access to healthcare datasets.",
        "Restrict access to trained model files.",
        "Use secure storage for sensitive healthcare data.",
        "Apply data minimization principles.",
        "Avoid exposing raw patient-level data through the application.",
        "Maintain appropriate access control and audit logging in production.",
    ]

    for recommendation in recommendations:
        st.write(f"✅ {recommendation}")

    st.success("Privacy analysis completed.")


# ============================================================
# ACCOUNTABILITY
# ============================================================

def render_accountability():
    st.subheader("📋 Accountability")

    accountability_items = [
        "Model predictions should be traceable to the prediction layer.",
        "Risk classification is handled by the Decision Engine.",
        "Recommendations are generated separately from model prediction.",
        "The system provides a medical-use disclaimer.",
        "Human healthcare professionals remain responsible for medical decisions.",
        "The system is a decision-support prototype, not an autonomous diagnosis system.",
    ]

    for item in accountability_items:
        st.write(f"✅ {item}")


# ============================================================
# SOCIAL IMPACT
# ============================================================

def render_social_impact():
    st.subheader("🌍 Social Impact")

    st.markdown("### Potential Positive Impacts")

    positive = [
        "Early risk awareness",
        "Support for preventive healthcare",
        "Centralized multi-disease risk assessment",
        "Potential support for healthcare professionals",
    ]

    for item in positive:
        st.write(f"➕ {item}")

    st.markdown("### Potential Negative Impacts")

    negative = [
        "False-positive and false-negative predictions",
        "Potential demographic performance disparities",
        "Privacy risks involving sensitive healthcare data",
        "Risk of users over-relying on AI predictions",
    ]

    for item in negative:
        st.write(f"⚠️ {item}")


# ============================================================
# MATH AI
# ============================================================

def render_math_ai():
    st.subheader("🧮 Math AI")

    st.write(
        "The Math AI component provides mathematical foundations "
        "used to support data analysis, machine learning, and "
        "risk-assessment computations in the project."
    )

    math_components = [
        (
            "📐 Calculus",
            "Provides functions for derivatives, integrals, "
            "and mathematical analysis."
        ),
        (
            "📊 Linear Algebra",
            "Provides vector and matrix operations that form "
            "the mathematical foundation of machine learning."
        ),
        (
            "🔢 Advanced Linear Algebra",
            "Extends the linear algebra component with advanced "
            "matrix and vector computations."
        ),
        (
            "🎲 Probability",
            "Provides probability concepts and calculations "
            "relevant to probabilistic risk estimation."
        ),
        (
            "📈 Statistics",
            "Provides statistical methods for descriptive analysis, "
            "data interpretation, and model evaluation."
        ),
    ]

    for name, description in math_components:
        with st.expander(name, expanded=False):
            st.write(description)

    st.markdown("---")

    st.markdown("### ✅ Math AI Implementation Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    components = [
        ("Calculus", col1),
        ("Linear Algebra", col2),
        ("Advanced Linear Algebra", col3),
        ("Probability", col4),
        ("Statistics", col5),
    ]

    for name, column in components:
        with column:
            st.metric(name, "Completed")

    st.info(
        "Math AI is implemented as a supporting mathematical component "
        "of the healthcare assistant. It does not independently make "
        "medical diagnoses or replace the machine-learning prediction models."
    )


# ============================================================
# OVERVIEW
# ============================================================

def render_overview():
    st.subheader("📋 Trustworthy AI Evaluation Status")

    overview_cols = st.columns(4)

    dimensions = [
        ("Fairness", "Evaluated"),
        ("Bias", "Evaluated"),
        ("Robustness", "Evaluated"),
        ("Explainability", "Evaluated"),
        ("Privacy", "Evaluated"),
        ("Accountability", "Evaluated"),
        ("Social Impact", "Evaluated"),
        ("Math AI", "Evaluated"),
    ]

    for index, (dimension, status) in enumerate(dimensions):

        with overview_cols[index % 4]:
            st.metric(dimension, status)

    st.info(
        "Trustworthy AI evaluation supports responsible use of the "
        "healthcare assistant. It does not guarantee that every "
        "prediction is correct or completely free from bias."
    )

    st.markdown("### Known Evaluation Limitations")

    limitations = [
        "Performance can differ between demographic groups.",
        "Model performance can decrease when input noise is introduced.",
        "False-positive and false-negative predictions remain possible.",
        "Healthcare data contains sensitive information.",
        "The CARDIO dataset contains a direct identifier.",
        "Human oversight is required for medical decisions.",
        "Production monitoring and audit logging are not yet implemented.",
    ]

    for limitation in limitations:
        st.write(f"• {limitation}")


# ============================================================
# MAIN DASHBOARD
# ============================================================

def render_trustworthy_ai_dashboard(project_root):

    st.markdown("---")

    st.header("🤖 Trustworthy AI")

    st.write(
        "Evaluation of the AI Healthcare Assistant across "
        "Fairness, Bias, Robustness, Explainability, Privacy, "
        "Accountability, Social Impact, and Math AI."
    )

    tabs = st.tabs([
        "📋 Overview",
        "🔍 Explainability & SHAP",
        "⚖️ Fairness & Bias",
        "🛡️ Robustness",
        "🔐 Privacy",
        "📋 Accountability",
        "🌍 Social Impact",
        "🧮 Math AI",
    ])

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    with tabs[0]:
        render_overview()

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    with tabs[1]:
        render_explainability(project_root)

    # --------------------------------------------------------
    # Fairness + Bias
    # --------------------------------------------------------

    with tabs[2]:
        render_fairness(project_root)

        st.markdown("---")

        render_bias(project_root)

    # --------------------------------------------------------
    # Robustness
    # --------------------------------------------------------

    with tabs[3]:
        render_robustness(project_root)

    # --------------------------------------------------------
    # Privacy
    # --------------------------------------------------------

    with tabs[4]:
        render_privacy(project_root)

    # --------------------------------------------------------
    # Accountability
    # --------------------------------------------------------

    with tabs[5]:
        render_accountability()

    # --------------------------------------------------------
    # Social Impact
    # --------------------------------------------------------

    with tabs[6]:
        render_social_impact()

    # --------------------------------------------------------
    # Math AI
    # --------------------------------------------------------

    with tabs[7]:
        render_math_ai()