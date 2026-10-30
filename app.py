"""
Streamlit Web Application: Generative AI-Based Personalized Feedback System
for Academic Staff Stress-Level Assessment (Akwa Ibom State University)

Author: OKON, GOODNEWS ASUQUO (AK22/PHS/CSC/078)
        Department of Computer Science, Akwa Ibom State University
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from stress_system_core import (
    ALL_23_FEATURE_COLS,
    AUGMENTED_DATA_PATH,
    EXCEL_PATH,
    FEATURE_DICTIONARY,
    FEATURE_DOMAINS,
    FEATURE_LABELS,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PEARSON_15_FEATURES,
    TABLES_DIR,
    VALUE_DECODERS,
    decode_feature_value,
    decode_staff_responses,
    encode_feature_value,
    encode_staff_responses,
    explain_individual_staff,
    generate_personalized_feedback,
    load_all_project_sheets,
)

st.set_page_config(
    page_title="Academic Staff Stress Assessment & Gen-AI Feedback System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle_and_data():
    bundle_path = MODELS_DIR / "stress_assessment_bundle.joblib"
    if not bundle_path.exists():
        st.error(f"Model bundle not found at {bundle_path}. Please run train_and_build_pipeline.py first.")
        st.stop()
    bundle = joblib.load(bundle_path)

    if AUGMENTED_DATA_PATH.exists():
        df_aug = pd.read_csv(AUGMENTED_DATA_PATH)
    else:
        sheets_temp = load_all_project_sheets(EXCEL_PATH)
        df_aug = sheets_temp["sheet3_digitized"]

    sheets = load_all_project_sheets(EXCEL_PATH)
    return bundle, df_aug, sheets


def main():
    bundle, df_aug, sheets = load_bundle_and_data()
    df3_orig = sheets["sheet3_digitized"]
    df4 = sheets["sheet4_transpose"]
    df5 = sheets["sheet5_attributes"]
    df1_raw = sheets.get("sheet1_raw", pd.DataFrame())

    # -----------------------------------------------------------------------
    # SIDEBAR: MODEL & GENERATIVE AI SETTINGS
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("⚙️ Settings")
        st.subheader("Model Selection")
        model_names = list(bundle["trained_models"].keys())
        default_model = bundle.get("best_model_name", "Ridge Logistic Regression")
        model_choice = st.selectbox(
            "Classifier:",
            options=model_names,
            index=model_names.index(default_model) if default_model in model_names else 0,
        )
        active_pipeline = bundle["trained_models"][model_choice]

        st.divider()
        st.subheader("Generative AI Engine")
        api_key_input = st.text_input(
            "Gemini API Key (Optional):",
            type="password",
            placeholder="AIzaSy...",
            help="Enter key for live Gemini 3.8 Flash generation. Otherwise, the deterministic literature RAG generator is used.",
        )
        use_live = st.toggle("Enable Live Gemini API", value=bool(api_key_input or os.environ.get("GEMINI_API_KEY")))

    # -----------------------------------------------------------------------
    # MAIN HEADER
    # -----------------------------------------------------------------------
    st.title("🎓 Academic Staff Stress Assessment & Personalized Feedback System")

    tab1, tab2, tab3 = st.tabs([
        "🩺 Stress Assessment Test Page (New Data)",
        "📊 Dataset Explorer",
        "⚙️ Model Evaluation & Metrics",
    ])

    # =======================================================================
    # TAB 1: NEW DATA STRESS ASSESSMENT TEST PAGE & RESULTS
    # =======================================================================
    with tab1:
        c_ref, c_spacer = st.columns([1.5, 2.5])
        with c_ref:
            test_staff_id = st.text_input("Staff Reference ID / Code:", value="TEST_STAFF_01")

        custom_vals = {}

        # Category 1: Socio-Demographic & Academic Profile
        with st.expander("🎓 Socio-Demographic & Academic Profile", expanded=True):
            dom1_cols = ["academic_rank", "educational_qualification", "teaching_experience", "age", "gender"]
            c1, c2 = st.columns(2)
            for idx_c, f_key in enumerate(dom1_cols):
                f_info = FEATURE_DICTIONARY[f_key]
                with (c1 if idx_c % 2 == 0 else c2):
                    chosen_opt = st.selectbox(
                        f_info["label"],
                        options=f_info["options"],
                        index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                        help=f_info["question_text"],
                        key=f"new_{f_key}",
                    )
                    custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

        # Category 2: Occupational & Institutional Stressors
        with st.expander("📚 Occupational & Institutional Stressors", expanded=True):
            dom2_cols = [
                "teaching_workload", "research_responsibilities",
                "administrative_responsibilities", "exam_supervision_marking",
                "teaching_hours"
            ]
            c3, c4 = st.columns(2)
            for idx_c, f_key in enumerate(dom2_cols):
                f_info = FEATURE_DICTIONARY[f_key]
                with (c3 if idx_c % 2 == 0 else c4):
                    chosen_opt = st.selectbox(
                        f_info["label"],
                        options=f_info["options"],
                        index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                        help=f_info["question_text"],
                        key=f"new_{f_key}",
                    )
                    custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

        # Category 3: Lifestyle, Health & Coping Mechanisms
        with st.expander("🏃 Lifestyle, Health & Coping Mechanisms", expanded=False):
            dom3_cols = [
                "physical_exercise", "physical_exercise_frequency",
                "break_frequency", "health_problems_affect_coping",
                "smoking", "alcohol_consumption", "alcohol_frequency"
            ]
            c5, c6 = st.columns(2)
            for idx_c, f_key in enumerate(dom3_cols):
                f_info = FEATURE_DICTIONARY[f_key]
                with (c5 if idx_c % 2 == 0 else c6):
                    chosen_opt = st.selectbox(
                        f_info["label"],
                        options=f_info["options"],
                        index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                        help=f_info["question_text"],
                        key=f"new_{f_key}",
                    )
                    custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

        # Category 4: Generative AI Adoption & Feedback Preferences
        with st.expander("🤖 Generative AI Adoption & Feedback Preferences", expanded=False):
            dom4_cols = [
                "genai_tool_frequency", "genai_research_teaching_usefulness",
                "genai_admin_use", "genai_system_trust",
                "feedback_preference", "feedback_frequency"
            ]
            c7, c8 = st.columns(2)
            for idx_c, f_key in enumerate(dom4_cols):
                f_info = FEATURE_DICTIONARY[f_key]
                with (c7 if idx_c % 2 == 0 else c8):
                    chosen_opt = st.selectbox(
                        f_info["label"],
                        options=f_info["options"],
                        index=f_info["options"].index(f_info["default"]) if f_info["default"] in f_info["options"] else 0,
                        help=f_info["question_text"],
                        key=f"new_{f_key}",
                    )
                    custom_vals[f_key] = encode_feature_value(f_key, chosen_opt)

        # Build complete test respondent series
        custom_vals["staff_code"] = test_staff_id.strip() or "TEST_STAFF"
        custom_vals["total_score"] = sum(custom_vals[c] for c in ALL_23_FEATURE_COLS)
        custom_vals["stress_class"] = 1 if custom_vals["total_score"] >= 40 else 0
        staff_row = pd.Series(custom_vals)

        # Execute Model Classification
        explanation = explain_individual_staff(staff_row, active_pipeline, df_aug, features=PEARSON_15_FEATURES)

        st.divider()

        # Results Summary Cards
        st.subheader("Assessment Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Assessed Stress Category",
            explanation["predicted_label"],
            delta="High Priority Intervention" if explanation["predicted_class"] == 1 else "Equilibrium Maintenance",
            delta_color="inverse" if explanation["predicted_class"] == 1 else "normal",
        )
        m2.metric("High-Stress Probability", f"{explanation['probability_high_stress']*100:.1f}%")
        m3.metric("Composite Score (`TOTAL`)", f"{int(staff_row['total_score'])}")
        m4.metric("Active Model", model_choice)

        st.divider()

        # Generative AI Feedback Section
        st.subheader("🤖 Generative AI Personalized Feedback Report")
        fb_result = generate_personalized_feedback(
            staff_row=staff_row,
            explanation=explanation,
            selected_model_name=model_choice,
            api_key=api_key_input.strip() or None,
            use_live_gemini=use_live,
        )
        st.info(f"**Generation Engine:** `{fb_result['generation_mode']}`")
        st.markdown(fb_result["feedback_text"])

        with st.expander("📚 View Retrieved RAG Evidence Citations & Contributing Indicators"):
            st.dataframe(pd.DataFrame(fb_result["retrieved_evidence"]), use_container_width=True)

    # =======================================================================
    # TAB 2: DATASET EXPLORER (HISTORICAL DATA)
    # =======================================================================
    with tab2:
        st.subheader("Dataset Reconciliation & Feature Codebook")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Cohort", len(df3_orig))
        c2.metric("Augmented Cohort (N=110)", len(df_aug))
        c3.metric("Low Stress Staff (Class 0)", f"{(df_aug['stress_class']==0).sum()} (61.8%)")
        c4.metric("High Stress Staff (Class 1)", f"{(df_aug['stress_class']==1).sum()} (38.2%)")

        view_choice = st.radio(
            "Select View:",
            [
                "Augmented Dataset (N=110, Used for Model Training & 5-Fold CV)",
                "Raw Questionnaire Responses (Sheet 1 Text Responses, N=75)",
                "Digitized Questionnaire Responses (Sheet 3 Likert Encodings, N=74)",
                "Feature Codebook & Dictionary (All 23 Indicators, Input Types & Encodings)",
            ],
            horizontal=True,
        )

        if view_choice.startswith("Augmented Dataset"):
            st.markdown("##### Augmented Academic Staff Dataset ($N=110$)")
            st.dataframe(df_aug, use_container_width=True, height=280)
        elif view_choice.startswith("Raw Questionnaire"):
            st.markdown("##### Sheet 1: Raw Survey Questionnaire Responses (Text Input Types)")
            st.dataframe(df1_raw, use_container_width=True, height=280)
        elif view_choice.startswith("Digitized Questionnaire"):
            st.markdown("##### Sheet 3: Digitized Survey Responses (0–6 Integer Encodings)")
            st.dataframe(df3_orig, use_container_width=True, height=280)
        else:
            st.markdown("##### Feature Codebook & Dictionary: Input Types & Integer Mappings")
            codebook_rows = []
            for fk, finfo in FEATURE_DICTIONARY.items():
                codebook_rows.append({
                    "Feature Key": fk,
                    "Label": finfo["label"],
                    "Domain": finfo["domain"],
                    "Survey Question Text": finfo["question_text"],
                    "Allowed Text Options (Input Types)": " | ".join(finfo["options"]),
                    "Integer Encodings": ", ".join(f"{k}→{v}" for k, v in finfo["decoder"].items()),
                })
            st.dataframe(pd.DataFrame(codebook_rows), use_container_width=True, height=350)

        col_s4, col_s5 = st.columns([1.2, 1])
        with col_s4:
            st.markdown("##### Sheet 4 (`transpose`): Staff-by-Attribute Matrix")
            st.dataframe(df4, use_container_width=True, height=300)
        with col_s5:
            st.markdown("##### Sheet 5 (`attributes responses`): Cumulative Domain Burden")
            st.dataframe(df5[["feature_label", "domain", "Responses", "mean_score_per_staff"]].sort_values("Responses", ascending=False), use_container_width=True, height=300)

        st.divider()
        st.markdown("##### Exploratory Visualizations")
        if (FIGURES_DIR / "fig1_class_and_score_distribution.png").exists():
            st.image(str(FIGURES_DIR / "fig1_class_and_score_distribution.png"), caption="Figure 1: Original vs. Augmented Class & Score Distributions")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if (FIGURES_DIR / "fig2_sheet5_attribute_responses.png").exists():
                st.image(str(FIGURES_DIR / "fig2_sheet5_attribute_responses.png"), caption="Figure 2: Sheet 5 Cumulative Responses by Domain")
        with f_col2:
            if (FIGURES_DIR / "fig3_sheet4_transpose_heatmap.png").exists():
                st.image(str(FIGURES_DIR / "fig3_sheet4_transpose_heatmap.png"), caption="Figure 3: Respondent Profile Heatmap Ordered by Stress Class")

    # =======================================================================
    # TAB 3: MODEL EVALUATION & METRICS
    # =======================================================================
    with tab3:
        st.subheader("Pearson Feature Selection & Stratified 5-Fold Cross-Validation Metrics")

        comp_path = TABLES_DIR / "model_performance_comparison.csv"
        if comp_path.exists():
            c_df = pd.read_csv(comp_path)
            disp_cols = [c for c in c_df.columns if not c.endswith("_num")]
            st.dataframe(c_df[disp_cols], use_container_width=True)

        if (FIGURES_DIR / "fig5_pearson_feature_selection.png").exists():
            st.image(str(FIGURES_DIR / "fig5_pearson_feature_selection.png"), caption="Figure 5: Pearson Correlation Feature Selection Ranking (|r| >= 0.30, p < 0.01)")

        if (FIGURES_DIR / "fig6_model_comparison_metrics.png").exists():
            st.image(str(FIGURES_DIR / "fig6_model_comparison_metrics.png"), caption="Figure 6: Stratified 5-Fold Cross-Validation Performance Comparison")

        cm_c, roc_c = st.columns(2)
        with cm_c:
            if (FIGURES_DIR / "fig7_confusion_matrices.png").exists():
                st.image(str(FIGURES_DIR / "fig7_confusion_matrices.png"), caption="Figure 7: Out-of-Fold Confusion Matrices Across the 4 Classifiers")
        with roc_c:
            if (FIGURES_DIR / "fig8_roc_curves.png").exists():
                st.image(str(FIGURES_DIR / "fig8_roc_curves.png"), caption="Figure 8: Receiver Operating Characteristic (ROC) Curves")

        f_col3, f_col4 = st.columns(2)
        with f_col3:
            if (FIGURES_DIR / "fig9_feature_importance.png").exists():
                st.image(str(FIGURES_DIR / "fig9_feature_importance.png"), caption="Figure 9: Model Feature Coefficients & Weights (Ridge Logistic Regression)")
        with f_col4:
            if (FIGURES_DIR / "fig4_correlation_matrix.png").exists():
                st.image(str(FIGURES_DIR / "fig4_correlation_matrix.png"), caption="Figure 4: Pearson Correlation Matrix across the 15 High-Contribution Indicators")


if __name__ == "__main__":
    main()
