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
    AUGMENTED_DATA_PATH,
    EXCEL_PATH,
    FEATURE_DOMAINS,
    FEATURE_LABELS,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PEARSON_15_FEATURES,
    TABLES_DIR,
    VALUE_DECODERS,
    explain_individual_staff,
    generate_personalized_feedback,
    load_all_project_sheets,
)

st.set_page_config(
    page_title="AKSU Academic Staff Stress Assessment & Generative AI System",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def get_cached_sheets():
    return load_all_project_sheets(EXCEL_PATH)


@st.cache_data
def get_cached_augmented_data():
    if AUGMENTED_DATA_PATH.exists():
        return pd.read_csv(AUGMENTED_DATA_PATH)
    from train_and_build_pipeline import run_complete_pipeline
    run_complete_pipeline()
    return pd.read_csv(AUGMENTED_DATA_PATH)


@st.cache_resource
def get_cached_model_bundle():
    bundle_path = MODELS_DIR / "stress_assessment_bundle.joblib"
    if not bundle_path.exists():
        from train_and_build_pipeline import run_complete_pipeline
        run_complete_pipeline()
    return joblib.load(bundle_path)


def main():
    sheets = get_cached_sheets()
    df3_orig = sheets["sheet3_digitized"]
    df4 = sheets["sheet4_transpose"]
    df5 = sheets["sheet5_attributes"]
    df_aug = get_cached_augmented_data()
    bundle = get_cached_model_bundle()

    best_model_name = bundle.get("best_model_name", "Ridge Logistic Regression")
    selected_model = bundle.get("selected_model")
    trained_models = bundle.get("trained_models", {})
    cv5_results = bundle.get("cv_results_5fold", {})

    # Sidebar
    st.sidebar.markdown("## 🎓 AKSU Stress Assessment System")
    st.sidebar.caption(
        "**Researcher:** Okon, Goodnews Asuquo (`AK22/PHS/CSC/078`)  \n"
        "**Department:** Computer Science  \n"
        "**Institution:** Akwa Ibom State University (AKSU)"
    )
    st.sidebar.divider()

    st.sidebar.markdown("### ⚙️ Model Selection")
    model_choice = st.sidebar.selectbox(
        "Active Classifier:",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name) if best_model_name in trained_models else 0,
        help="Select among the 4 models evaluated via Stratified 5-Fold Cross-Validation.",
    )
    active_pipeline = trained_models.get(model_choice, selected_model)

    st.sidebar.divider()
    st.sidebar.markdown("### 🏆 Optimal Classifier (Ridge)")
    if best_model_name in cv5_results:
        cv_best = cv5_results[best_model_name]
        st.sidebar.success(
            f"**Selected Model:** `{best_model_name}`  \n"
            f"**5-Fold CV Accuracy:** `{cv_best['mean_accuracy']*100:.2f}%`  \n"
            f"**5-Fold CV F1-Score:** `{cv_best['mean_f1']*100:.2f}%`  \n"
            f"**5-Fold CV ROC-AUC:** `{cv_best['mean_roc_auc']*100:.2f}%`  \n"
            f"**OOF Errors:** `{cv_best['fp'] + cv_best['fn']} / 110`"
        )

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 Generative AI Configuration")
    api_key_input = st.sidebar.text_input(
        "Gemini API Key (`GEMINI_API_KEY`)",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Optional: Enter a Google Gemini API key to invoke live gemini-3.8-flash via the google-genai SDK. When blank, the system uses the built-in RAG-Grounded Local Generative Engine.",
    )
    use_live = st.sidebar.checkbox("Enable Live Gemini 3.8 Flash API", value=True)

    # Main Header
    st.title("🎓 Generative AI-Based Personalized Feedback System for Academic Staff Stress-Level Assessment")
    st.markdown(
        "**Case Study:** Academic Staff of Akwa Ibom State University, Ikot Akpaden, Nigeria  \n"
        "*Integrating Pearson Correlation Feature Selection (15 Indicators), Supervised Classifiers "
        "(`Elastic Net`, `Ridge Logistic Regression`, `Shallow Decision Tree`, `Random Forest` under Stratified 5-Fold CV on $N=110$ Augmented Dataset), "
        "and Retrieval-Augmented Generative AI (`Gemini 3.8 Flash`) for Personalized Wellbeing Feedback.*"
    )

    tab1, tab2, tab3 = st.tabs([
        "🩺 1. Stress Assessment & Generative AI Personalized Feedback",
        "📊 2. Dataset Explorer (Original N=74 & Augmented N=110)",
        "⚙️ 3. Pearson Feature Selection & Model Evaluation",
    ])

    # =======================================================================
    # TAB 1: INDIVIDUAL STRESS ASSESSMENT & GENERATIVE AI FEEDBACK
    # =======================================================================
    with tab1:
        st.subheader("Stress Assessment Module & Generative AI Personalized Feedback Engine")
        mode = st.radio(
            "Select Assessment Input Mode:",
            [
                "Select Existing Academic Staff Respondent from Augmented Cohort (`A001` – `A110`)",
                "Complete Interactive 15-Item Pearson Stress Questionnaire (New Assessment)",
            ],
            horizontal=True,
        )

        if mode.startswith("Select Existing"):
            col_sel, col_info = st.columns([1.2, 1.8])
            with col_sel:
                staff_options = [
                    f"{row['staff_code']} — {VALUE_DECODERS['academic_rank'].get(int(row['academic_rank']), 'Rank')} "
                    f"({'HIGH STRESS [1]' if row['stress_class'] == 1 else 'LOW STRESS [0]'}, TOTAL={row['total_score']})"
                    for _, row in df_aug.iterrows()
                ]
                default_idx = 38 if len(staff_options) > 38 else 0  # Default A039
                chosen_label = st.selectbox("Choose Staff Code:", staff_options, index=default_idx)
                chosen_code = chosen_label.split(" — ")[0]
                staff_row = df_aug[df_aug["staff_code"] == chosen_code].iloc[0].copy()

            with col_info:
                st.info(
                    f"**Staff Code:** `{staff_row['staff_code']}` | "
                    f"**Rank:** {VALUE_DECODERS['academic_rank'].get(int(staff_row['academic_rank']), 'Unknown')} | "
                    f"**Qualification:** {VALUE_DECODERS['educational_qualification'].get(int(staff_row['educational_qualification']), 'Unknown')} | "
                    f"**Experience:** {VALUE_DECODERS['teaching_experience'].get(int(staff_row['teaching_experience']), 'Unknown')}  \n"
                    f"**Recorded Composite Score (`TOTAL`):** `{staff_row['total_score']}` | "
                    f"**Ground-Truth Class (`CLASSES`):** `{'1 (High Stress)' if staff_row['stress_class'] == 1 else '0 (Low Stress)'}`"
                )
        else:
            st.markdown("#### Complete the 15-Item Pearson Academic Staff Stress Questionnaire")
            custom_vals = {}
            doms = list(dict.fromkeys([FEATURE_DOMAINS[f] for f in PEARSON_15_FEATURES]))
            for dom in doms:
                with st.expander(f"📋 {dom}", expanded=(dom == "Occupational & Institutional Stressors")):
                    dom_cols = [c for c in PEARSON_15_FEATURES if FEATURE_DOMAINS[c] == dom]
                    grid_cols = st.columns(2)
                    for idx_c, f_key in enumerate(dom_cols):
                        decoder = VALUE_DECODERS.get(f_key, {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"})
                        options_keys = list(decoder.keys())
                        default_idx = min(2, len(options_keys) - 1)
                        with grid_cols[idx_c % 2]:
                            selected_k = st.selectbox(
                                FEATURE_LABELS[f_key],
                                options=options_keys,
                                format_func=lambda k, d=decoder: f"{k} — {d[k]}",
                                index=default_idx,
                                key=f"q_{f_key}",
                            )
                            custom_vals[f_key] = int(selected_k)

            # Fill remaining indicators with median for compatibility
            for c in sheets["sheet3_digitized"].columns:
                if c not in custom_vals and c not in ["staff_code", "total_score", "stress_class"]:
                    custom_vals[c] = int(sheets["sheet3_digitized"][c].median())

            custom_vals["staff_code"] = "NEW_EVAL"
            custom_vals["total_score"] = sum(custom_vals[c] for c in PEARSON_15_FEATURES)
            custom_vals["stress_class"] = 1 if custom_vals["total_score"] >= 28 else 0
            staff_row = pd.Series(custom_vals)

        # Run Assessment via the selected model
        explanation = explain_individual_staff(staff_row, active_pipeline, df_aug, features=PEARSON_15_FEATURES)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Assessed Stress Category",
            explanation["predicted_label"],
            delta="High Priority Coping Intervention" if explanation["predicted_class"] == 1 else "Equilibrium Maintenance",
            delta_color="inverse" if explanation["predicted_class"] == 1 else "normal",
        )
        m2.metric("High-Stress Probability", f"{explanation['probability_high_stress']*100:.1f}%")
        m3.metric("Composite Score (`TOTAL`)", f"{int(staff_row['total_score'])}")
        m4.metric("Active Model", model_choice)

        st.divider()

        # Generative AI Feedback Section (Direct from Classification)
        st.markdown("### 🤖 Generative AI Personalized Feedback System")
        st.caption(
            f"The Generative AI Personalized Feedback Engine directly takes the stress classification result "
            f"(**{explanation['predicted_label']}** via `{model_choice}`) and individual questionnaire responses, "
            f"generating tailored, empathetic, and evidence-grounded recommendations for Akwa Ibom State University."
        )

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
    # TAB 2: DATASET EXPLORER
    # =======================================================================
    with tab2:
        st.subheader("Multi-Sheet Dataset Reconciliation & Augmentation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Cohort (Sheet 3)", len(df3_orig))
        c2.metric("Augmented Cohort (N=110)", len(df_aug))
        c3.metric("Low Stress Staff (Class 0)", f"{(df_aug['stress_class']==0).sum()} (61.8%)")
        c4.metric("High Stress Staff (Class 1)", f"{(df_aug['stress_class']==1).sum()} (38.2%)")

        st.markdown("##### 1. Augmented Academic Staff Dataset ($N=110$)")
        st.dataframe(df_aug, use_container_width=True, height=260)

        col_s4, col_s5 = st.columns([1.2, 1])
        with col_s4:
            st.markdown("##### 2. Sheet 4 (`transpose`): Staff-by-Attribute Matrix")
            st.dataframe(df4, use_container_width=True, height=300)
        with col_s5:
            st.markdown("##### 3. Sheet 5 (`attributes responses`): Cumulative Domain Burden")
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
    # TAB 3: PEARSON FEATURE SELECTION & MODEL EVALUATION
    # =======================================================================
    with tab3:
        st.subheader("Pearson Correlation Feature Selection & Stratified 5-Fold Cross-Validation")
        st.markdown(
            "Features were screened using bivariate Pearson product-moment correlation analysis ($|r| \\ge 0.30, p < 0.01$). "
            "The four specified classifiers (**Elastic Net**, **Ridge Logistic Regression**, **Shallow Decision Tree**, **Random Forest**) "
            "were then trained and cross-validated using **Stratified 5-Fold CV** on the augmented $N=110$ dataset."
        )

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
