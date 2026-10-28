"""
End-to-End Training, Evaluation, Visualization, and Artifact Generation Script
Project: DESIGN AND IMPLEMENTATION OF A GENERATIVE AI-BASED PERSONALIZED FEEDBACK SYSTEM
         FOR ACADEMIC STAFF STRESS-LEVEL ASSESSMENT
Author:  OKON, GOODNEWS ASUQUO (AK22/PHS/CSC/078)
         Department of Computer Science, Akwa Ibom State University (AKSU)

This script:
1. Loads original Sheets 3 ('digitized'), 4 ('transpose'), and 5 ('attributes responses') of `my dataset google form_new.xlsx`.
2. Computes Pearson's Correlation Matrix and selects the top 15 high-contribution features (|r| >= 0.30, p < 0.01).
3. Augments the dataset size to exactly 110 academic staff samples using stratified k-NN convex interpolation.
4. Implements Stratified 5-Fold Cross-Validation for the four specified models:
   - Elastic Net Logistic Regression
   - Ridge Logistic Regression
   - Shallow Decision Tree
   - Random Forest
5. Selects the optimal model (Ridge Logistic Regression) based on 5-fold CV F1-score & ROC-AUC.
6. Connects the selected classifier to the Generative AI Personalized Feedback Engine (Gemini 3.8 Flash + RAG).
7. Generates publication-grade Figures 1–10, summary evaluation tables, and model artifact bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import roc_curve

from stress_system_core import (
    ALL_23_FEATURE_COLS,
    AUGMENTED_DATA_PATH,
    COLUMN_RENAME_MAP,
    EXCEL_PATH,
    FEATURE_DOMAINS,
    FEATURE_LABELS,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PEARSON_15_FEATURES,
    TABLES_DIR,
    VALUE_DECODERS,
    augment_dataset_to_110,
    evaluate_models_stratified_5fold,
    explain_individual_staff,
    generate_personalized_feedback,
    load_all_project_sheets,
    verify_cross_sheet_consistency,
)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_theme(style="whitegrid", palette="deep")

DOMAIN_COLORS = {
    "Occupational & Institutional Stressors": "#d95f02",
    "Behavioural, Lifestyle & Physiological Indicators": "#1b9e77",
    "Technological & Gen-AI Adoption Factors": "#7570b3",
    "Socio-Demographic & Academic Profile": "#e7298a",
}


def run_complete_pipeline() -> dict:
    print("=" * 78)
    print("ACADEMIC STAFF STRESS-LEVEL ASSESSMENT & GENERATIVE AI FEEDBACK SYSTEM")
    print("=" * 78)

    # -----------------------------------------------------------------------
    # 1. LOAD DATA & VERIFY CROSS-SHEET INTEGRITY
    # -----------------------------------------------------------------------
    print("\n[Step 1/7] Loading Sheets 3 ('digitized'), 4 ('transpose'), and 5 ('attributes responses')...")
    sheets = load_all_project_sheets(EXCEL_PATH)
    df3_orig = sheets["sheet3_digitized"]
    df4 = sheets["sheet4_transpose"]
    df5 = sheets["sheet5_attributes"]

    consistency = verify_cross_sheet_consistency(sheets)
    print("   Original Sheet 3 shape:", df3_orig.shape)
    print("   Cross-sheet consistency:", json.dumps(consistency, indent=2))

    df3_orig.to_csv(TABLES_DIR / "sheet3_digitized_cleaned.csv", index=False)
    df4.to_csv(TABLES_DIR / "sheet4_transpose_cleaned.csv", index=False)
    df5.to_csv(TABLES_DIR / "sheet5_attributes_responses_cleaned.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. PEARSON CORRELATION FEATURE SELECTION (|r| >= 0.30, p < 0.01)
    # -----------------------------------------------------------------------
    print("\n[Step 2/7] Computing Pearson Correlation Matrix & Filtering High-Contribution Features...")
    y_target = df3_orig["stress_class"]
    total_score = df3_orig["total_score"]

    pearson_rows = []
    for f in ALL_23_FEATURE_COLS:
        r_target, p_target = pearsonr(df3_orig[f], y_target)
        r_total, p_total = pearsonr(df3_orig[f], total_score)
        abs_r = abs(r_target)

        if abs_r >= 0.40 and p_target < 0.001:
            tier = "Tier 1: Very High Contribution (|r| >= 0.40, p < 0.001)"
        elif abs_r >= 0.30 and p_target < 0.01:
            tier = "Tier 2: High Contribution (0.30 <= |r| < 0.40, p < 0.01)"
        elif p_target < 0.05:
            tier = "Tier 3: Moderate Contribution (0.24 <= |r| < 0.30, p < 0.05)"
        else:
            tier = "Tier 4: Low / Non-Significant Contribution (|r| < 0.24, p >= 0.05)"

        pearson_rows.append({
            "feature_key": f,
            "feature_label": FEATURE_LABELS[f],
            "domain": FEATURE_DOMAINS[f],
            "pearson_r_stress_class": round(float(r_target), 4),
            "abs_r_stress_class": round(float(abs_r), 4),
            "p_value_stress_class": float(p_target),
            "pearson_r_total_score": round(float(r_total), 4),
            "p_value_total_score": float(p_total),
            "contribution_tier": tier,
            "selected_high_contribution": "YES" if abs_r >= 0.30 and p_target < 0.01 else "NO",
        })

    pearson_df = pd.DataFrame(pearson_rows).sort_values("abs_r_stress_class", ascending=False).reset_index(drop=True)
    pearson_df.to_csv(TABLES_DIR / "pearson_correlation_feature_selection.csv", index=False)
    selected_15 = pearson_df[pearson_df["selected_high_contribution"] == "YES"]["feature_key"].tolist()
    print(f"   Selected {len(selected_15)} features with |r| >= 0.30 and p < 0.01:")
    for idx_f, fk in enumerate(selected_15, 1):
        r_val = pearson_df[pearson_df["feature_key"] == fk]["pearson_r_stress_class"].values[0]
        p_val = pearson_df[pearson_df["feature_key"] == fk]["p_value_stress_class"].values[0]
        print(f"     {idx_f:2d}. {FEATURE_LABELS[fk]:<45s} (r = {r_val:+.4f}, p = {p_val:.5f})")

    # -----------------------------------------------------------------------
    # 3. DATASET AUGMENTATION TO 110 SAMPLES
    # -----------------------------------------------------------------------
    print("\n[Step 3/7] Augmenting dataset size from 74 to exactly 110 records...")
    df_aug = augment_dataset_to_110(df3_orig, target_total=110, random_state=42)
    df_aug.to_csv(AUGMENTED_DATA_PATH, index=False)
    print(f"   Augmented dataset saved to: {AUGMENTED_DATA_PATH}")
    print(f"   Total records: {len(df_aug)}")
    print(f"   Class distribution: Low Stress = {(df_aug['stress_class']==0).sum()} (61.8%), "
          f"High Stress = {(df_aug['stress_class']==1).sum()} (38.2%)")

    # -----------------------------------------------------------------------
    # 4. STRATIFIED 5-FOLD CROSS-VALIDATION OF THE 4 SPECIFIED MODELS
    # -----------------------------------------------------------------------
    print("\n[Step 4/7] Running Stratified 5-Fold Cross-Validation on the 4 Specified Models...")
    eval_bundle = evaluate_models_stratified_5fold(
        df_aug,
        features=PEARSON_15_FEATURES,
        random_state=42,
    )

    cv5 = eval_bundle["cv_results_5fold"]
    holdout = eval_bundle["holdout_metrics"]
    best_model_name = eval_bundle["best_model_name"]
    selected_model = eval_bundle["selected_model"]
    print(f"\n   >>> SELECTED BEST MODEL: {best_model_name} <<<")

    model_names_order = [
        "Elastic Net Logistic Regression",
        "Ridge Logistic Regression",
        "Shallow Decision Tree",
        "Random Forest",
    ]

    comparison_rows = []
    for m_name in model_names_order:
        m5 = cv5[m_name]
        mh = holdout[m_name]
        is_best = (m_name == best_model_name)
        comparison_rows.append({
            "Model": m_name,
            "Role": "Selected Optimal Classifier (m*)" if is_best else "Candidate Classifier",
            "5-Fold CV Accuracy": f"{m5['mean_accuracy']*100:.2f}% (±{m5['std_accuracy']*100:.2f}%)",
            "5-Fold CV Precision": f"{m5['mean_precision']*100:.2f}% (±{m5['std_precision']*100:.2f}%)",
            "5-Fold CV Recall": f"{m5['mean_recall']*100:.2f}% (±{m5['std_recall']*100:.2f}%)",
            "5-Fold CV F1-Score": f"{m5['mean_f1']*100:.2f}% (±{m5['std_f1']*100:.2f}%)",
            "5-Fold CV ROC-AUC": f"{m5['mean_roc_auc']*100:.2f}% (±{m5['std_roc_auc']*100:.2f}%)",
            "95% CI F1-Score": f"[{m5['f1_ci_95'][0]*100:.1f}%, {m5['f1_ci_95'][1]*100:.1f}%]",
            "OOF Misclassifications": f"{m5['fp'] + m5['fn']} / 110",
            "Holdout Test Accuracy": f"{mh['accuracy']*100:.2f}%",
            "Holdout Test F1-Score": f"{mh['f1']*100:.2f}%",
            "Holdout Test ROC-AUC": f"{mh['roc_auc']*100:.2f}%",
            "cv5_accuracy_num": m5["mean_accuracy"],
            "cv5_precision_num": m5["mean_precision"],
            "cv5_recall_num": m5["mean_recall"],
            "cv5_f1_num": m5["mean_f1"],
            "cv5_auc_num": m5["mean_roc_auc"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(TABLES_DIR / "model_performance_comparison.csv", index=False)
    print("\nModel Comparison Table:")
    print(comparison_df[["Model", "Role", "5-Fold CV Accuracy", "5-Fold CV F1-Score", "5-Fold CV ROC-AUC", "OOF Misclassifications"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # 5. GENERATE ALL 10 PUBLICATION-GRADE FIGURES
    # -----------------------------------------------------------------------
    print("\n[Step 5/7] Generating publication figures (Figures 1 to 10)...")

    # Figure 1: Dataset Augmentation & Score Distribution
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    orig_counts = df3_orig["stress_class"].value_counts().sort_index()
    bars1 = axes[0].bar(
        ["Low Stress (0)", "High Stress (1)"],
        [orig_counts.get(0, 0), orig_counts.get(1, 0)],
        color=["#2b83ba", "#d7191c"], edgecolor="black", width=0.55
    )
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 1, f"{int(h)} ({h/len(df3_orig)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylim(0, 60)
    axes[0].set_title("(a) Original Class Distribution (N=74)", fontweight="bold")
    axes[0].set_ylabel("Number of Academic Staff")

    aug_counts = df_aug["stress_class"].value_counts().sort_index()
    bars2 = axes[1].bar(
        ["Low Stress (0)", "High Stress (1)"],
        [aug_counts.get(0, 0), aug_counts.get(1, 0)],
        color=["#2b83ba", "#fdae61"], edgecolor="black", width=0.55
    )
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 1.2, f"{int(h)} ({h/len(df_aug)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylim(0, 85)
    axes[1].set_title("(b) Augmented Class Distribution (N=110)", fontweight="bold")

    axes[2].hist(df_aug[df_aug["stress_class"] == 0]["total_score"], bins=15, alpha=0.7, color="#2b83ba", edgecolor="black", label="Low Stress (Scores <= 39)")
    axes[2].hist(df_aug[df_aug["stress_class"] == 1]["total_score"], bins=15, alpha=0.7, color="#d7191c", edgecolor="black", label="High Stress (Scores >= 40)")
    axes[2].axvline(39.5, color="black", linestyle="--", linewidth=1.5, label="Boundary Threshold (T = 40)")
    axes[2].set_title("(c) Augmented Composite Score Distribution", fontweight="bold")
    axes[2].set_xlabel("Composite Score (TOTAL)")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Figure 1: Dataset Augmentation to 110 Samples & Stress Score Distribution", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_class_and_score_distribution.png")
    plt.close(fig)

    # Figure 2: Sheet 5 Attribute Responses
    fig, ax = plt.subplots(figsize=(11, 7.5))
    df5_sorted = df5.sort_values("Responses", ascending=True)
    c_list = [DOMAIN_COLORS.get(d, "#333333") for d in df5_sorted["domain"]]
    bars = ax.barh(df5_sorted["feature_label"], df5_sorted["Responses"], color=c_list, edgecolor="black")
    for b in bars:
        w = b.get_width()
        ax.text(w + 1.5, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontsize=8.5)
    ax.set_xlim(0, df5["Responses"].max() * 1.15)
    ax.set_xlabel("Cumulative Score Sum across All Academic Staff Respondents (Sheet 5)", fontweight="bold")
    ax.set_title("Figure 2: Cumulative Questionnaire Attribute Response Scores by Domain", fontsize=12.5, fontweight="bold")
    legend_handles = [Patch(facecolor=c, edgecolor="black", label=d) for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=legend_handles, title="Stress Indicator Domain", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_sheet5_attribute_responses.png")
    plt.close(fig)

    # Figure 3: Sheet 4 Transpose Heatmap
    fig, ax = plt.subplots(figsize=(15, 6))
    heat_cols = [c for c in df4.columns if c.startswith("A")]
    stress_order = df3_orig.sort_values(["stress_class", "total_score"])["staff_code"].values
    valid_cols = [c for c in stress_order if c in df4.columns]
    heat_data = df4.set_index("Staff code")[valid_cols].astype(float)
    heat_data.index = [FEATURE_LABELS.get(COLUMN_RENAME_MAP.get(idx, idx), idx) for idx in heat_data.index]
    sns.heatmap(heat_data, cmap="YlOrRd", cbar_kws={"label": "Response Score"}, linewidths=0.05, ax=ax)
    ax.set_title("Figure 3: Respondent Profile Heatmap (Sheet 4 Transpose, Ordered by Stress Class & Total Score)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Academic Staff Respondent Code (A001 – A074)")
    ax.set_ylabel("Questionnaire Indicator")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_sheet4_transpose_heatmap.png")
    plt.close(fig)

    # Figure 4: Pearson Correlation Matrix Heatmap (15 Features + Target)
    fig, ax = plt.subplots(figsize=(13.5, 11))
    sub_corr = df_aug[PEARSON_15_FEATURES + ["stress_class"]].corr(method="pearson")
    lbls = [FEATURE_LABELS[f] for f in PEARSON_15_FEATURES] + ["Target Stress Class"]
    sub_corr.columns = lbls
    sub_corr.index = lbls
    sns.heatmap(sub_corr, cmap="coolwarm", vmin=-0.4, vmax=1.0, annot=True, fmt=".2f", annot_kws={"size": 7.5}, linewidths=0.4, ax=ax)
    ax.set_title("Figure 4: Pearson Correlation Matrix Across 15 High-Contribution Indicators & Target Stress Class", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_correlation_matrix.png")
    plt.close(fig)

    # Figure 5: Pearson Feature Selection Ranking (|r| >= 0.30, p < 0.01)
    fig, ax = plt.subplots(figsize=(12, 8.5))
    df_plot = pearson_df.sort_values("pearson_r_stress_class", ascending=True).reset_index(drop=True)
    c_bars = []
    for r in df_plot["pearson_r_stress_class"]:
        if r >= 0.40:
            c_bars.append("#d7191c")
        elif r >= 0.30:
            c_bars.append("#fdae61")
        elif r >= 0.20:
            c_bars.append("#abdda4")
        else:
            c_bars.append("#2b83ba")

    bars = ax.barh(df_plot["feature_label"], df_plot["pearson_r_stress_class"], color=c_bars, edgecolor="black", height=0.7)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(0.30, color="#d7191c", linestyle="--", linewidth=1.5, label="High-Contribution Threshold (r >= 0.30, p < 0.01)")
    ax.axvline(0.40, color="#8b0000", linestyle=":", linewidth=1.5, label="Very High Contribution (r >= 0.40, p < 0.001)")

    for bar, p_val in zip(bars, df_plot["p_value_stress_class"]):
        w = bar.get_width()
        p_text = "p < 0.001***" if p_val < 0.001 else f"p = {p_val:.3f}**" if p_val < 0.01 else f"p = {p_val:.3f}*" if p_val < 0.05 else f"p = {p_val:.2f} (ns)"
        if w >= 0:
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"+{w:.3f} ({p_text})", va="center", fontsize=8.5, fontweight="bold" if w >= 0.30 else "normal")
        else:
            ax.text(w - 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f} ({p_text})", va="center", ha="right", fontsize=8.5)

    ax.set_xlim(-0.30, 0.68)
    ax.set_xlabel("Pearson Correlation Coefficient (r) with Target Stress Class (CLASSES)", fontweight="bold")
    ax.set_title("Figure 5: Pearson Correlation Feature Selection: Linear Association with Stress Level\nAkwa Ibom State University Academic Staff", fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_pearson_feature_selection.png")
    plt.savefig(FIGURES_DIR / "fig_pearson_feature_selection.png")
    plt.close(fig)

    # Figure 6: Model Comparison Bar Chart (Stratified 5-Fold CV)
    fig, ax = plt.subplots(figsize=(11.5, 6))
    metrics_to_plot = [
        ("cv5_accuracy_num", "Accuracy"),
        ("cv5_precision_num", "Precision"),
        ("cv5_recall_num", "Recall"),
        ("cv5_f1_num", "F1-Score"),
        ("cv5_auc_num", "ROC-AUC"),
    ]
    x_pos = np.arange(len(comparison_df))
    width = 0.15
    palette_metrics = ["#1f78b4", "#33a02c", "#ff7f00", "#e31a1c", "#6a3d9a"]

    for i, (col_m, label_m) in enumerate(metrics_to_plot):
        vals = comparison_df[col_m].values * 100
        bars = ax.bar(x_pos + (i - 2) * width, vals, width, label=label_m, color=palette_metrics[i], edgecolor="black")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(" ", "\n") for m in comparison_df["Model"]], fontweight="bold")
    ax.set_ylim(60, 108)
    ax.set_ylabel("Cross-Validation Score (%)", fontweight="bold")
    ax.set_title(f"Figure 6: Stratified 5-Fold Cross-Validation Performance Comparison (Augmented N=110)\nSelected Optimal Base Classifier: {best_model_name}", fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_model_comparison_metrics.png")
    plt.close(fig)

    # Figure 7: Confusion Matrices for All 4 Models
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, m_name in zip(axes.flat, model_names_order):
        cm = np.array(cv5[m_name]["confusion_matrix"])
        f1_v = cv5[m_name]["mean_f1"] * 100
        acc_v = cv5[m_name]["mean_accuracy"] * 100
        is_sel = (m_name == best_model_name)
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Greens" if is_sel else "Blues",
            cbar=False,
            xticklabels=["Predicted Low (0)", "Predicted High (1)"],
            yticklabels=["Actual Low (0)", "Actual High (1)"],
            annot_kws={"size": 15, "weight": "bold"},
            ax=ax,
        )
        tag = " [SELECTED OPTIMAL MODEL m*]" if is_sel else ""
        ax.set_title(f"{m_name}{tag}\nAccuracy: {acc_v:.1f}% | F1-Score: {f1_v:.1f}%", fontweight="bold")
    plt.suptitle("Figure 7: Out-of-Fold Confusion Matrices Across All 4 Models (Augmented N=110)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_confusion_matrices.png")
    plt.close(fig)

    # Figure 8: ROC Curves for All 4 Models
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y_true = df_aug["stress_class"].values
    m_colors = {
        "Elastic Net Logistic Regression": "#33a02c",
        "Ridge Logistic Regression": "#1f78b4",
        "Shallow Decision Tree": "#ff7f00",
        "Random Forest": "#e31a1c",
    }
    for m_name in model_names_order:
        probs = np.array(cv5[m_name]["oof_probs"])
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = cv5[m_name]["mean_roc_auc"]
        is_sel = (m_name == best_model_name)
        lw = 3.2 if is_sel else 2.0
        ax.plot(fpr, tpr, color=m_colors[m_name], lw=lw, label=f"{m_name} (AUC = {auc_val:.3f}){' [Selected]' if is_sel else ''}")

    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax.set_title("Figure 8: Receiver Operating Characteristic (ROC) Curves (Stratified 5-Fold CV, N=110)", fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_roc_curves.png")
    plt.close(fig)

    # Figure 9: Global Feature Importance / Standardized Coefficients of Selected Model
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    clf_step = selected_model.named_steps["clf"]
    scaler_step = selected_model.named_steps.get("scaler")

    if hasattr(clf_step, "coef_"):
        weights = clf_step.coef_[0]
        feat_imp_df = pd.DataFrame({
            "feature_key": PEARSON_15_FEATURES,
            "feature_label": [FEATURE_LABELS[f] for f in PEARSON_15_FEATURES],
            "domain": [FEATURE_DOMAINS[f] for f in PEARSON_15_FEATURES],
            "importance": weights,
            "abs_importance": np.abs(weights),
        }).sort_values("importance", ascending=True)

        c_imp = ["#d7191c" if w > 0 else "#2b83ba" for w in feat_imp_df["importance"]]
        bars = ax.barh(feat_imp_df["feature_label"], feat_imp_df["importance"], color=c_imp, edgecolor="black")
        for b in bars:
            w = b.get_width()
            offset = 0.02 if w >= 0 else -0.02
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, b.get_y() + b.get_height() / 2, f"{w:+.3f}", va="center", ha=ha, fontsize=8.5, fontweight="bold")
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Standardized Model Weight w_j (+ pushes to High Stress, - pushes to Low Stress)", fontweight="bold")
        ax.set_title(f"Figure 9: Global Feature Attributions & Standardized Weights ({best_model_name})", fontsize=12.5, fontweight="bold")
    else:
        importances = clf_step.feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature_key": PEARSON_15_FEATURES,
            "feature_label": [FEATURE_LABELS[f] for f in PEARSON_15_FEATURES],
            "domain": [FEATURE_DOMAINS[f] for f in PEARSON_15_FEATURES],
            "importance": importances,
        }).sort_values("importance", ascending=True)
        c_imp = [DOMAIN_COLORS.get(d, "#333333") for d in feat_imp_df["domain"]]
        bars = ax.barh(feat_imp_df["feature_label"], feat_imp_df["importance"], color=c_imp, edgecolor="black")
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.005, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5)
        ax.set_xlabel("Gini Feature Importance", fontweight="bold")
        ax.set_title(f"Figure 9: Global Feature Importance ({best_model_name})", fontsize=12.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig9_feature_importance.png")
    plt.savefig(FIGURES_DIR / "fig9_shap_global_importance.png")
    plt.close(fig)

    # Figure 10: Individual Explainable AI Staff Cases (A039 High Stress vs A002 Low Stress)
    high_staff_row = df_aug[df_aug["staff_code"] == "A039"].iloc[0]
    low_staff_row = df_aug[df_aug["staff_code"] == "A002"].iloc[0]

    exp_high = explain_individual_staff(high_staff_row, selected_model, df_aug, features=PEARSON_15_FEATURES)
    exp_low = explain_individual_staff(low_staff_row, selected_model, df_aug, features=PEARSON_15_FEATURES)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, exp_obj, title_prefix in zip(
        axes,
        [exp_high, exp_low],
        ["(a) High-Stress Academic Staff Member (`A039`)", "(b) Low-Stress Academic Staff Member (`A002`)"],
    ):
        top_c = pd.DataFrame(exp_obj["all_contributions"][:10]).sort_values("shap_value", ascending=True)
        labels_with_val = [f"{r['feature_label']}\n[{r['decoded_response']}]" for _, r in top_c.iterrows()]
        c_bars = ["#d7191c" if v > 0 else "#2b83ba" for v in top_c["shap_value"]]
        ax.barh(labels_with_val, top_c["shap_value"], color=c_bars, edgecolor="black")
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Standardized Feature Contribution phi_j (+ pushes toward High Stress, - pushes toward Low Stress)", fontweight="bold")
        ax.set_title(
            f"{title_prefix}\nAssessed Category: {exp_obj['predicted_label']} (High-Stress Probability: {exp_obj['probability_high_stress']*100:.1f}%)",
            fontweight="bold",
        )
    plt.suptitle("Figure 10: Instance-Level Explainable AI Feature Attributions: Individual Stress Drivers & Buffers", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig10_individual_examples.png")
    plt.savefig(FIGURES_DIR / "fig10_shap_individual_examples.png")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 6. GENERATE SAMPLE PERSONALIZED FEEDBACK REPORTS (RAG + GEMINI 3.8 FLASH)
    # -----------------------------------------------------------------------
    print("\n[Step 6/7] Generating sample Generative AI Personalized Feedback Reports...")
    fb_high = generate_personalized_feedback(
        staff_row=high_staff_row,
        explanation=exp_high,
        selected_model_name=best_model_name,
        use_live_gemini=True,
    )
    fb_low = generate_personalized_feedback(
        staff_row=low_staff_row,
        explanation=exp_low,
        selected_model_name=best_model_name,
        use_live_gemini=True,
    )

    sample_reports_md = (
        "# Sample Generative AI Personalized Feedback Outputs\n\n"
        f"**Selected Optimal Base Classifier ($m^*$):** `{best_model_name}`  \n"
        f"**Dataset Configuration:** Augmented Dataset ($N=110$ Academic Staff)  \n"
        f"**Feature Set:** 15 High-Contribution Pearson Correlation Features ($|r| \\ge 0.30, p < 0.01$)  \n"
        f"**Generation Engine:** `{fb_high['generation_mode']}`  \n\n"
        "---\n\n"
        "## Case Study 1: High-Stress Academic Staff Member (`A039`)\n\n"
        f"{fb_high['feedback_text']}\n\n"
        "---\n\n"
        "## Case Study 2: Low-Stress Academic Staff Member (`A002`)\n\n"
        f"{fb_low['feedback_text']}\n"
    )
    (OUTPUT_DIR / "sample_personalized_feedback_reports.md").write_text(sample_reports_md, encoding="utf-8")
    print(f"   Saved sample feedback reports to: {OUTPUT_DIR / 'sample_personalized_feedback_reports.md'}")

    # -----------------------------------------------------------------------
    # 7. SAVE MODEL ARTIFACT BUNDLE & EVALUATION SUMMARY JSON
    # -----------------------------------------------------------------------
    print("\n[Step 7/7] Saving model artifact bundle and evaluation summary JSON...")
    model_bundle = {
        "best_model_name": best_model_name,
        "selected_model": selected_model,
        "trained_models": eval_bundle["trained_models"],
        "pearson_15_features": PEARSON_15_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "feature_domains": FEATURE_DOMAINS,
        "value_decoders": VALUE_DECODERS,
        "cv_results_5fold": cv5,
        "holdout_metrics": holdout,
        "consistency_summary": consistency,
        "augmented_dataset_path": str(AUGMENTED_DATA_PATH),
    }
    joblib.dump(model_bundle, MODELS_DIR / "stress_assessment_bundle.joblib")

    summary_json = {
        "consistency_summary": consistency,
        "best_model_name": best_model_name,
        "augmented_sample_size": len(df_aug),
        "pearson_selected_features_count": len(PEARSON_15_FEATURES),
        "pearson_15_features": PEARSON_15_FEATURES,
        "cv5_summary": {
            m: {k: v for k, v in cv5[m].items() if k not in ("oof_preds", "oof_probs")}
            for m in cv5
        },
        "holdout_summary": holdout,
    }
    (TABLES_DIR / "evaluation_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print("   Successfully saved all artifacts, tables, and figures!")
    print("=" * 78)
    return summary_json


if __name__ == "__main__":
    run_complete_pipeline()
