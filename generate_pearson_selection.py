import sys, os
os.chdir(r"c:\Users\USER\Desktop\Goodnews Project")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from stress_system_core import (
    load_all_project_sheets,
    FEATURE_COLS,
    FEATURE_LABELS,
    FEATURE_DOMAINS,
    FIGURES_DIR,
    TABLES_DIR,
)

sheets = load_all_project_sheets()
df3 = sheets["sheet3_digitized"]
y_target = df3["stress_class"]
total_score = df3["total_score"]

rows = []
for f in FEATURE_COLS:
    r_target, p_target = pearsonr(df3[f], y_target)
    r_total, p_total = pearsonr(df3[f], total_score)
    
    # Tier classification
    abs_r = abs(r_target)
    if abs_r >= 0.40 and p_target < 0.001:
        tier = "Tier 1: High Contribution (|r| >= 0.40, p < 0.001)"
    elif abs_r >= 0.30 and p_target < 0.01:
        tier = "Tier 2: Moderate-High Contribution (0.30 <= |r| < 0.40, p < 0.01)"
    elif p_target < 0.05:
        tier = "Tier 3: Moderate-Low Contribution (0.24 <= |r| < 0.30, p < 0.05)"
    else:
        tier = "Tier 4: Low / Non-Significant Contribution (|r| < 0.24, p >= 0.05)"
        
    rows.append({
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

res_df = pd.DataFrame(rows).sort_values("abs_r_stress_class", ascending=False).reset_index(drop=True)
res_df.to_csv(TABLES_DIR / "pearson_correlation_feature_selection.csv", index=False)
print("Saved pearson_correlation_feature_selection.csv")

# Generate publication-grade Pearson correlation figure
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 8.5))

df_plot = res_df.sort_values("pearson_r_stress_class", ascending=True).reset_index(drop=True)
colors = []
for r in df_plot["pearson_r_stress_class"]:
    if r >= 0.40:
        colors.append("#d7191c")  # High positive
    elif r >= 0.30:
        colors.append("#fdae61")  # Moderate-high positive
    elif r >= 0.20:
        colors.append("#abdda4")  # Moderate-low positive
    else:
        colors.append("#2b83ba")  # Low / negative

bars = ax.barh(df_plot["feature_label"], df_plot["pearson_r_stress_class"], color=colors, edgecolor="black", height=0.7)
ax.axvline(0, color="black", linewidth=1.0)
ax.axvline(0.30, color="#d7191c", linestyle="--", linewidth=1.5, label="High-Contribution Threshold (r >= 0.30, p < 0.01)")
ax.axvline(0.40, color="#8b0000", linestyle=":", linewidth=1.5, label="Very High Contribution (r >= 0.40, p < 0.001)")

for bar, p_val in zip(bars, df_plot["p_value_stress_class"]):
    w = bar.get_width()
    p_text = "p < 0.001***" if p_val < 0.001 else f"p = {p_val:.3f}**" if p_val < 0.01 else f"p = {p_val:.3f}*" if p_val < 0.05 else f"p = {p_val:.2f} (ns)"
    if w >= 0:
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"+{w:.3f} ({p_text})", va="center", fontsize=8.5, fontweight="bold" if w >= 0.30 else "normal")
    else:
        ax.text(w - 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f} ({p_text})", va="center", ha="right", fontsize=8.5)

ax.set_xlim(-0.30, 0.68)
ax.set_xlabel("Pearson Correlation Coefficient (r) with Target Stress Class (CLASSES)", fontweight="bold", fontsize=11)
ax.set_title("Pearson Correlation Feature Selection: Linear Association with Academic Staff Stress Level (N=74)\nAkwa Ibom State University (AKSU)", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig_pearson_feature_selection.png", dpi=150)
plt.close(fig)
print("Saved fig_pearson_feature_selection.png")

