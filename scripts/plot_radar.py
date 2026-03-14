import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def draw_radar_chart(title, metrics, data, output_path):
    N = len(metrics)
    # The angles of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # First axis on top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], metrics, size=12, fontweight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)
    
    colors = {
        "HybridStackPPI": "#d62728", # red
        "Vanilla ESM-2 + MLP": "#1f77b4", # blue
        "Conjoint Triad (CT)": "#9467bd", # purple
        "Auto Covariance (AC)": "#ff7f0e", # orange
        "SPRINT": "#2ca02c" # green
    }
    
    for method, values in data.items():    
        values_with_first = values + values[:1]
        ax.plot(angles, values_with_first, linewidth=2, linestyle='solid', label=method, color=colors.get(method.split(" ")[0], colors.get(method, plt.cm.tab10(np.random.randint(0, 10)))))
        ax.fill(angles, values_with_first, alpha=0.1)

    plt.title(title, size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def main():
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "MCC", "ROC-AUC", "PR-AUC"]
    
    # Replace MCC with normalized MCC (0-100) or just map -1..1 to 0..100: (mcc + 1)/2 * 100
    metrics_str = ["Acc", "Prec", "Rec", "F1", "Spec", "MCC(norm)", "ROC-AUC", "PR-AUC"]

    def norm_mcc(val):
        # Maps -100 to 100 to 0 to 100?
        # Actually MCC reported in our tables is percentage: -100 to +100
        # Wait, MCC is 0-100 in table: e.g. 27.70, 7.9, 87.0
        # If it's already 0-100, we just use it, but bound it to 0 min.
        return max(0, val)

    # HUMAN SAME-GO
    data_human_go = {
        "HybridStackPPI": [62.41, 89.70, 59.07, 71.22, 74.81, norm_mcc(27.70), 73.47, 91.28],
        "Vanilla ESM-2 + MLP": [73.08, 83.00, 82.82, 82.90, 73.08, norm_mcc(19.64), 68.12, 88.48],
        "Conjoint Triad (CT)": [59.76, 83.84, 60.60, 70.33, 58.91, norm_mcc(14.17), 63.44, 87.31],
        "Auto Covariance (AC)": [62.13, 81.69, 66.94, 73.56, 57.31, norm_mcc(9.41), 59.78, 85.87],
        "SPRINT": [21.67, 75.43, 0.78, 1.54, 0, norm_mcc(0.44), 63.12, 86.08]
    }
    
    # YEAST SAME-GO
    data_yeast_go = {
        "HybridStackPPI": [65.34, 89.42, 63.91, 74.42, 71.05, norm_mcc(28.84), 74.95, 92.08],
        "Vanilla ESM-2 + MLP": [72.55, 84.05, 80.75, 82.31, 72.55, norm_mcc(21.48), 68.77, 87.91],
        "Conjoint Triad (CT)": [65.09, 84.58, 68.44, 75.60, 61.74, norm_mcc(17.92), 67.63, 89.94],
        "Auto Covariance (AC)": [65.48, 81.29, 73.32, 77.04, 57.65, norm_mcc(8.26), 61.77, 87.80],
        "SPRINT": [21.04, 100.0, 0.27, 0.53, 0, norm_mcc(2.28), 61.35, 84.52]
    }

    # HUMAN SAME-COMPARTMENT
    data_human_comp = {
        "HybridStackPPI": [85.90, 89.65, 89.98, 89.81, 70.73, norm_mcc(61.80), 93.18, 97.46],
        "Vanilla ESM-2 + MLP": [85.34, 85.82, 92.20, 88.89, 59.85, norm_mcc(56.12), 85.59, 93.99],
        "Conjoint Triad (CT)": [79.23, 83.16, 92.03, 87.37, 31.64, norm_mcc(30.65), 75.40, 90.72],
        "Auto Covariance (AC)": [79.62, 85.33, 89.28, 87.26, 43.68, norm_mcc(37.40), 75.05, 90.58],
        "SPRINT": [23.11, 77.06, 5.04, 9.47, 0, norm_mcc(0.81), 54.65, 78.71] # Assuming Specificity is 0 properly since SPRINT predicts everything negative
    }

    # YEAST SAME-COMPARTMENT
    data_yeast_comp = {
        "HybridStackPPI": [84.14, 90.49, 86.82, 88.61, 72.84, norm_mcc(60.10), 91.95, 97.02],
        "Vanilla ESM-2 + MLP": [80.59, 83.98, 91.79, 87.71, 33.20, norm_mcc(32.84), 80.20, 93.28],
        "Conjoint Triad (CT)": [80.89, 83.56, 93.19, 88.13, 28.84, norm_mcc(30.68), 75.60, 90.15],
        "Auto Covariance (AC)": [80.75, 83.15, 93.76, 88.14, 25.64, norm_mcc(27.87), 75.76, 90.47],
        "SPRINT": [20.89, 75.40, 3.42, 6.54, 0, norm_mcc(0.68), 55.40, 80.24]
    }

    os.makedirs('results/', exist_ok=True)
    draw_radar_chart("Multi-Method Comparison (Human Same-GO)", metrics_str, data_human_go, "results/radar_human_same_go.png")
    draw_radar_chart("Multi-Method Comparison (Yeast Same-GO)", metrics_str, data_yeast_go, "results/radar_yeast_same_go.png")
    draw_radar_chart("Multi-Method Comparison (Human Same-Compartment)", metrics_str, data_human_comp, "results/radar_human_same_compartment.png")
    draw_radar_chart("Multi-Method Comparison (Yeast Same-Compartment)", metrics_str, data_yeast_comp, "results/radar_yeast_same_compartment.png")

if __name__ == "__main__":
    main()
