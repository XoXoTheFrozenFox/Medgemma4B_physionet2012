#!/usr/bin/env python3
"""
plot_red_metrics_and_cm.py
- Red key-metrics bar chart (like your screenshot)
- Red confusion matrix
- Prints the classification report

Run:
  python plot_red_metrics_and_cm.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# =========================
# 1) KEY METRICS (edit if needed)
# =========================
run_name = "qlora_main"

metric_names = [
    "json_parse_rate",
    "required_keys_rate_mean",
    "status_accuracy",
    "drivers_f1_mean",
    "checks_f1_mean",
    "rougeL_narrative_mean",
]

# If you have exact numbers, drop them here:
metric_values = [
    1.00,  # json_parse_rate
    1.00,  # required_keys_rate_mean
    1.00,  # status_accuracy
    1.00,  # drivers_f1_mean
    1.00,  # checks_f1_mean
    1.00,  # rougeL_narrative_mean
]


# =========================
# 2) STATUS LABELS + PERFECT PREDICTIONS (matches your report supports)
# =========================
labels = ["stable", "worsening"]
support_stable = 40
support_worsening = 59

y_true = np.array([0] * support_stable + [1] * support_worsening)
y_pred = y_true.copy()  # perfect predictions -> precision/recall/f1 all 1.00


# =========================
# 3) PRINT CLASSIFICATION REPORT
# =========================
print(classification_report(y_true, y_pred, target_names=labels, digits=2))


# =========================
# 4) PLOT: KEY METRICS (RED)
# =========================
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(metric_names, metric_values, color="tab:red")  # <- red bars
ax1.set_ylim(0.0, 1.05)
ax1.set_title(f"Key metrics: {run_name}")
ax1.tick_params(axis="x", labelrotation=45)
for tick in ax1.get_xticklabels():
    tick.set_ha("right")

# Optional: value labels on top of bars
for i, v in enumerate(metric_values):
    ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("key_metrics_red.png", dpi=300)


# =========================
# 5) PLOT: CONFUSION MATRIX (RED)
# =========================
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
im = ax2.imshow(cm, cmap="Reds")  # <- red colormap

ax2.set_xticks(np.arange(len(labels)))
ax2.set_yticks(np.arange(len(labels)))
ax2.set_xticklabels(labels)
ax2.set_yticklabels(labels)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("True")
ax2.set_title("Status Confusion Matrix")

# Annotate cells
thresh = cm.max() / 2.0 if cm.max() > 0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        ax2.text(
            j, i, str(value),
            ha="center", va="center",
            color="white" if value > thresh else "black",
            fontsize=14,
            fontweight="bold"
        )

cbar = plt.colorbar(im, ax=ax2)
cbar.set_label("Count")

plt.tight_layout()
plt.savefig("confusion_matrix_red.png", dpi=300)

plt.show()
