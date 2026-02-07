import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix values
# rows = true labels, cols = predicted labels
cm = np.array([
    [20, 0],   # true stable
    [0, 30]    # true worsening
])

labels = ["stable", "worsening"]

fig, ax = plt.subplots(figsize=(5, 5))

# Plot matrix using a blue-white colormap
im = ax.imshow(cm, cmap="Blues")

# Axis labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Status Confusion Matrix")

# Annotate cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        ax.text(
            j, i, value,
            ha="center", va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=14,
            fontweight="bold"
        )

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Count")

plt.tight_layout()
plt.show()
