import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
n_samples = 200
n_features = 2
shift = 3.0

# Generate dataset with batch effect (two distinct clusters)
X1 = np.random.randn(n_samples // 2, n_features)
X2 = np.random.randn(n_samples // 2, n_features) + np.array([shift, 0])
X_batch = np.vstack((X1, X2))
y_batch = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Generate dataset without batch effect (blended cluster), but with group labels
X_no_batch = np.random.randn(n_samples, n_features)
y_no_batch = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
perm = np.random.permutation(n_samples)
X_no_batch = X_no_batch[perm]
y_no_batch = y_no_batch[perm]

def set_white(ax):
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('white')
        legend.get_frame().set_alpha(0)  # transparent legend background
        legend.get_frame().set_edgecolor('white')  # white border for legend

    # Set white border for axes
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)

# Font size settings
TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 16
LEGEND_SIZE = 24

# Choose two colors from tab20
tab20 = plt.get_cmap('tab20')
color1 = tab20(19)
color2 = tab20(2)

# Plot dataset with batch effect
fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='none')
ax1.scatter(X_batch[y_batch == 0, 0], X_batch[y_batch == 0, 1], label='Date/Operator 1', color=color1)
ax1.scatter(X_batch[y_batch == 1, 0], X_batch[y_batch == 1, 1], label='Date/Operator 2', color=color2)
ax1.set_title("Dataset with Batch Effect", fontsize=TITLE_SIZE)
ax1.set_xlabel("PC 1 or t-SNE 1", fontsize=LABEL_SIZE)
ax1.set_ylabel("PC 2 or t-SNE 2", fontsize=LABEL_SIZE)
ax1.tick_params(axis='both', labelsize=TICK_SIZE)
set_white(ax1)
fig1.patch.set_alpha(0)  # transparent figure background
plt.tight_layout()
plt.savefig("batch_effect_comparison.png", dpi=300, bbox_inches='tight', transparent=True)

# Plot dataset without batch effect (blended), colored by group
fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='none')
ax2.scatter(
    X_no_batch[y_no_batch == 0, 0], 
    X_no_batch[y_no_batch == 0, 1], 
    label='Date/Operator 1',
    color=color1
)
ax2.scatter(
    X_no_batch[y_no_batch == 1, 0], 
    X_no_batch[y_no_batch == 1, 1], 
    label='Date/Operator 2',
    color=color2
)
ax2.set_title("Dataset without Batch Effect", fontsize=TITLE_SIZE)
ax2.set_xlabel("PC 1 or t-SNE 1", fontsize=LABEL_SIZE)
ax2.set_ylabel("PC 2 or t-SNE 2", fontsize=LABEL_SIZE)
# ax2.legend(fontsize=LEGEND_SIZE, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.tick_params(axis='both', labelsize=TICK_SIZE)
set_white(ax2)
fig2.patch.set_alpha(0)  # transparent figure background
plt.tight_layout()
plt.savefig("no_batch_effect_comparison.png", dpi=300, bbox_inches='tight', transparent=True)
