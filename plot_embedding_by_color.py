import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict

# Load JSON file
file_path = r'C:\Users\yifei\Documents\NoodlePy\output_plots\2023_03_03_Bec_HNC_batch_corrected_model_embeddings.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract embeddings, dates, staging, and patient IDs
embeddings = np.array([entry['embeddings'] for entry in data])
dates = [entry['date'] for entry in data]
staging = [entry['staging'] for entry in data]
patient_ids = [entry['patient_id'] for entry in data]

# Apply T-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Function to plot TSNE with color bar
def plot_tsne(embeddings_2d, labels, title):
    values = labels
    unique_vals = sorted(set(values))
    cmap = plt.cm.get_cmap('cool', len(unique_vals))
    point_colors = [unique_vals.index(v) for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=point_colors,
        cmap=cmap,
        alpha=0.6,
        picker=True
    )
    
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(unique_vals)))
    cbar.set_ticklabels([str(v) for v in unique_vals])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(title)
    
    ax.set_title(f"2D T-SNE of Spectra Embedding - Colored by {title}")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")
    plt.show()

# Plot with different label groupings
# plot_tsne(embeddings_2d, dates, "Date")
plot_tsne(embeddings_2d, staging, "Staging")
# plot_tsne(embeddings_2d, patient_ids, "Patient ID")
