import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, set_link_color_palette
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import seaborn as sns
from matplotlib.colors import to_hex
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

#########
threshold = 0.4
number_example_spectrum = 50
#########

# Load JSON file
with open('metadata_with_embeddings.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
metadata_df = pd.DataFrame(data)

# Extract embeddings as a NumPy array
embeddings = np.array(metadata_df['embeddings'].tolist())

# Perform hierarchical clustering
complete_clustering = linkage(embeddings, method="complete", metric="euclidean")

# Assign cluster labels to each observation
cluster_labels = cut_tree(complete_clustering, height=threshold).reshape(-1,)
metadata_df['Cluster'] = cluster_labels

# Sort clusters by size
cluster_sizes = metadata_df['Cluster'].value_counts().sort_values(ascending=True)
cluster_order = cluster_sizes.index.tolist()
cluster_map = {old: new for new, old in enumerate(cluster_order)}
metadata_df['Cluster'] = metadata_df['Cluster'].map(cluster_map)
num_clusters = len(cluster_order)

# print a summary of the clusters
print("Cluster Summary:")
for i in range(num_clusters):
    cluster_data = metadata_df[metadata_df['Cluster'] == i]
    print(f"Cluster {i}: {len(cluster_data)} spectra")

# Define colors
colors = sns.color_palette("Paired", num_clusters)
colors = [to_hex(color) for color in colors]
set_link_color_palette(colors)

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram_data = dendrogram(
    complete_clustering,
    labels=metadata_df.index,
    color_threshold=threshold,
    above_threshold_color="#808080"
)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title(f"Dendrogram with Threshold at {threshold}")
plt.xlabel("Spectrum Index")

# Add legend
handles = [Line2D([0], [0], color=colors[i], lw=4) for i in range(num_clusters)]
plt.legend(handles=handles, labels=[f"Cluster {i}" for i in range(num_clusters)], title="Cluster", loc="upper right")
plt.savefig(f'Dendrogram with threshold at {threshold}.png')

# Example spectra for each cluster
fig, axes = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
if num_clusters == 1:
    axes = [axes]
for i in range(num_clusters):
    cluster_data = metadata_df[metadata_df['Cluster'] == i]
    cluster_data = cluster_data.reset_index()
    raman_shift = cluster_data['raman_shift'][0]
    number_spectrum = min(len(cluster_data), number_example_spectrum)
    for j in range(number_spectrum):
        intensity_values = cluster_data['intensity'][j][0]
        sns.lineplot(x=raman_shift, y=intensity_values, color=colors[i], alpha=0.5, ax=axes[i])
    axes[i].plot([], label=f"Cluster {i}", color=colors[i])
    axes[i].legend(loc='upper right')
    axes[i].set_ylabel("Normalized Intensity")
    axes[i].set_xlabel("Raman Shift (cm^-1)")
    # axes[i].set_title(f"Cluster has {len(cluster_data)} Spectra")
plt.suptitle(f"Example Spectra for Each Cluster (clustering_threshold_{threshold})", fontsize=16)
plt.tight_layout()
plt.savefig(f'Example Spectra clustering with threshold {threshold}.png')

# Histogram of different stages in each cluster
category_map = {
    0: "Healthy", 
    1: "Early Stage", 
    2: "Late Stage"
}
metadata_df['staging'] = metadata_df['staging'].map(category_map)
fig, axes = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
if num_clusters == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    cluster_data = metadata_df[metadata_df['Cluster'] == i]
    sns.histplot(
        cluster_data['staging'], 
        color=colors[i], 
        ax=ax, 
        kde=False
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Healthy", "Early Stage", "Late Stage"])
    ax.set_title(f"Cluster {i} Histogram")
plt.suptitle(f"Histogram of Different Stages in Each Cluster (clustering_threshold_{threshold})")
plt.tight_layout()
plt.savefig(f'Histogram_of_Different_Stages_in_Each_Cluster threshold{threshold}.png')

# Histogram of different sample types in each cluster
fig, axes = plt.subplots(num_clusters, 1, sharex=True, figsize=(10, num_clusters * 2))
if num_clusters == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    cluster_data = metadata_df[metadata_df['Cluster'] == i]
    sns.histplot(
        cluster_data['sample_type'], 
        color=colors[i], 
        ax=ax, 
        kde=False
    )
    ax.set_title(f"Cluster {i} Histogram")
    ax.set_xlabel("Sample Type")
    ax.set_ylabel("Spectrum Count")
plt.suptitle(f"Histogram of Different Sample Types in Each Cluster (clustering threshold {threshold})")
plt.tight_layout()
plt.savefig(f'Histogram of Different Sample Types in Each Cluster threshold {threshold}.png')

# visualize the embeddings in 2D space by dendrogram cluster
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
embeddings_2d_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
embeddings_2d_df['Cluster'] = metadata_df['Cluster']
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    cluster_data = embeddings_2d_df[embeddings_2d_df['Cluster'] == i]
    plt.scatter(cluster_data['x'], cluster_data['y'], color=colors[i], label=f"Cluster {i}", alpha=0.7)
plt.title(f"t-SNE Visualization of Embeddings by Cluster (clustering threshold {threshold})")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster", loc="upper right")
plt.savefig(f't-SNE Visualization of Embeddings by Cluster threshold{threshold}.png')

# visualize the embeddings in 2D space by staging and sample type
embeddings_2d_df['staging'] = metadata_df['staging']
embeddings_2d_df['sample_type'] = metadata_df['sample_type']
embeddings_2d_df['patient_id'] = metadata_df['patient_id']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=embeddings_2d_df, x='x', y='y', hue='staging', palette='rocket', alpha=0.7)
plt.title(f"t-SNE Visualization of Embeddings by Staging (clustering threshold {threshold})")
plt.legend(title="Staging")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(f't-SNE Visualization of Embeddings by Staging threshold {threshold}.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=embeddings_2d_df, x='x', y='y', hue='sample_type', palette='coolwarm', alpha=0.7)
plt.title(f"t-SNE Visualization of Embeddings by Sample Type (clustering threshold {threshold})")
plt.legend(title="Sample Type")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(f't-SNE Visualization of Embeddings by Sample Type threshold {threshold}.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=embeddings_2d_df, x='x', y='y', hue='patient_id', palette='Spectral', alpha=0.7)
plt.title(f"t-SNE Visualization of Embeddings by Patient ID (clustering threshold {threshold})")
plt.legend(title="Cluster")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(f't-SNE Visualization of Embeddings by Patient ID threshold {threshold}.png')

######### save low quality spectra to json file #########
##### Change the cluster number #####
low_quality_spectra = []
for i in range(4):
    cluster_data = metadata_df[metadata_df['Cluster'] == i]
    # re index the cluster data
    cluster_data = cluster_data.reset_index()
    low_quality_spectra_for_this_cluster = {}
    for j in range(len(cluster_data)):
        patient_id = int(cluster_data['patient_id'][j])
        sample_type = cluster_data['sample_type'][j]
        spectrum_id = int(cluster_data['spectrum_id'][j])
        if (patient_id, sample_type) not in low_quality_spectra_for_this_cluster:
            low_quality_spectra_for_this_cluster[(patient_id, sample_type)] = []
        low_quality_spectra_for_this_cluster[(patient_id, sample_type)].append(spectrum_id)

    for key, value in low_quality_spectra_for_this_cluster.items():
        if len(value) > 1:
            low_quality_spectra.append([key[0], key[1], value])

# save low quality spectra to json file
with open('low_quality_spectra.json', 'w') as f:
    json.dump(low_quality_spectra, f)

