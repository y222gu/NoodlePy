import tensorflow as tf
import os
import pandas as pd
import plotly.express as px
import umap

# Directory to save TensorBoard logs and embedding files
log_dir = '/mnt/c/Users/Yifei/Documents/NoodlePy/output_plots/tensorflow_embedding'


config = tf.compat.v1.placeholder(tf.compat.v1.ConfigProto)

# Create the projector object
projector = tf.compat.v1.summary.ProjectorConfig()

# Add embedding
embedding = projector.embeddings.add()
embedding.tensor_name = 'embedding_tensor'
embedding.metadata_path = 'labels.tsv'
embedding.tensor_path = 'embeddings.tsv'

# Save the configuration file
config_path = os.path.join(log_dir, 'projector_config.pbtxt')
with open(config_path, 'w') as f:
    f.write(str(projector))

# Create a summary writer
summary_writer = tf.compat.v1.summary.FileWriter(log_dir)

# Add the configuration to the summary writer
tf.compat.v1.summary.FileWriter(log_dir).add_graph(tf.compat.v1.get_default_graph())
tf.compat.v1.summary.ProjectorConfig()

# Load embeddings and labels
embeddings = pd.read_csv('embeddings.tsv', sep='\t', header=None)
labels = pd.read_csv('labels.tsv', sep='\t', header=None, names=['label'])

# Run UMAP to reduce dimensions to 3D
umap_embeddings = umap.UMAP(n_components=3).fit_transform(embeddings)

# Create a DataFrame with UMAP embeddings and labels
df = pd.DataFrame(umap_embeddings, columns=['x', 'y', 'z'])
df['label'] = labels['label']

# Plot using plotly
fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', 
                    symbol='label', size_max=10, opacity=0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
