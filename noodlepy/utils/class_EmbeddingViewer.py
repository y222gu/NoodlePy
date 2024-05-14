import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import torch
import csv

class EmbeddingViewer():
    def __init__(self, embeddings, label_dict_list, map=None):
        if map is None:
            self.map = {0:'green', 1:'gold', 2:'orangered', 3:'red', 4:'purple', # staging
                        'Male':'xkcd:blue', 'Female':'xkcd:golden brown', # gender
                        'White':'xkcd:salmon', # race
                        'plasma':"P", 'saliva':">",
                        'default_color':'teal', 'default_marker':'*'} #  'plasma':"circle", 'saliva':"cross"
        else:
            self.map = map
        self.embeddings = embeddings
        self.labels = EmbeddingViewer.reorganize_dicts(label_dict_list)

    def map_label(self, labels, type):
        if type == 'to_color':
            default_label = self.map['default_color']
        elif type == 'to_marker':
            default_label = self.map['default_marker']
        else:
            raise ValueError("Invalid type. Choose 'to_color' or 'to_marker'")
        
        mapped_labels = []
        for label in labels:
            if label in self.map:
                mapped_labels.append(self.map[label])
            else:
                mapped_labels.append(default_label)
        return mapped_labels

    def compute_tsne(embeddings, dim=3, **kwargs):
        tsne = TSNE(n_components=dim, **kwargs)
        embeddings_tsne = tsne.fit_transform(embeddings)
        return embeddings_tsne
    
    def save_files_for_tf_embedding_projector(self, embedding_file_name='embeddings', metadata_file_name='metadata'):
        embedding_file_name = os.path.join(os.getcwd(), "output_plots", embedding_file_name + ".tsv")        
        with open(embedding_file_name, 'w') as f:
            for embedding in self.embeddings:
                embedding_str = '\t'.join(map(str, embedding))
                f.write(embedding_str + '\n')

        
        metadata_file_name = os.path.join(os.getcwd(), "output_plots", metadata_file_name + ".tsv")
        with open(metadata_file_name, 'w', newline='\n') as tsvfile:
            tsv_writer = csv.writer(tsvfile, delimiter='\t')
            tsv_writer.writerow(self.labels.keys())
            for row in zip(*self.labels.values()):
                tsv_writer.writerow(row)
    
    def break_up_dictionary_list(input_list):
        def break_up_dictionary(input_dict):
            num_entries = len(input_dict['patient_id'])
            result = []
            for i in range(num_entries):
                entry = {}
                for key, value in input_dict.items():
                    if isinstance(value, list):
                        entry[key] = value[i]
                    else:
                        entry[key] = value[i].item()
                result.append(entry)
            return result

        # Convert tensor to list
        for d in input_list:
            for key, value in d.items():
                if torch.is_tensor(value):
                    d[key] = value.tolist()

        output_list = []
        for d in input_list:
            output_list.extend(break_up_dictionary(d))
        return output_list

    def reorganize_dicts(list_of_label_dicts):
        reorganized_label_dict = {}
        for d in list_of_label_dicts:
            for key, value in d.items():
                if torch.is_tensor(value):
                    value = value.tolist()
                if key not in reorganized_label_dict:
                    reorganized_label_dict[key] = []
                reorganized_label_dict[key].extend(value)
        return reorganized_label_dict
    
    def tsne2d(self, label_name_for_color='staging', label_name_for_marker='sample_type', title="2d_tsne_plot"):
        embeddings_2d = EmbeddingViewer.compute_tsne(self.embeddings, dim=2)
        embeddingsdf = pd.DataFrame()
        embeddingsdf['x'] = embeddings_2d[:, 0]
        embeddingsdf['y'] = embeddings_2d[:, 1]

        colors = EmbeddingViewer.map_label(self, labels=self.labels[label_name_for_color], type='to_color')
        markers = EmbeddingViewer.map_label(self, labels=self.labels[label_name_for_marker], type='to_marker')

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter points, set alpha low to make points translucent
        for i in range(len(embeddingsdf.x)):
            ax.scatter(embeddingsdf.x[i], embeddingsdf.y[i], c=colors[i], marker=markers[i] ,alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        save_path = os.path.join(os.getcwd(), "output_plots", title + ".png")
        plt.savefig(save_path)


    def tsne3d(self, embeddings_3d, labels_for_color, labels_for_marker, title='t-SNE 3D Visualization'):

        colors = EmbeddingViewer.map_label(self, labels=labels_for_color, type='to_color')
        markers = EmbeddingViewer.map_label(self, labels=labels_for_marker, type='to_marker')
    
        embeddings_3d = EmbeddingViewer.compute_tsne(embeddings_3d, dim=3)
        fig = go.Scatter3d(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], mode='markers', marker=dict(color=colors, size=5, symbol=markers))
                
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False),
                bgcolor='rgba(0,0,0,0)'
            ))
        fig.write_html(os.path.join(os.getcwd(), "output_plots", title + ".html"))
