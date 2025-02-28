import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import torch
import csv
from sklearn.cluster import DBSCAN

class EmbeddingViewer:
    def __init__(self, embeddings=None, label_dict_list=None, embeddings_file_path=None, metadata_file_path=None, map=None, exp_name = "test"):

        self.exp_name = exp_name
        if map is None:
            self.map = {0:'lightskyblue', 1:'#deb209', 2:'#e97132', 3:'#e97132', 4:'#e97132', # staging gold mediumslateblue purple #a484df
                        'Male':'xkcd:blue', 'Female':'xkcd:golden brown', # gender
                        'White':'xkcd:salmon', # race
                        'plasma':"#de8749", 'saliva':"#8a75da",# ">", "x" 'plasma':"#de8749", 'saliva':"#8a75da"
                        'default_color':'teal', 'default_marker':'*'} #  'plasma':"circle", 'saliva':"cross"
        else:
            self.map = map
        
        if embeddings_file_path and metadata_file_path:
            self.embeddings, self.labels = self.load_embeddings_from_file(embeddings_file_path, metadata_file_path)
        elif embeddings is not None and label_dict_list is not None:
            self.embeddings = embeddings
            self.labels = EmbeddingViewer.reorganize_dicts(label_dict_list)
        else:
            raise ValueError("Provide either (embeddings and label_dict_list) or (embedding_file_name and metadata_file_name)")

    @staticmethod
    def map_label(labels, map):
        mapped_labels = []
        for label in labels:
            if label in map:
                mapped_labels.append(map[label])
            else:
                raise ValueError("Label not found in the color map")
        return mapped_labels
    
    @staticmethod
    def map_label_location(labels):
        # Indices to step through colormap
        color_values = np.linspace(0.0, 1.0, 50)
        colormap = mpl.colormaps['gist_rainbow']
        map = colormap(color_values)
        # find the unique values in the labels
        unique_labels = list(set(labels))
        # sort the unique values from smallest to largest
        unique_labels.sort()

        # create a dictionary to map the unique values to a color
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = map[i]

        mapped_labels = []
        for label in labels:
            mapped_labels.append(color_map[label])
        return mapped_labels
    
    @staticmethod
    def map_label_patient_id(labels):
        num_color = np.unique(labels).shape[0]
        # Indices to step through colormap
        color_values = np.linspace(0.0, 1.0, num_color)
        colormap = mpl.colormaps['gist_rainbow']
        map = colormap(color_values)
        # find the unique values in the labels
        unique_labels = list(set(labels))
        # sort the unique values from smallest to largest
        unique_labels.sort()

        # create a dictionary to map the unique values to a color
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = map[i]

        mapped_labels = []
        for label in labels:
            mapped_labels.append(color_map[label])
        return mapped_labels
    

    @staticmethod
    def compute_tsne(embeddings, dim=3, **kwargs):
        tsne = TSNE(n_components=dim, **kwargs)
        embeddings_tsne = tsne.fit_transform(embeddings)
        return embeddings_tsne
    
    def save_files_for_tf_embedding_projector(self):
        metadata_df = pd.DataFrame(self.labels)

        # Add embeddings to the DataFrame as a column
        metadata_df['embeddings'] =  self.embeddings.tolist()

        # save to output_plots folder as a json file
        metadata_file_name = os.path.join(os.getcwd(), "output_plots", self.exp_name + "_model_embeddings.json")
        metadata_df.to_json(metadata_file_name, orient='records')

    @staticmethod
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

    @staticmethod
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
    
    def tsne2d(self, label_name_for_color='staging'):
        embeddings_2d = EmbeddingViewer.compute_tsne(self.embeddings, dim=2)

        # Save the 2d embedding file
        # embedding_file_name = os.path.join(os.getcwd(), "output_plots", self.exp_name +"_embedding.tsv")        
        # with open(embedding_file_name, 'w') as f:
        #     for embedding in self.embeddings:
        #         embedding_str = '\t'.join(map(str, embedding))
        #         f.write(embedding_str + '\n')

        embeddingsdf = pd.DataFrame()
        embeddingsdf['x'] = embeddings_2d[:, 0]
        embeddingsdf['y'] = embeddings_2d[:, 1]

        if label_name_for_color == 'location':
            colors = EmbeddingViewer.map_label_location(self.labels[label_name_for_color])
        elif label_name_for_color == 'patient_id':
            colors = EmbeddingViewer.map_label_patient_id(self.labels[label_name_for_color])
        else:
            colors = EmbeddingViewer.map_label(self.labels[label_name_for_color], self.map)

        fig, ax = plt.subplots(figsize=(14, 11))

        # Scatter points, set alpha low to make points translucent
        for i in range(len(embeddingsdf.x)):
            ax.scatter(embeddingsdf.x[i], embeddingsdf.y[i], c=colors[i], marker=">", alpha=1, s=250)
        # plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        # remove the frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # change the color of the remaining spines to white
        ax.spines['bottom'].set_color('white')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_linewidth(3)
        # change the color of the ticks to white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', which= 'major',colors='white', labelsize=25)
        ax.tick_params(axis='y', which= 'major',colors='white', labelsize=25)
        ax.title.set_color('white')

        # tight layout
        plt.tight_layout()

        #set the background color to black
        fig.patch.set_facecolor('white')
        save_path = os.path.join(os.getcwd(), "output_plots", self.exp_name + "_2d_tsne_plot_by_" + label_name_for_color + ".svg")
        plt.savefig(save_path, transparent=True)


        dbscan = DBSCAN(eps=0.01, min_samples=5)
        labels = dbscan.fit(self.embeddings)
        labels = labels.labels_
        # change -1 to  10
        labels[labels == -1] = 10

        # map the labels to colors 
        # create a colormap
        cmap = mpl.colormaps['gist_rainbow']
        # create a list of colors
        colors = cmap(np.linspace(0, 1, len(np.unique(labels))))
        # create a dictionary to map the labels to colors
        label_color_map = {}
        for i, label in enumerate(np.unique(labels)):
            label_color_map[label] = colors[i]

        # find the unique labels
        unique_labels = np.unique(labels)
        print(unique_labels)
        # visualize the clusters
        plt.figure(figsize=(10, 8))
        for i in range(len(embeddingsdf.x)): 
            plt.scatter(embeddingsdf.x[i], embeddingsdf.y[i], c=label_color_map[labels[i]], marker=">", alpha=1, s=250) #marker=markers[i],
        plt.title("DBSCAN Clustering")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(os.path.join(os.getcwd(), "output_plots", self.exp_name + "_dbscan_clustering" + ".png"))


    def tsne3d(self, embeddings_3d, labels_for_color, labels_for_marker):
        map = {0:'green', 1:'gold', 2:'orangered', 3:'red', 4:'purple', # staging
                'Male':'xkcd:blue', 'Female':'xkcd:golden brown', # gender
                'White':'xkcd:salmon', # race
                'plasma':"cross", 'saliva':"circle",
                'default_color':'teal', 'default_marker':'*'} #  'plasma':"circle", 'saliva':"cross"

        colors = EmbeddingViewer.map_label(labels_for_color, map, type='to_color')
        markers = EmbeddingViewer.map_label(labels_for_marker, map, type='to_marker')

    
        embeddings_3d = EmbeddingViewer.compute_tsne(embeddings_3d, dim=3)
        fig = go.Figure(data=go.Scatter3d(x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2], 
                                          mode='markers', marker=dict(color=colors, size=10, symbol=markers)))
        
        # set the size of the plot
        fig.update_layout(width=800, height=800)

        fig.update_layout(
            title=self.exp_name + "_3D_embedding",
            paper_bgcolor='rgba(1,1,1,1)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(showgrid=False, backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(showgrid=False, backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(showgrid=False, backgroundcolor='rgba(0,0,0,0)'),
                bgcolor='rgba(0,0,0,0)'
            ))
        
        # display the plot
        # save the plot
        fig.write_html(os.path.join(os.getcwd(), "output_plots", self.exp_name + "_3D_embedding" + ".html"))

    def tsne2d_by_patient(self, title='t-SNE 2D Visualization by Patient'):
        embeddings_2d = EmbeddingViewer.compute_tsne(self.embeddings, dim=2)
        embeddingsdf = pd.DataFrame()
        embeddingsdf['x'] = embeddings_2d[:, 0]
        embeddingsdf['y'] = embeddings_2d[:, 1]

        colors = EmbeddingViewer.map_label(self.labels['patient_id'], self.map, type='to_color')
        markers = EmbeddingViewer.map_label(self.labels['sample_type'], self.map, type='to_marker')

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter points, set alpha low to make points translucent
        for i in range(len(embeddingsdf.x)):
            ax.scatter(embeddingsdf.x[i], embeddingsdf.y[i], c=colors[i], marker=markers[i], alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        save_path = os.path.join(os.getcwd(), "output_plots", title + ".png")
        plt.savefig(save_path)

    @staticmethod
    def load_embeddings_from_file(embedding_file_path='embeddings', metadata_file_path='metadata'):
        # Load embeddings
        embeddings = []
        with open(embedding_file_path, 'r') as f:
            for line in f:
                embedding = list(map(float, line.strip().split('\t')))
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_file_path, delimiter='\t')
        labels = metadata_df.to_dict(orient='list')
        
        return embeddings, labels

    def dbscan(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.embeddings)
        # visualize the clusters
        plt.figure(figsize=(10, 8)) 
        plt.scatter(self.embeddings[:, 0], self.embeddings[:, 1], c=labels, cmap='rainbow')
        plt.title("DBSCAN Clustering")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(os.path.join(os.getcwd(), "output_plots", self.exp_name + "_dbscan_clustering" + ".png"))
        return labels

if __name__ == '__main__':
    embedding_file_path = os.path.join(r"C:\Users\Yifei\Documents\NoodlePy\output_plots\HNC_pretrained_embedding.tsv")
    metadata_file_path = os.path.join(r"C:\Users\Yifei\Documents\NoodlePy\output_plots\HNC_pretrained_metadata.tsv")
    embedding_viewer = EmbeddingViewer(embeddings_file_path = embedding_file_path, metadata_file_path = metadata_file_path, exp_name = "training_data")
    embedding_viewer.tsne2d(label_name_for_color='patient_id')
    embedding_viewer.tsne2d(label_name_for_color='staging')
    embedding_viewer.tsne2d(label_name_for_color='sample_type')