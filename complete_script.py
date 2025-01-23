import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import argparse   
from sklearn.metrics.pairwise import cosine_similarity
import json

# Change here
NUM_CLUSTERS = 10

# datasets = ['humaneval', 'magpie_filtered', 'magpie_deepseek',  'magpie_qwen']

color_dict = {
    -1: 'black',
    0: 'red',
    1: 'orange',
    2: 'blue',
    3: 'yellow',
    4: 'cyan',
    5: 'green',
    6: 'magenta',
    7: 'purple',
    8: 'brown',
    9: 'pink',
    10: 'gray',
    11: 'olive',
    12: 'navy',
    13: 'teal',
    14: 'lime',
    15: 'indigo',
    16: 'gold',
    17: 'coral',
    18: 'violet',
    19: 'turquoise',
    20: 'silver',
    21: 'maroon'
}

def read_stat_file(stats_file_path):
    with open(stats_file_path, 'r') as f:
        data = json.loads(f.read())
    
    return data


def extract_cluster_size(data_dict, dataset_size):

    cluster_count = list(data_dict['Total Counts'].values())
    # print(cluster_count)
    # return cluster_count
    cluster_percentage = [count/dataset_size for count in cluster_count]
    
    return cluster_percentage


def generate_clusters(np_memmap, save_centroid_path=None):
    kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(np_memmap)
    cluster_indices = kmeans.labels_

    cluster_indices_shape = cluster_indices.shape

    cluster_indices = cluster_indices.reshape((cluster_indices_shape[0], 1))

    # df = pd.DataFrame(cluster_indices, columns=['cluster_index'])
    # df.to_json(save_cluster_indices_path, orient='records', lines=True)


    centroids = kmeans.cluster_centers_
    print('The centorids are:', centroids)
    
    if save_centroid_path is not None:
        print(f'Saving Centroids to :{save_centroid_path}')
        np.save(save_centroid_path, centroids)

    # return cluster_indices


class AnalyzeDataset():
    def __init__(
        self, 
        threshold_list, 
        centroid_file, 
        embedding_file, 
        shape, 
        index_file_path,
        stats_file_path,
        outlier_file_path,
        analysis_file_path
    ):
        self.threshold_list = threshold_list
        self.centroids = self.load_centroids(centroid_file)
        self.index_file_path =index_file_path
        self.emb = np.memmap(
            embedding_file,
            dtype=np.float32,
            mode='r',
            shape=shape
        )

        scaler = StandardScaler()
        self.emb = scaler.fit_transform(self.emb)

        self.analysis_file_path = analysis_file_path
        self.outlier_file_path = outlier_file_path
        self.stats_file_path = stats_file_path

        self.cluster_idx_df = pd.DataFrame(
            # np.zeros((shape[0], len(self.threshold_list)), dtype=np.int),
            # columns = ['Threshold: '+str(key) for key in threshold_list]
        )

        self.analysis_table = None
        self._analyze()
    

    def _analyze(self):

        for t in self.threshold_list:
            print('\n\nAnalyzing at threshold: ', t, '\n')
            analysis_path = self.analysis_file_path+f'_{t}.jsonl'
            stats_path = self.stats_file_path+f'_{t}.jsonl'
            outlier_path = self.outlier_file_path+f'_{t}.jsonl'

            self.generate_table(t)
            self.save_as_jsonl(analysis_path)

            self.generate_analysis( NUM_CLUSTERS, stats_path, outlier_path)
        

        self.cluster_idx_df.to_json(self.index_file_path, orient='records', lines=True)

    
    def generate_table(self , threshold):
        
        similarity = cosine_similarity(self.emb, self.centroids)
        
        analysis = []

        for idx, similiarity_row in enumerate(tqdm(similarity, desc='Assigning Cluster Ids')):
            # print(similiarity_row.shape)
            cluster_idx = np.argmax(similiarity_row)
            cluster_similarity = similiarity_row[cluster_idx]

            

            if cluster_similarity < threshold:
                self.cluster_idx_df.loc[idx, str(threshold)] = -1
                
                analysis.append(
                    {
                        'cluster_idx': -1,
                        'cluster_similarity': float(cluster_similarity), 
                        'closest_idx': int(cluster_idx)
                    }
                )
            else:

                self.cluster_idx_df.loc[idx, str(threshold)] = cluster_idx

                analysis.append(
                    {
                        'cluster_idx': int(cluster_idx), 
                        'cluster_similarity': float(cluster_similarity), 
                        'closest_idx': int(cluster_idx)
                    }
                )
        

        self.analysis_table = analysis


    def generate_analysis(self,num_clusters, stats_path, outlier_path):
        outlier = []
        cluster_total_counts = {key:0 for key in range(num_clusters)}
        cluster_total_cosine_sim_value = {key:0.0 for key in range(num_clusters)}

        for idx, data_sample in enumerate(tqdm(self.analysis_table, desc='Generating Analysis')):
            if data_sample['cluster_idx'] == -1:
                outlier.append(idx)
            else:
                cluster_total_counts[data_sample['cluster_idx']] += 1
                cluster_total_cosine_sim_value[data_sample['cluster_idx']] += data_sample['cluster_similarity']
        
        cluster_mean_sim_values = {key: cluster_total_cosine_sim_value[key]/(cluster_total_counts[key]+1) for key in range(num_clusters)}

        data = {
            "Cluster Index": list(cluster_total_counts.keys()),
            "Total Counts": list(cluster_total_counts.values()),
            "Mean Similarity": list(cluster_mean_sim_values.values())
        }

        outlier_row = pd.DataFrame(
            {
                'Cluster Index': [int(-1)],
                'Total Counts': [len(outlier)],
                'Mean Similarity': [None]
            }
        )
        
        df = pd.DataFrame(data)
        df = pd.concat([df, outlier_row], ignore_index=True)
        df.to_json(stats_path)
        # Print the table
        # print(df.to_string(index=False))
        # save the data frame as cluster_stats

        with open(outlier_path, "w") as file:
            json.dump(outlier, file)


        
    def save_as_jsonl(self, file_path):
        with open(file_path, "w") as jsonl_file:
            for record in self.analysis_table:
                jsonl_file.write(json.dumps(record) + "\n")

        print(f"Data has been saved to {file_path}")


    def load_centroids(self, centroid_file):

        np_centroid = np.load(centroid_file)
        return np_centroid


def plot_graph(data_matrix, threshold, datasets, plot_save_path):

    data_matrix = np.array(list(data_matrix.values()))
    # print(data_matrix)
    x = np.arange(len(datasets))

    fig = plt.figure(figsize=(30,12))

    for cluster_idx in range(NUM_CLUSTERS+1):
        plt.subplot(4, 5, cluster_idx+1)

        for dataset_idx, dataset_name in enumerate(datasets):
            bar = plt.bar(
                x[dataset_idx] + cluster_idx * 0.2,
                data_matrix[dataset_idx, cluster_idx],
                width=0.2,
                color=color_dict[dataset_idx],
                label=dataset_name if cluster_idx == 0 else None,
            )

            for rect in bar:
                height = rect.get_height()
                plt.text(
                    rect.get_x() + rect.get_width() /2,
                    height + 0.005,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='black',
                )


        if cluster_idx == NUM_CLUSTERS:
            # plt.title(f'Outlier percentage at Threshold: {threshold}')
            plt.title(f'Outlier')
        else:
            # plt.title(f'Cluster {cluster_idx} percentage at Threshold: {threshold}')
            plt.title(f'{cluster_idx}')
        plt.xlabel('Datasets')
        plt.ylabel('Percentage')
        # plt.xticks(x + 0.4, datasets)
        plt.xticks([])

        fig.suptitle(f'Threshold: {threshold}')

        if cluster_idx == 0:
            plt.legend()
        
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()


def plot_tsne(embedding_files, indices_file_paths, shape_list, title_list, plot_save_path_list):

    for file_idx in range(len(embedding_files)):
        np_memmap = np.memmap(
            embedding_files[file_idx],
            dtype=np.float32,
            mode='r',
            shape=shape_list[file_idx]
        )

        scaler = StandardScaler()
        np_memmap = scaler.fit_transform(np_memmap)


        # with open(indices_file_paths[file_idx], 'r') as 
        indices_df = pd.read_json(indices_file_paths[file_idx],lines=True)
        # print(indices_df.columns)
        # for threshold in indices_df.columns:
        tsne = TSNE()
        tsne_emb =tsne.fit_transform(np_memmap)

        color_list = [color_dict[indices_df.iloc[i, 2]] for i in range(len(indices_df))]

        # plot_title = title_list[file_idx]+f'_{threshold}'
        plot_title = title_list[file_idx]+f'_0.7'
        # plot_save_path = 'tsne_plots_humaneval/'+plot_save_path_list[file_idx]+ f'_{threshold}.png'
        plot_save_path = plot_save_path_list[file_idx]+ f'_0.7.png'

        plt.figure(figsize=(8,8))
        plt.scatter(tsne_emb[:,0], tsne_emb[:,1], c=color_list, edgecolor='k')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[i], markersize=10, label=f'Cluster_{i}') for i in range(NUM_CLUSTERS)
        ]

        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[-1], markersize=10, label=f'Outliers')
        )

        plt.legend(handles=legend_elements, loc='upper left')

        plt.title(f'Tsne plot of {plot_title} at Threshold: 0.7')
        plt.savefig(plot_save_path)
        plt.show()

        print('Done: ', f'Tsne plot of {plot_title}')

    # for emb_file in embedding_files:
    #     for index_file in indices_file_paths:


def main():
    ## run script using following
    ## python complete_script.py --benchmark humaneval --num-entries 164 --num-cluster 19
    parser = argparse.ArgumentParser(description='Analyze the datasets')
    parser.add_argument('--benchmark', type=str, default='mbpp', help='The benchmark dataset')

    parser.add_argument('--num-entries', type=int, help='number of entries in benchmark dataset')

    parser.add_argument('--num-clusters', type=int, help='How many clusters you want for this benchmark')

    args = parser.parse_args()

    global NUM_CLUSTERS
    NUM_CLUSTERS = args.num_clusters

    benchmark = args.benchmark
    num_entries = args.num_entries



    # Change here
    shape_list = [(num_entries, 768), (15,266,581, 768)]
    # Change here
    datasets = []
    datasets.append(benchmark)
    datasets.append('v12p8')
    # datasets = ['mbpp', 'complete']
    base_path = '/mlf-transfers-only/srinjoym/harsh_req'
    # benchmark = 'mbpp'

    thresholds = [0.77, 0.75, 0.7, 0.6, 0.5]

    embedding_files = [ f'{base_path}/embeddings/{dataset}_emb_memory.npy' for dataset in datasets]

    centroid_files = [ f'{base_path}/centroids_{benchmark}/{dataset}_centroids.npy' for dataset in datasets ]

    cluster_indices_paths = [f'{base_path}/cluster_assigned_indices_{benchmark}/{dataset}.jsonl' for dataset in datasets]

    outliers_paths = [f'{base_path}/outliers_{benchmark}/{dataset}_outliers' for dataset in datasets]

    stats_file_paths = [f'{base_path}/stats_{benchmark}/{dataset}_stats' for dataset in datasets]
    
    analysis_file_paths = [f'{base_path}/analysis_{benchmark}/{dataset}_analysis' for dataset in datasets]

    # print(analysis_file_paths)
    # print(cluster_indices_paths)
    # print(stats_file_paths)
    # print(outliers_paths)


    ## create the directories
    os.makedirs(f'{base_path}/analysis_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/centroids_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/cluster_assigned_indices_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/outliers_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/stats_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/plots_{benchmark}', exist_ok=True)
    os.makedirs(f'{base_path}/tsne_plots_{benchmark}', exist_ok=True)


    for i in range(len(datasets)):
        file_path = embedding_files[i]
        data = np.memmap(file_path,dtype=np.float32, mode='r', shape=shape_list[i])

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # centroid_
        generate_clusters(data, centroid_files[i])
        break
    
    print('>>>>>>>>>>Centroids are Generated')


    for i in range(len(datasets)):
        
        ## Change here
        centroid_file = f'{base_path}/centroids_{benchmark}/{datasets[0]}_centroids.npy'
        embedding_file = embedding_files[i]
        cluster_idx_file_path = cluster_indices_paths[i]
        stats_file = stats_file_paths[i]
        outlier_file = outliers_paths[i]
        analysis_file = analysis_file_paths[i]
        shape = shape_list[i]

        print('\n\nAnalyzing dataset:', datasets[i])
        AnalyzeDataset(thresholds, centroid_file, embedding_file, shape, cluster_idx_file_path, stats_file, outlier_file, analysis_file)

    print('>>>>>>>>The datasets are analyzed')


    for threshold in thresholds:

        req_data = {}
        for idx, dataset_name in enumerate(datasets):
            data_dict = read_stat_file(stats_file_paths[idx] + f'_{threshold}.jsonl')
            req_data[dataset_name] = extract_cluster_size(data_dict, shape_list[idx][0])

        plot_save_path_list = f'{base_path}/plots_{benchmark}/analysis_plot_{threshold}.png'

        plot_graph(req_data, threshold, datasets, plot_save_path_list)

    print('>>>>>>>>The Comparative PLots are generated')


    # def plot_tsne(embedding_files, indices_file_paths, shape_list, title_list, plot_save_path_list):

    # title_list = ['Human Eval', 'Magpie Filtered', 'Magpie DeepSeek', 'Magpie Qwen']

    # plot_save_path_list = ['mbpp_tsne', 'magpie_filtered_tsne', 'magpie_deepseek_tsne', 'magpie_qwen_tsne']
    
    # plot_tsne(embedding_files, cluster_indices_paths, shape_list, title_list, plot_save_path_list)

if __name__ == '__main__':
    main()
