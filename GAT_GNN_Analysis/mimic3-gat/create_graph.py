from torch.utils.data import DataLoader, random_split, Dataset, Subset
import argparse
import glob
import itertools
import os
import pickle
import random
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from edge_strategies import KNNGraph
from read_embeddings import get_features 

DATE_TIME=datetime.now().strftime("%d%m%Y-%H%M%S")

# Setting the Homogenous Graphs from the MIMIC3 Benchmark Data Processing Pipeline (Multitask learning and benchmarking with clinical time series data)
# Source: https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23/edge_strategies.py
#- Omit the parallel pickle file reading
#- Omit the get_most_recent_file()
#- Omit the feature_anomaly_edges_automatic() from the pre-processing pipeline analysis of how nodes should be connected


def read_pkl(pickle_file):
    print(pickle_file)
    with open(pickle_file, "rb") as f:
        fcontent = pickle.load(f)
        f.close()
        return fcontent

def read_all_pkl(pkl_files):
    print("Reading pickle files")
    res = []
    for i in range(len(pkl_files)): res.append(read_pkl(pkl_files[i]))
    return res

def shortened_path(data, train_df, val_df, test_df):

    # Record the start time for execution tracking
    counter_time_execution = datetime.now()

    # Combine all indices from train, validation, and test DataFrames into a single list
    all_data = train_df.index.tolist() + val_df.index.tolist() + test_df.index.tolist()

    # Create a mapping from index to name
    mapping_index_name = pd.Series(all_data).to_dict()

    # Reverse mapping from name to index for quick lookup
    mapping_name_index = {v: k for k, v in mapping_index_name.items()}

    # Print the time taken to load and map data
    print("Loaded! Time taken to load data:", datetime.now() - counter_time_execution)

    # Check if 'data' is a pandas Series, otherwise use it directly
    if isinstance(data, pd.Series):
        # Flatten the list of lists into a single list
        data1 = [item for sublist in data for item in sublist]
        print(f"Data: list of groups detected! Time execution: {datetime.now() - counter_time_execution}")
        
        # Remove duplicate entries
        data2 = list(set(data1))
    else: data2 = data

    # Print the length of the processed data
    print("len(data2):", len(data2))
    print(f"Regex findall! Time execution: {datetime.now() - counter_time_execution}")

    # # Use regex to extract filenames matching the pattern '/xxx_timeseries.csv'
    # extracted_strings = re.findall(r"\/([^\/]*_timeseries\.csv)", str(data2))

    # Process in chunks to avoid large memory usage
    extracted_strings = []
    chunk_size = 1000000
    data_str = str(data2)
    
    for i in range(0, len(data_str), chunk_size):
        chunk = data_str[i:i+chunk_size]
        extracted_strings.extend(re.findall(r"\/([^\/]*_timeseries\.csv)", chunk))

    # Pair the extracted filenames into tuples representing edges
    shorted_node_names = [tuple(extracted_strings[i : i + 2]) for i in range(0, len(extracted_strings), 2)]

    # Ensure that the first element of the first tuple is a string
    assert isinstance(shorted_node_names[0][0], str), "First element should be a string"

    # Create edges by mapping node names to their corresponding indices
    edges = [(mapping_name_index[v[0]], mapping_name_index[v[1]]) for v in tqdm(shorted_node_names)]

    # Validate that 'edges' is a list
    assert isinstance(edges, list), "edges should be a list"

    # Validate that the first element of the first edge is an integer index
    assert isinstance(edges[0][0], int), "Edge source should be an integer index"

    return edges



class MyOwnDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        node_type_embeddings="lstm",
        edge_strategy_name="expert_exact",
        k=10,
        n_edges=None,
        distance_euclidean=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        device="cpu",
    ):
        self.edge_strategy_name = edge_strategy_name
        self.k = k
        self.n_edges = n_edges
        self.distance_euclidean = distance_euclidean
        self.node_type_embeddings = node_type_embeddings
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device(device))

    def raw_file_names(self):
        return ["some_file_1", "some_file_2"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):

        # Loading features
        print("Loading Features...")
        train_df, val_df, test_df = get_features()
        self.node_embeddings = (train_df, val_df, test_df)
        assert (len(train_df.index.tolist() + val_df.index.tolist() + test_df.index.tolist()) > 0), "There should be more than 0 entries"

        print("Reading Node Embeddings...")
        train_df, val_df, test_df = self.node_embeddings

        # Select node embeddings based on the specified type
        if self.node_type_embeddings == "lstm":
            print("Using LSTM Embeddings...")
            train_X = train_df["lstm_embedding"]
            val_X = val_df["lstm_embedding"]
            test_X = test_df["lstm_embedding"]
            train_y = train_df["ys"]
            val_y = val_df["ys"]
            test_y = test_df["ys"]

        if self.node_type_embeddings == "stat":
            print("Using STAT Embeddings...")
            train_X = train_df["stat_features"]
            val_X = val_df["stat_features"]
            test_X = test_df["stat_features"]
            train_y = train_df["ys"]
            val_y = val_df["ys"]
            test_y = test_df["ys"]

        # print(train_df["stat_features"].head(10))
        # print(train_df["lstm_embedding"].head(10))
        # print(train_df.head(10))

        # Concatenate all node features
        X_all = torch.cat((torch.Tensor(train_X), torch.Tensor(val_X), torch.Tensor(test_X),), 0,)

        print("Creating edges")
        # Create edges based on the specified strategy
        assert self.edge_strategy_name in ["expert_medium", "trivial"], "Invalid edge strategy"
        if "expert" in self.edge_strategy_name:
            
            assert self.edge_strategy_name in ["expert_medium"], "Unsupported expert strategy"
            
            PATH = "."
            EDGE_FILES = {"expert_medium": "A_m2_expert_edges_inter_category.pk"}
            
            data = pickle.load(open(os.path.join(PATH, EDGE_FILES[self.edge_strategy_name]), "rb"))
            tensor_edges = torch.tensor(shortened_path(data, train_df, val_df, test_df))
            tensor_edges = torch.swapaxes(tensor_edges, 1, 0)
            edge_index_mod_self, edge_attr = add_self_loops(tensor_edges)
            graph = Data(x=X_all, edge_index=edge_index_mod_self)

        if self.edge_strategy_name == "trivial":
            
            edge_index = torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=X_all, edge_index=edge_index)
            num_nodes = X_all.size(0)
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            graph.edge_index = edge_index

        print("Creating labels...")
        all_Y = torch.cat([torch.tensor(train_y), torch.tensor(val_y), torch.tensor(test_y)])
        train_mask = torch.cat([torch.ones(len(train_X)), torch.zeros(len(val_X)), torch.zeros(len(test_X))], 0)
        val_mask = torch.cat([torch.zeros(len(train_X)), torch.ones(len(val_X)), torch.zeros(len(test_X))], 0)
        test_mask = torch.cat([torch.zeros(len(train_X)), torch.zeros(len(val_X)), torch.ones(len(test_X))], 0)

        graph.y = all_Y
        graph.train_mask = train_mask.bool()
        graph.val_mask = val_mask.bool()
        graph.test_mask = test_mask.bool()
        data_list = [graph]

        # Apply pre-filter and pre-transform if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Collate and save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Processed data to:", self.processed_paths[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_strategy_name", type=str, required=True)
    parser.add_argument("--node_embeddings_type", type=str, required=True)  # e.g., 'lstm_pca_6', 'lstm_pca50'
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_edges", type=int, default=300_000)
    parser.add_argument("--folder_name", type=str, required=True)

    # Add boolean argument to the parser and store False if specified
    parser.add_argument("--distance_euclidean", action="store_false")

    args = parser.parse_args()
    print("Arguments:", args)

    edge_strategy_name = args.edge_strategy_name
    assert edge_strategy_name in ["expert_medium", "trivial"], "Invalid edge strategy name"

    k = args.k
    distance_euclidean_else_cosine = args.distance_euclidean
    name_distance = "cosine" if distance_euclidean_else_cosine else "euclidean"

    node_embeddings_type = args.node_embeddings_type
    folder_name = args.folder_name
    assert folder_name is not None and folder_name != "", "folder_name should be specified"
    folder_name = f"{folder_name}/data_{edge_strategy_name}_{node_embeddings_type}"

    print("SAVING folder_name:", folder_name)

    # Initialize the custom dataset
    dataset = MyOwnDataset(
        root=folder_name,
        node_type_embeddings=node_embeddings_type,
        edge_strategy_name=edge_strategy_name,
        k=k,  # For knn_graph only
        n_edges=args.n_edges,  # For random only
    )
    data = dataset[0]
    print("Created file:")

    # Define folder name for reporting
    DATE_TIME = datetime.now().strftime("%d%m%Y-%H%M%S")
    FOLDER_NAME = f"data_{edge_strategy_name}_k{str(k)}_{name_distance}_{node_embeddings_type.replace('.', '_')}"

    def append_to_file(file_name, text_to_append):
        with open(file_name, "a+") as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("")
            file_object.write(text_to_append)

    # Generate report name and dataset statistics
    report_name = "data_creation_" + DATE_TIME + "_" + FOLDER_NAME
    NUMBER_NODES = data.x.shape[0]
    NUMBER_EDGES = data.edge_index.shape[1]

    # Append the report to 'graph_creation.txt'
    append_to_file("graph_creation.txt", f"{report_name}: Number of nodes: {NUMBER_NODES} \t Number of edges: {NUMBER_EDGES}\n")