from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph, RadiusGraph
import torch_geometric 
import itertools as iter
import os
import pdb
import pickle
import random
import string
import numpy as np
import pandas as pd
import torch

# Setting the Edge Selection Strategy from the MIMIC3 Benchmark Data Processing Pipeline (Multitask learning and benchmarking with clinical time series data)
# Source: https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23/edge_strategies.py
#- Omit the check for unique edges in the edge strategies
#- Omit the edge strategies that are not used in the code
#- Omit the feature_anomaly_edges_automatic() from the pre-processing pipeline analysis of how nodes should be connected (feature anomalies derived from patient data)


make_random_edges = lambda n_nodes, n_edges: random.choices(list(iter.combinations(range(n_nodes), 2)), k=n_edges)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_csv_as_string(filename):
    with open(filename, "r") as f:
        return f.read()

def regex_replace_with_dict(s: str, d):
    for k, v in d.items():
        s = s.replace(k, v)
    return s

def save_lines_as_csv(lines, filename):
    # pdb.set_trace()
    with open(filename, "w") as f:
        f.write("\n".join(lines))

def save_string_as_csv(s, filename):
    with open(filename, "w") as f:
        f.write(s)

def inverse_dict(d):
    return {v: k for k, v in d.items()}

def find_duplicate_tuples_in_list(l):
    return [x for n, x in enumerate(l) if x in l[:n]]


class KNNGraph:
    def __init__(self, node_features, loop=True, distance_cosine=True, k=100, num_workers=5):
        """Make edges between k nearest neighbours in node_features"""
        self.graph = KNNGraph(k=k, loop=True, force_undirected=False)

    def __call__(self, node_features):
        global device

        new_data = Data(pos=torch.tensor(node_features))
        new_data.to(device=device)

        result_graph = self.graph(new_data)

        return result_graph


