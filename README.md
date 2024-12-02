# Base Directory README  

## Overview  
This repository contains two subprojects focused on exploring and evaluating graph-based machine learning techniques applied to Electronic Health Records (EHR) data and citation network datasets. The first subproject conducts a comparative analysis of Graph Neural Networks (GNNs), while the second implements Graph Attention Networks (GATs) for node classification tasks.  

---

## Subproject 1: **GAT-GNN Comparative Analysis**  
This subproject extends the work from "Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data" by adapting and benchmarking GNN architectures (ChebConv, Cluster-GCN, and GATConv) on a standardized MIMIC-III dataset.  

### Key Features  
- Implements multiple GNN architectures on graphs derived from MIMIC-III EHR data.  
- Compares weighted AUC scores across transductive and inductive tasks using Statistical (STAT) and LSTM embeddings.  
- Includes preprocessing scripts for data extraction, phenotype embedding creation, and graph construction using various edge strategies (Trivial and Expert-Medium).  

### Getting Started  
1. Download the preprocessed datasets or use the provided preprocessing pipeline to generate datasets.  
2. Train models with provided scripts for GAT, Cluster-GCN, and ChebConv.  
3. Evaluate model performance using the integrated training and evaluation pipelines.  

For detailed instructions, refer to the `README` in the `GAT_GNN_Analysis` directory.  

---

## Subproject 2: **Graph Attention Networks (GAT) for Node Classification**  
This subproject replicates and evaluates the GAT architecture introduced in the paper ["Graph Attention Networks"](https://arxiv.org/abs/1710.10903) by Veličković et al. It focuses on node classification tasks for citation networks such as **Cora**, **Citeseer**, and **PubMed**.  

### Key Features  
- Full implementation of GAT with multi-head attention and LeakyReLU activation for neighbor weighting.  
- Automated dataset downloading, preprocessing, and training integrated into a single pipeline.  
- Performance visualization with generated accuracy and loss plots.  

### Getting Started  
1. Clone the repository and install dependencies.  
2. Train the GAT model on Cora, Citeseer, or PubMed datasets using the `gat_transductive.py` script.  
3. Visualize training results and evaluate performance with saved plots.  

For detailed instructions, refer to the `README` in the `GAT_Implementation` directory.  

---

## Repository Structure  
```plaintext
Base Directory
├── GAT_GNN_Analysis/        # Subproject 1: Comparative analysis of GAT and GNNs
├── GAT_Implementation/      # Subproject 2: Graph Attention Networks (GAT)
└── README.md                # Base directory overview
```

---

## Requirements  
- Python 3.8+  
- Conda environment files and `requirements.txt` are included in both subproject directories for easy setup.  

---

## References  
1. Harutyunyan, H. (2018). Multitask learning and benchmarking with clinical time series data.  
2. Proios, D., Yazdani, A., et al. (2023). Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data.  
3. Veličković, P., et al. (2018). Graph Attention Networks.  
