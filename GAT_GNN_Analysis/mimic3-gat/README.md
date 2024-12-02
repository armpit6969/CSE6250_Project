# GAT-GNN Comparative Analysis

This directory contains the comparative analysis between the **Graph Attention Network (GAT)** and the CHEB/CGCN from "Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data" adaptation of MIMIC-III Data Pre-Processing Pipeline for GNN. Here, using a standardized graph dataset produced from the MIMIC-III pipeline, we modified and run some scripts from the original work to compare the performance of the model.

To run the project

Option 1:
1. Please download the zipped project directory, including the preprocessed data files and graph structures, from: ***
2. Extract the project directory
3. Follow the instructions in "Details" below

Option 2:
1. Please pull the project directory from the Github repository:

<details>
  
###### Benchmark/Dataset Creation 
<pre><code>

# Create a conda environment using the provided environment.yaml
conda env create -f environment.yaml
conda activate mimic3_graph
pip install -r requirements.txt # (Optional, however libraries are outdated compared to the environment.yaml)

# Get the MIMIC-III dataset by script or copy the files here:
wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/ # this will create a physionet folder with the Database csvs

# Run the MIMIC-III PreProcessing scripts from mimic3-benchmarks:
python -m mimic3benchmark.scripts.extract_subjects physionet.org/files/mimiciii-demo/1.4/ data/root
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/

# Run the scripts from the mimic3-benchmarks (the 'phenotyping' script is used for the purpose of this evaluation):
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

# Split the training set from "data/phenotyping" where the processed dataset phenotypes are used for this evaluation:
python -m mimic3models.split_train_val data/phenotyping

# Run the below command to create the STAT embeddings using the previous Logistic Regression model:
python -um mimic3models.phenotyping.logistic.main --output_dir mimic3models/phenotyping/logistic

# Run the below command to evaluate the STAT embeddings from the previous Logistic Regression model:
python -m mimic3benchmark.evaluation.evaluate_phenotyping data/phenotyping/train data/phenotyping/train predictions/phenotyping/logistic train

# Run the below command to create the LSTM embeddings:
python -m mimic3models.train_lstm --network mimic3models/lstm.py --data data/phenotyping/ --save

# Create the graph edges for the respective edge strategy rules (in this case, we use expert_medium because expert_lenient is too large and requires 150-300GB+ RAM)
python -m gnn__models.connectivity_strategies.expert_graph_m2_inter_category # Creates expert_medium edge strategy graph connections

# Create the homogenous graphs using the STAT/LSTM embeddings (expert_lenient):
python create_graph.py --edge_strategy trivial --node_embeddings_type stat --folder_name graphs
python create_graph.py --edge_strategy expert_medium --node_embeddings_type stat --folder_name graphs

python create_graph.py --edge_strategy trivial --node_embeddings_type lstm --folder_name graphs
python create_graph.py --edge_strategy expert_medium --node_embeddings_type lstm --folder_name graphs


# Training
To execute the training of each model you can either run the below commands individually or run the provided bash script: `./run_script.sh`

# Run the below to train the GAT models
python train_gnn.py --model GATConv --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_transductive_stat_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name GATConv_transductive_trivial_stat
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_transductive_stat_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name GATConv_transductive_expert_medium_stat 

python train_gnn.py --model GATConv --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_transductive_lstm_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name GATConv_transductive_trivial_lstm 
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_transductive_lstm_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name GATConv_transductive_expert_medium_lstm 

python train_gnn.py --model GATConv --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_inductive_stat_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name GATConv_inductive_trivial_stat
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_inductive_stat_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name GATConv_inductive_expert_medium_stat 

python train_gnn.py --model GATConv --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_inductive_lstm_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name GATConv_inductive_trivial_lstm 
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name GATConv_inductive_lstm_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name GATConv_inductive_expert_medium_lstm 


# Run the below to train the CGCN models
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_transductive_stat_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ClusterGCNConv_transductive_trivial_stat 
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_transductive_stat_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ClusterGCNConv_transductive_expert_medium_stat 

python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_transductive_lstm_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ClusterGCNConv_transductive_trivial_lstm
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_transductive_lstm_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ClusterGCNConv_transductive_expert_medium_lstm

python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_inductive_stat_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ClusterGCNConv_inductive_trivial_stat 
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_inductive_stat_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ClusterGCNConv_inductive_expert_medium_stat 

python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_inductive_lstm_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ClusterGCNConv_inductive_trivial_lstm
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ClusterGCNConv_inductive_lstm_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ClusterGCNConv_inductive_expert_medium_lstm


# Run the below to train the CHEB (ChebConv) models
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_transductive_stat_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ChebConv_transductive_symK1_trivial_stat 
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_transductive_stat_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ChebConv_transductive_symK1_expert_medium_stat 

python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_transductive_lstm_es_trivial_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ChebConv_transductive_symK1_trivial_lstm 
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_transductive_lstm_es_expert_medium_2024_11_30 --training_mode transductive --model_folder graph_model --experiment_name ChebConv_transductive_symK1_expert_medium_lstm 

python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_inductive_stat_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ChebConv_inductive_symK1_trivial_stat 
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_stat/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_inductive_stat_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ChebConv_inductive_symK1_expert_medium_stat 

python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_inductive_lstm_es_trivial_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ChebConv_inductive_symK1_trivial_lstm 
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_lstm/processed/ --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 --model_name ChebConv_symK1_inductive_lstm_es_expert_medium_2024_11_30 --training_mode inductive --model_folder graph_model --experiment_name ChebConv_inductive_symK1_expert_medium_lstm 


</code>
</pre>


The above code was adapted from "Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data" adaptation of MIMIC-III Data Pre-Processing Pipeline for GNN. The code was refactored and extraneous code relevant to the original paper's benchmark of GNN was removed. Furthermore, some minor modifications to the Data Pre-Processing Pipeline were added to get it running (potentially due to later revisions in MIMIC-III ie. v1.4). Furthermore, this enabled a fair comparison between GNN and GAT on a level playing field, based on the same replicated EHR MIMIC-III data. 

### Source code for the implementation of the paper: 
"Leveraging patient similarities via graph neural networks to predict phenotypes from temporal data" 

1. [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/10302556)  
2. [GitHub repository for original MIMIC-III Data Pre-Processing Pipeline](https://github.com/YerevaNN/mimic3-benchmarks)
3. [GitHub repository for adapted MIMIC-III Data Pre-Processing Pipeline adapted for GNN](https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23)
