#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting training of GATConv models..."

# GATConv Transductive Models
echo "Training GATConv_transductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model GATConv --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_transductive_stat_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name GATConv_transductive_trivial_stat

echo "Training GATConv_transductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name GATConv_transductive_expert_medium_stat 

echo "Training GATConv_transductive_stat_es_trivial_2024_11_30 (LSTM)..."
python train_gnn.py --model GATConv --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_transductive_stat_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name GATConv_transductive_trivial_lstm 

echo "Training GATConv_transductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name GATConv_transductive_expert_medium_lstm 

# GATConv Inductive Models
echo "Training GATConv_inductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model GATConv --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_inductive_stat_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name GATConv_inductive_trivial_stat

echo "Training GATConv_inductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name GATConv_inductive_expert_medium_stat 

echo "Training GATConv_inductive_stat_es_trivial_2024_11_30 (LSTM)..."
python train_gnn.py --model GATConv --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_inductive_stat_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name GATConv_inductive_trivial_lstm 

echo "Training GATConv_inductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model GATConv --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name GATConv_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name GATConv_inductive_expert_medium_lstm 

echo "GATConv model training completed."
echo "----------------------------------------"

echo "Starting training of ClusterGCNConv models..."

# ClusterGCNConv Transductive Models
echo "Training ClusterGCNConv_transductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_transductive_stat_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_transductive_trivial_stat 

echo "Training ClusterGCNConv_transductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_transductive_expert_medium_stat 

echo "Training ClusterGCNConv_transductive_lstm_es_trivial_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_transductive_lstm_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_transductive_trivial_lstm 

echo "Training ClusterGCNConv_transductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_transductive_expert_medium_lstm 

# ClusterGCNConv Inductive Models
echo "Training ClusterGCNConv_inductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_inductive_stat_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_inductive_trivial_stat 

echo "Training ClusterGCNConv_inductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_inductive_expert_medium_stat 

echo "Training ClusterGCNConv_inductive_lstm_es_trivial_2024_11_30..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_inductive_lstm_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_inductive_trivial_lstm 

echo "Training ClusterGCNConv_inductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model ClusterGCNConv --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ClusterGCNConv_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ClusterGCNConv_inductive_expert_medium_lstm 

echo "ClusterGCNConv model training completed."
echo "----------------------------------------"

echo "Starting training of ChebConv_symK1 models..."

# ChebConv_symK1 Transductive Models
echo "Training ChebConv_symK1_transductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_transductive_stat_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ChebConv_transductive_symK1_trivial_stat 

echo "Training ChebConv_symK1_transductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ChebConv_transductive_symK1_expert_medium_stat 

echo "Training ChebConv_symK1_transductive_stat_es_trivial_2024_11_30 (LSTM)..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_transductive_stat_es_trivial_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ChebConv_transductive_symK1_trivial_lstm 

echo "Training ChebConv_symK1_transductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_transductive_stat_es_expert_medium_2024_11_30 \
  --training_mode transductive --model_folder graph_model \
  --experiment_name ChebConv_transductive_symK1_expert_medium_lstm 

# ChebConv_symK1 Inductive Models
echo "Training ChebConv_symK1_inductive_stat_es_trivial_2024_11_30..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_inductive_stat_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ChebConv_inductive_symK1_trivial_stat 

echo "Training ChebConv_symK1_inductive_stat_es_expert_medium_2024_11_30..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_stat/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ChebConv_inductive_symK1_expert_medium_stat 

echo "Training ChebConv_symK1_inductive_stat_es_trivial_2024_11_30 (LSTM)..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_trivial_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_inductive_stat_es_trivial_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ChebConv_inductive_symK1_trivial_lstm 

echo "Training ChebConv_symK1_inductive_stat_es_expert_medium_2024_11_30 (LSTM)..."
python train_gnn.py --model ChebConv_symK1 --data_folder graphs/data_expert_medium_lstm/processed/ \
  --epochs 10 --WD 0.001 --lr 0.0001 --hidden 128 --batch_size 512 \
  --model_name ChebConv_symK1_inductive_stat_es_expert_medium_2024_11_30 \
  --training_mode inductive --model_folder graph_model \
  --experiment_name ChebConv_inductive_symK1_expert_medium_lstm 

echo "ChebConv_symK1 model training completed."
echo "----------------------------------------"

echo "All training commands executed successfully!"