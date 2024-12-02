from tqdm import tqdm 
import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
from torch_geometric.data import DataLoader, Data, Dataset

from create_graph import MyOwnDataset
from torch_geometric.utils import subgraph
from torch_geometric.nn import ChebConv, GATConv, ClusterGCNConv
import matplotlib.pyplot as plt


# Setting the Edge Selection Strategy from the MIMIC3 Benchmark Data Processing Pipeline (Multitask learning and benchmarking with clinical time series data)
# Source: https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23/edge_strategies.py
#- Omit the check for unique edges in the edge strategies
#- Omit the edge strategies that are not used in the code
#- Omit the feature_anomaly_edges_automatic() from the pre-processing pipeline analysis of how nodes should be connected


activation_fns = {
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
    # 'relu': nn.ReLU,
    # 'leaky_relu': nn.LeakyReLU,
    # 'sigmoid': nn.Sigmoid,
    # 'tanh': nn.Tanh,
    # 'elu': nn.ELU,
    # 'softmax': nn.Softmax,
    # 'log_softmax': nn.LogSoftmax,
    # 'prelu': nn.PReLU,
    # 'softplus': nn.Softplus,
    # 'softsign': nn.Softsign,
    # 'tanhshrink': nn.Tanhshrink,
    # 'softmin': nn.Softmin,
    # 'hardtanh': nn.Hardtanh,
    # 'hardshrink': nn.Hardshrink,
    # 'hardswish': nn.Hardswish,
    # 'softshrink': nn.Softshrink,
    # 'threshold': nn.Threshold,
}

graph_layers = {
    'ChebConv_symK1':{
        'impl': ChebConv,
        'params': {'default_convs': True, 'K': 2, 'normalization': 'sym', 'heads': 1}
    },
    'ClusterGCNConv':{
        'impl': ClusterGCNConv,
        'params': { "diag_lambda": 0.0, 'heads': 1 }
    },
    'GATConv':{
        'impl': GATConv,
        'params': {'heads': 4}
    }
}

# Initialize a list to store metrics for each epoch
all_metrics = []

class myGCN(torch.nn.Module):
    def __init__(self,hidden_channels, dataset, layer_type, NUM_HIDDEN_LAYERS=1, NUM_MLP_LAYERS=1, POST_NUM_MLP_LAYERS=1, aggr_fn="mean", dropout_rate=0.0, activation_fn_name="gelu", model_name = "", **kwargs,):
        
        super().__init__()

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SHEET_NUMBER)

        # Store configuration parameters
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.NUM_MLP_LAYERS = NUM_MLP_LAYERS
        self.POST_NUM_MLP_LAYERS = POST_NUM_MLP_LAYERS
        self.dropout_rate = dropout_rate
        self.model_name = model_name

        # Initialize activation function
        self.activation_fn = activation_fns[activation_fn_name]()

        # Retrieve layer implementation and parameters based on layer_type
        layer = graph_layers[layer_type]["impl"]
        layer_params = graph_layers[layer_type]["params"]
        layer_params['aggr'] = aggr_fn

        # Optional Layer Normalization
        self.layer_norm = torch.nn.LayerNorm(hidden_channels)
        self.activation_fn = activation_fns[activation_fn_name]()
        self.model_name = model_name
        layers = OrderedDict()
        for i in range(NUM_HIDDEN_LAYERS):
            print("added graph hidden layer: ", i)
            layers[str(i)] = layer(hidden_channels, hidden_channels, **layer_params)
            for k in layers[str(i)].state_dict().keys():
                torch.nn.init.xavier_uniform_(
                    layers[str(i)].state_dict()[k].reshape(1, -1)
                ).reshape(-1)
        self.hidden_layers = Sequential(layers)

        mlp_layers = OrderedDict()
        for i in range(NUM_MLP_LAYERS):
            print("added mlp layer: ", i)
            if i == 0:
                mlp_layers[str(i)] = torch.nn.Linear(
                    dataset.num_features, hidden_channels
                )
            else:
                mlp_layers[str(i)] = torch.nn.Linear(hidden_channels, hidden_channels)
            torch.nn.init.xavier_uniform_(mlp_layers[str(i)].weight)
        self.mlp_layers = Sequential(mlp_layers)

        post_mlp_layers = OrderedDict()
        for i in range(POST_NUM_MLP_LAYERS):
            print("added post-mlp layer: ", i)
            if i == 0:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(dataset.num_features + hidden_channels * layer_params['heads']),
                    torch.nn.Linear(dataset.num_features + hidden_channels * layer_params['heads'], hidden_channels)
                )
            else:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(hidden_channels * layer_params['heads']),
                    torch.nn.Linear(hidden_channels * layer_params['heads'], hidden_channels)
                )
            torch.nn.init.xavier_uniform_(post_mlp_layers[str(i)][1].weight)
        self.post_mlp_layers = Sequential(post_mlp_layers)
        self.lin1 = torch.nn.Linear(hidden_channels,dataset.num_classes)
        torch.nn.init.xavier_uniform_(self.lin1.weight)

    def forward(self, raw, edge_index, device, edge_weight=None):

        # Move edge indices to the specified device
        edge_index = edge_index.to(device)

        # MLP Layers Before Graph Convolutions
        x = raw 
        for i in range(self.NUM_MLP_LAYERS):
            x = self.mlp_layers[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Hidden Graph Convolutional Layers
        node_embeddings = []
        for i in range(self.NUM_HIDDEN_LAYERS):  # 0
            if edge_weight is None:
                node_embeddings = self.hidden_layers[i](x, edge_index)
            else:
                node_embeddings = self.hidden_layers[i](x, edge_index, edge_weight)
            x = self.activation_fn(node_embeddings)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Concatenate raw features with final embeddings
        x = torch.cat((raw, x), dim=1)

        # Post MLP Layers After Graph Convolutions
        for i in range(self.POST_NUM_MLP_LAYERS):
            x = self.post_mlp_layers[i](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Final Linear Layer for Classification
        x = self.lin1(x)

        # Apply Sigmoid Activation
        return x.sigmoid(), node_embeddings

def train(model, data, train_params, model_dir, train_mode, model_path=None, **kwargs):

    # Set global access variables
    global latest_loss, max_patience_count
    
    model.to(device)
    
    batch_size = train_params["batch_size"]
    max_epochs = train_params["NUM_EPOCHS"]
    lr = train_params["LR"]
    wtd = train_params["WD"]
    num_nodes = train_params["NUM_NODES"]
    train_nodes = data.train_mask.nonzero(as_tuple=False).squeeze()
    DD_MM_YYYY_HH_MM_SS_epoch = datetime.now().strftime("%d%m%Y-%H%M%S")
    epoch = 0

    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wtd)

    # Initialize Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Initialize NeighborSampler
    train_loader = NeighborSampler(
        data.edge_index if train_mode == 'transductive' else subgraph(train_nodes, data.edge_index, relabel_nodes=True)[0],
        sizes=[num_nodes],
        batch_size=batch_size,
        shuffle=True
    )

    # Training Loop
    while True:

        total_loss = 0
        epoch += 1
        print(f"Current Epoch: {epoch}")

        # Train Mode
        model.train()
        for _, n_id, adjs in tqdm(train_loader):

            # Zero gradients
            optimizer.zero_grad() 

            # Forward Pass
            y_pred, _ = model(data.x[n_id], adjs[0], device)

            # Compute Train Loss
            train_loss = loss_fn(y_pred, data.y[n_id].float())

            # Cleanup
            del adjs
            torch.cuda.empty_cache()

            # Backward Pass
            train_loss.backward()

            # Update model params
            optimizer.step()

            # Sum Total Loss
            total_loss += train_loss.item()
        
        # Update learning rate based on latest training loss
        lr_scheduler.step(train_loss)

        # Calculate average epoch loss
        total_loss /= len(train_loader)
        writer.add_scalar('Loss/train', total_loss, epoch)
        
        # Eval Mode
        model.eval()
        with torch.no_grad():
            
            # Forward Pass on validation data
            logits, _ = model(data.x, data.edge_index, device)
            val_logits = logits[data.val_mask].float()
            val_labels = data.y[data.val_mask].float()
            validation_loss = loss_fn(val_logits, val_labels)
            writer.add_scalar('Loss/val', validation_loss.item(), epoch)

            # Calculate AUC metrics
            scores = torch.sigmoid(val_logits).cpu().numpy()
            truth_labels = val_labels.cpu().numpy()
            metrics_res = evaluate(scores, truth_labels, epoch)

        # Append metrics
        to_save = {
            "gnn": train_params['layer'],
            **{i: metrics_res["cls_auc"][i] for i in range(len(metrics_res["cls_auc"]))},
            "wt_auc": metrics_res["wt_auc"],
            "micro_auc": metrics_res["micro_auc"],
            "macro_auc": metrics_res["macro_auc"],
            "Epoch": epoch
        }
        all_metrics.append(to_save)

        print(f"Epoch: {epoch:03d}, Train Loss: {total_loss:.4f}, Val Loss: {validation_loss.item():.4f}")

        if (validation_loss < latest_loss):

            # Create filename for saving the model
            current_model = f"{model_dir}/e{epoch}_valLos_{validation_loss.item():.4f}.pt"

            # Update Saved Model Dictionary
            saved[current_model] = validation_loss
            # torch.save(model.state_dict(), f"{current_model}")
            if save_model: torch.save(model.state_dict(), f"{model_dir}/best_model.pt")

            # Update most current loss to current validation loss
            latest_loss = validation_loss
            print(f"Epoch {epoch}: Validation loss improved. Model saved to {current_model} and best_model.pt.")

        elif (validation_loss >= latest_loss):

            # Decrease patience count if no improvement
            max_patience_count -= 1
            print(f"Epoch {epoch}: Validation loss did not improve. Patience remaining: {max_patience_count}")

            # If patience is exhausted, exit the training loop
            if max_patience_count == 0: 
                print("Early stopping triggered.")
                break

        if epoch == max_epochs: 
            print(f"Reached maximum epochs: {max_epochs}. Stopping training.")
            break

    return optimizer, epoch, latest_loss


def test(model, data, model_name, train_params, epoch):
    print("Testing model")

    model.eval()
    with torch.no_grad():
        
        # Run model on test data
        logits, node_embeddings = model(data.x.to(device), data.edge_index.to(device), device)
        test_logits = logits[data.test_mask].float()
        scores = torch.sigmoid(test_logits).cpu().detach().numpy()
        truth_labels = data.y[data.test_mask].cpu().detach().numpy()

        test_loss = loss_fn(test_logits, data.y[data.test_mask].float())
        writer.add_scalar('Loss/test', test_loss.item(), epoch)
        print(f"Test Loss: {test_loss.item():.4f}")

    return scores, node_embeddings, truth_labels


def evaluate(scores, truth_labels, epoch):
    print("Evaluating model")

    cls_auc = metrics.roc_auc_score(truth_labels, scores, average=None)
    wt_auc = metrics.roc_auc_score(truth_labels, scores, average="weighted")
    micro_auc = metrics.roc_auc_score(truth_labels, scores, average="micro")
    macro_auc = metrics.roc_auc_score(truth_labels, scores, average="macro")

    writer.add_scalar("auc_micro", micro_auc)
    writer.add_scalar("auc_macro", macro_auc)
    writer.add_scalar("wt_auc", wt_auc)

    print(f"AUC Scores\n\nAUC Scores by Class: {cls_auc} | Weighted Avg AUC Score: {wt_auc} | Micro-Avg AUC Score: {micro_auc} | Macro-Avg AUC Score: {macro_auc}")

    return {
        "cls_auc": cls_auc,
        "wt_auc": wt_auc,
        "micro_auc": micro_auc,
        "macro_auc": macro_auc,
    }


def run_model(dataset, config_params, train_params, layer, model_dir):
    
    # Load first graph in dataset
    data = dataset[0]
    data.to(device)

    # Initialize model
    model = myGCN(**model_params)
    print("Model:\n\n", model)

    # Run train
    optimizer, epoch, validation_loss = train(model, data, train_params, model_dir, train_mode=training_mode)
    
    # Select best model (lowest loss) from saved dictionary
    if saved:
        # best_model = min(saved, key=saved.get)
        # print("Best model loaded from:", best_model)
        model.load_state_dict(torch.load(f"{model_dir}/best_model.pt"))
        print("Best model loaded")
    else:
        print("No improvement detected during training. Using the last epoch's model.")

    # Run test
    scores, node_embeddings, truth_labels = test(model, data, layer, train_params, epoch)
    metrics_res = evaluate(scores, truth_labels, epoch)

    return model, metrics_res, node_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Run GATConv")
    parser.add_argument("--data_folder", type=str, required=True, help="Train/val/test PyG dataset folder")
    parser.add_argument("--model", type=str, default="GATConv")
    parser.add_argument("--edge_strategy", type=str, default="trivial")
    parser.add_argument("--activation", type=str, default="celu")
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--WD", type=float, default=0.00001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--NUM_NODES", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--NUM_HIDDEN_LAYERS", type=int, default=1)
    parser.add_argument("--POST_NUM_MLP_LAYERS", type=int, default=1)
    parser.add_argument("--NUM_MLP_LAYERS", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True, default="None")
    parser.add_argument("--model_folder", type=str, required=True, default="None")
    parser.add_argument("--training_mode", type=str, required=True, default="None", help="inductive or transductive")
    parser.add_argument("--outputdir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, required=True, default="")
    return parser.parse_args()

def loss_fn(y_pred, labels):
    criterion_BCE = nn.BCELoss()
    loss = criterion_BCE(y_pred, labels)
    return loss


if __name__ == "__main__":

    # Set global access variables
    global DD_MM_YYYY
    global max_patience_count
    global max_num_epochs
    global latest_loss
    global device
    global writer
    global FOLDER
    global training_mode
    global experiment_name
    global save_model

    save_model = True

    saved = {}
    DD_MM_YYYY = datetime.now().strftime("%d_%m_%Y")
    DATE_TIME = datetime.now().strftime("%d%m%Y-%H%M%S")
    DAY_MONTH_YEAR = datetime.now().strftime("%d%m%Y")
    RANDOM_SHEET_NUMBER = 42

    latest_loss = math.inf
    max_patience_count = 15 # 10  # 0
    max_epochs = 1

    # Parse command-line arguments
    args = parse_args()

    # Model configuration
    layer = args.model
    edge_strategy = args.edge_strategy 
    model_name = args.model_name
    model_folder = args.model_folder
    activation = args.activation
    aggr = args.aggr
    hidden = args.hidden
    dropout = args.dropout
    NUM_HIDDEN_LAYERS = args.NUM_HIDDEN_LAYERS
    NUM_MLP_LAYERS = args.NUM_MLP_LAYERS
    POST_NUM_MLP_LAYERS = args.POST_NUM_MLP_LAYERS
    NUM_NODES = args.NUM_NODES
    print("Model Configuration:\n")
    print(f"Layer: {layer}")
    print(f"Model Name: {model_name}")
    print(f"Model Folder: {model_folder}")
    print(f"Activation Function: {activation}")
    print(f"Aggregation Method: {aggr}")
    print(f"Hidden Units: {hidden}")
    print(f"Dropout Rate: {dropout}")
    print(f"Number of Hidden Layers: {NUM_HIDDEN_LAYERS}")
    print(f"Number of MLP Layers: {NUM_MLP_LAYERS}")
    print(f"Number of Post MLP Layers: {POST_NUM_MLP_LAYERS}")
    print()

    # Training configuration
    lr = args.lr
    batch_size = args.batch_size
    WD = args.WD
    NUM_EPOCHS = args.epochs
    training_mode = args.training_mode
    experiment_name = args.experiment_name

    # Create directories
    FNAME = os.path.join(os.path.join(experiment_name, 'runs'), model_name)
    MODEL_DIR = f"{model_folder}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    data_folder = args.data_folder
    dataset = MyOwnDataset(data_folder, edge_strategy_name=edge_strategy)
    print('Tensorboard logs saved in directory: ', MODEL_DIR)
    print('Model is saved in directory: ', FNAME)
    print("Data is extracted from directory: ", data_folder)

    # Parameter configuration
    config_params = {
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "POST_NUM_MLP_LAYERS": POST_NUM_MLP_LAYERS,
        "NUM_MLP_LAYERS": NUM_MLP_LAYERS,
        "activation_fn_name": activation,
        "batch_size": batch_size,
        "WD": WD,
        "dropout_rate": dropout,
        "LR": lr,
        "hidden_channels": hidden,
        "NUM_NODES": NUM_NODES,
        "NUM_EPOCHS": NUM_EPOCHS,
        "layer": layer,
        "aggr": aggr
    }
    train_params = {**config_params}
    model_params = {
        **{
            "hidden_channels": None,
            "dataset": dataset,
            "NUM_HIDDEN_LAYERS": None,
            "layer_type": layer,
            "aggr_fn": config_params['aggr'],
        },
        **config_params,
    }
    
    # FPARAMS = os.path.join(MODEL_DIR, "model_params.pt")
    # torch.save(model_params, os.path.join(MODEL_DIR, "model_params.pt"))
    # print(f"Model parameters are saved in file: {FPARAMS}")

    # Device configuration
    cuda = args.cuda
    device = torch.device(f"cuda:{str(cuda)}" if (torch.cuda.is_available() and cuda >= 0) else "cpu")
    print("Training device: ", device)

    writer = SummaryWriter(FNAME)

    print(f'Executing : {os.path.basename(__file__)}')
    model, metrics_res, node_embeddings = run_model(dataset, config_params, train_params, layer, MODEL_DIR)
    to_save = {
        **{"gnn": layer},
        **{i: metrics_res["cls_auc"][i] for i in range(len(metrics_res["cls_auc"]))},
        **{
            "wt_auc": metrics_res["wt_auc"],
            "micro_auc": metrics_res["micro_auc"],
            "macro_auc": metrics_res["macro_auc"],
        },
    }

    df = pd.DataFrame(to_save, index=[0])
    print(df.T)


    # Inside your training loop, after computing 'to_save'
    df_all = pd.DataFrame(all_metrics)
    final_metrics = os.path.join(MODEL_DIR, experiment_name + "_final_metrics.csv")
    df_all.to_csv(final_metrics, index=False)
    print(f"Saved metrics to {final_metrics}")

    # Plot Aggregated AUC Scores
    plt.figure(figsize=(10, 6))
    plt.plot(df_all['Epoch'], df_all['wt_auc'], label='Weighted AUC')
    plt.plot(df_all['Epoch'], df_all['micro_auc'], label='Micro AUC')
    plt.plot(df_all['Epoch'], df_all['macro_auc'], label='Macro AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.title('Aggregated AUC Scores per Epoch')
    plt.legend()
    plt.grid(True)
    aggregated_auc_path = os.path.join(MODEL_DIR, experiment_name + "_aggregated_auc_per_epoch.png")
    plt.savefig(aggregated_auc_path)
    plt.close()
    print(f"Saved Aggregated AUC plot to {aggregated_auc_path}")

    # Plot Class-wise AUC Scores
    class_columns = [col for col in df_all.columns if isinstance(col, int) or col.isdigit()]

    plt.figure(figsize=(15, 10))
    for cls in class_columns:
        plt.plot(df_all['Epoch'], df_all[cls], label=f'Class {cls}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.title('Class-wise AUC Scores per Epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    class_auc_path = os.path.join(MODEL_DIR, experiment_name + "_class_wise_auc_per_epoch.png")
    plt.savefig(class_auc_path)
    plt.close()
    print(f"Saved Class-wise AUC plot to {class_auc_path}")
#     gnn         GATConv
# 0          0.649144
# 1          0.800848
# 2          0.580024
# 3          0.620693
# 4          0.654396
# 5          0.565983
# 6          0.534775
# 7          0.602258
# 8          0.663096
# 9          0.738213
# 10         0.750269
# 11         0.659286
# 12         0.666205
# 13         0.612111
# 14         0.669749
# 15         0.621542
# 16         0.665115
# 17         0.532809
# 18         0.555857
# 19         0.553855
# 20         0.602041
# 21         0.670953
# 22         0.849973
# 23          0.74076
# 24         0.786974
# wt_auc     0.657282
# micro_auc  0.726319
# macro_auc  0.653877