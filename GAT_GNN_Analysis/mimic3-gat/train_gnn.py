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
from custom_layers import activation_name_implementation, get_layer_impl_from_layer_type
from torch_geometric.utils import subgraph

# Setting the Edge Selection Strategy from the MIMIC3 Benchmark Data Processing Pipeline (Multitask learning and benchmarking with clinical time series data)
# Source: https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23/edge_strategies.py
#- Omit the check for unique edges in the edge strategies
#- Omit the edge strategies that are not used in the code
#- Omit the feature_anomaly_edges_automatic() from the pre-processing pipeline analysis of how nodes should be connected


activation_fns = {
    # 'relu': nn.ReLU,
    # 'leaky_relu': nn.LeakyReLU,
    # 'sigmoid': nn.Sigmoid,
    # 'tanh': nn.Tanh,
    # 'elu': nn.ELU,
    'selu': nn.SELU,
    'celu': nn.CELU,
    'gelu': nn.GELU,
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




class myGCN(torch.nn.Module):
    def __init__(self,hidden_channels, dataset, layer_type, NUM_HIDDEN_LAYERS=1, NUM_MLP_LAYERS=1, POST_NUM_MLP_LAYERS=1, aggr_fn="mean", dropout_rate=0.0, activation_fn_name="gelu", layer_norm_flag=False, model_name = "", **kwargs,):
        
        super().__init__()

        # Set random seed for reproducibility
        torch.manual_seed(RANDOM_SHEET_NUMBER)

        # Store configuration parameters
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.NUM_MLP_LAYERS = NUM_MLP_LAYERS
        self.POST_NUM_MLP_LAYERS = POST_NUM_MLP_LAYERS
        self.layer_norm_flag = layer_norm_flag
        self.dropout_rate = dropout_rate
        self.model_name = model_name

        # Initialize activation function
        self.activation_fn = activation_fns[activation_fn_name]()

        # Retrieve layer implementation and parameters based on layer_type
        layer = get_layer_impl_from_layer_type[layer_type]["impl"]
        layer_params = get_layer_impl_from_layer_type[layer_type]["params"]
        layer_params['aggr'] = aggr_fn

        # Optional Layer Normalization
        self.layer_norm = torch.nn.LayerNorm(hidden_channels) if layer_norm_flag else None

        # Initialize Hidden Graph Convolutional Layers
        self.hidden_layers = self._build_hidden_layers(layer, hidden_channels, layer_params)

        # Initialize MLP Layers Before Graph Convolutions
        self.mlp_layers = self._build_mlp_layers(dataset.num_features, hidden_channels, NUM_MLP_LAYERS)

        # Initialize MLP Layers After Graph Convolutions
        self.post_mlp_layers = self._build_post_mlp_layers(dataset.num_features, hidden_channels, POST_NUM_MLP_LAYERS)

        # Final Linear Layer for Classification
        self.lin1 = torch.nn.Linear(hidden_channels, dataset.num_classes)
        torch.nn.init.xavier_uniform_(self.lin1.weight)

    def _build_hidden_layers(self, layer, hidden_channels, layer_params):

        layers = OrderedDict()
        for i in range(self.NUM_HIDDEN_LAYERS):
            print(f"Adding graph hidden layer: {i}")
            layers[str(i)] = layer(hidden_channels, hidden_channels, **layer_params)
            # Initialize weights using Xavier Uniform
            for param in layers[str(i)].parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        return Sequential(layers)

    def _build_mlp_layers(self, input_dim, hidden_channels, num_layers):

        mlp_layers = OrderedDict()
        for i in range(num_layers):
            print(f"Adding MLP layer: {i}")
            if i == 0:
                mlp_layers[str(i)] = torch.nn.Linear(input_dim, hidden_channels)
            else:
                mlp_layers[str(i)] = torch.nn.Linear(hidden_channels, hidden_channels)
            torch.nn.init.xavier_uniform_(mlp_layers[str(i)].weight)

        return Sequential(mlp_layers)

    def _build_post_mlp_layers(self, input_dim, hidden_channels, num_layers):

        post_mlp_layers = OrderedDict()
        for i in range(num_layers):
            print(f"Adding post-MLP layer: {i}")
            if i == 0:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(input_dim + hidden_channels),
                    torch.nn.Linear(input_dim + hidden_channels, hidden_channels)
                )
            else:
                post_mlp_layers[str(i)] = Sequential(
                    torch.nn.LayerNorm(hidden_channels),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
            # Initialize weights using Xavier Uniform
            torch.nn.init.xavier_uniform_(post_mlp_layers[str(i)][1].weight)
            
        return Sequential(post_mlp_layers)

    def forward(self, raw, edge_index, device, edge_weight=None):

        # Move edge indices to the specified device
        edge_index = edge_index.to(device)

        x = raw

        # MLP Layers Before Graph Convolutions
        for i in range(self.NUM_MLP_LAYERS):
            x = self.mlp_layers[str(i)](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        node_embeddings = []
        # Hidden Graph Convolutional Layers
        for i in range(self.NUM_HIDDEN_LAYERS):
            if edge_weight is None:
                node_embedding = self.hidden_layers[str(i)](x, edge_index)
            else:
                node_embedding = self.hidden_layers[str(i)](x, edge_index, edge_weight)
            node_embeddings.append(node_embedding)
            x = self.activation_fn(node_embedding)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Concatenate raw features with final embeddings
        x = torch.cat((raw, x), dim=1)

        # Post MLP Layers After Graph Convolutions
        for i in range(self.POST_NUM_MLP_LAYERS):
            x = self.post_mlp_layers[str(i)](x)
            x = self.activation_fn(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Final Linear Layer for Classification
        x = self.lin1(x)

        # Apply Sigmoid Activation
        return torch.sigmoid(x), node_embeddings


def train(model, data, train_params, model_dir, training_mode, model_path=None, **kwargs):

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
        data.edge_index if training_mode == 'transductive' else subgraph(train_nodes, data.edge_index, relabel_nodes=True)[0],
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

            # Compute Validation Loss
            validation_loss = loss_fn(logits[data.val_mask], data.y[data.val_mask].float())
            writer.add_scalar('Loss/val', validation_loss, epoch)

        print(f"Epoch: {epoch:03d}, Train Loss: {total_loss:.4f}, Val Loss: {validation_loss.item():.4f}")

        if (validation_loss < latest_loss):

            # Create filename for saving the model
            current_model = f"{model_dir}/e{epoch}_valLos_{validation_loss.item():.4f}.pt"

            # Update Saved Model Dictionary
            saved[current_model] = validation_loss
            torch.save(model.state_dict(), f"{current_model}")
            torch.save(model.state_dict(), f"{model_dir}/best_model.pt")

            # Update most current loss to current validation loss
            latest_loss = validation_loss
        
        elif (validation_loss >= latest_loss):

            # Decrease patience count if no improvement
            max_patience_count -= 1

            # If patience is exhausted, exit the training loop
            if max_patience_count == 0: break

        if epoch == max_epochs: break

    return optimizer, epoch, latest_loss


def test(model, data, model_name, train_params):
    print("Testing model")

    model.eval()
    with torch.no_grad():
        
        # Run model on test data
        logits, node_embeddings = model(data.x, data.edge_index, device)
        logits = logits[data.test_mask] # Filter output from test set
        scores = logits.cpu().detach().numpy() # Move logits to CPU and convert to numpy array

        print(f"Test Loss: {loss_fn(logits, data.y[data.test_mask].float()):.4f}")

        # Use mask on dataset to select test labels
        truth_labels = data.y[data.test_mask].cpu().detach().numpy()
        
        return scores, node_embeddings, truth_labels


def evaluate(scores, truth_labels):
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
    optimizer, epoch, validation_loss = train(model, data, train_params, model_dir, training_mode=training_mode)
    
    # Select best model (lowest loss) from saved dictionary
    best_model = min(saved, key=saved.get)
    model.load_state_dict(torch.load(best_model))
    print("Best model: ", best_model)

    # Run test
    scores, node_embeddings, truth_labels = test(model, data, layer, train_params)
    metrics_res = evaluate(scores, truth_labels)

    return model, metrics_res, node_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Run LECONV")
    parser.add_argument("--data_folder", type=str, required=True, help="Train/val/test PyG dataset folder")
    parser.add_argument("--model", type=str, default="LEConv")
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
    dataset = MyOwnDataset(data_folder)
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
    
    FPARAMS = os.path.join(MODEL_DIR, "model_params.pt")
    torch.save(model_params, os.path.join(MODEL_DIR, "model_params.pt"))
    print(f"Model parameters are saved in file: {FPARAMS}")

    # Device configuration
    cuda = args.cuda
    device = torch.device(f"cuda:{str(cuda)}" if (torch.cuda.is_available() and cuda >= 0) else "cpu")
    print("Training device: ", device)

    writer = SummaryWriter(FNAME)

    print(f'Executing : {os.path.basename(__file__)}')
    model, metrics_res, node_embeddings = run_model(dataset, config_params, train_params, layer, MODEL_DIR)
    to_save = {
        **{"gnn": layer},
        **{i: metrics_res["auc_c"][i] for i in range(len(metrics_res["auc_c"]))},
        **{
            "aucw": metrics_res["aucw"],
            "auc_micro": metrics_res["auc_micro"],
            "auc_macro": metrics_res["auc_macro"],
        },
    }

    df = pd.DataFrame(to_save, index=[0])
    print(df.T)