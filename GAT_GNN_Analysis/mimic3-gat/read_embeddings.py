import os 
import pandas as pd 
import torch
import pickle 
import numpy as np
from datetime import datetime

# Reading Embeddings from the MIMIC3 Benchmark Data Processing Pipeline (Multitask learning and benchmarking with clinical time series data)
# Source: https://github.com/ds4dh/mimic3-benchmarks-GraDSCI23/read_embeddings.py

# import pickle data  folder  data/phenotyping/statistical_features 
# ls data/phenotyping/statistical_features 
# test_X          test_ts         train_X         train_ts        val_X           val_ts
# test_names      test_y          train_names     train_y         val_names       val_y


def read_pickle_file(pickle_file):
    with open(pickle_file, "rb") as f: return pickle.load(f)

def get_statistical_features(FOLDER_PATH='data/phenotyping/statistical_features'):
    data = {}
    prefixes = ['train', 'val', 'test']
    suffixes = ['names', 'ts', 'X', 'y']
    
    for prefix in prefixes:
        for suffix in suffixes:
            data[f"{prefix}_{suffix}"] = read_pickle_file(os.path.join(FOLDER_PATH, f"{prefix}_{suffix}"))
    
    dfs = {}
    for prefix in prefixes:
        df = pd.DataFrame({
            'stat_features': data[f"{prefix}_X"].tolist(),
            'name': data[f"{prefix}_names"],
            'y': data[f"{prefix}_y"].tolist()
        })
        df.set_index('name', inplace=True)
        dfs[prefix] = df

    return dfs['train'], dfs['val'], dfs['test']


def get_lstm_embeddings(folder):
    print(f'Reading LSTM embeddings from folder: {folder}')

    # Read all LSTM Embeddings from MIMIC3 Benchmark Data Processing Pipeline - "Path to the data of (LSTM) phenotyping task"
    # Hidden States, Labels, Predictions, and Names for Train, Validation, and Test sets
    data = {}
    for file in os.listdir(folder): data[file] = np.load(os.path.join(folder, file))

    keys = {
        "hidden_hn_train.npy": "train_embedding",
        "hidden_hn_val.npy": "val_embedding",
        "hidden_hn_test.npy": "test_embedding",
        "labels_train.npy": "train_ys",
        "labels_val.npy": "val_ys",
        "labels_test.npy": "test_ys",
        "predictions_train.npy": "train_predictions",
        "predictions_val.npy": "val_predictions",
        "predictions_test.npy": "test_predictions",
        "name_train.npy": "train_name",
        "name_val.npy": "val_name",
        "name_test.npy": "test_name",
    }
    
    # Map numpy file names to their respective keys
    mappings = {}
    for file, variable in keys.items(): mappings[variable] = data[file]

    # Print the shape of each variable
    for key, value in mappings.items(): print(f"{key}: {value.shape}")

    return mappings


def read_lstm_embeddings(folder=None):

    if folder is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
        folder = f'data/BCE_LSTM_{current_date}/'

    data = get_lstm_embeddings(folder)

    # Separate the data into train, validation, and test sets
    #* Get the columns that contain the training set data
    train_cols = [k for k in data.keys() if 'train' in k and 'input' not in k]

    #TODO_DELETE: assert all first dimension is the same
    assert all([data[k].shape[0] == data[train_cols[0]].shape[0] for k in train_cols])
    # assert all second dimension is the same
    # assert all([data[k].shape[1] == data[train_cols[0]].shape[1] for k in train_cols])

    #TODO_CHANGE
    train_df = pd.DataFrame({ 'name':data['train_name'],'ys':data['train_ys'].tolist(),'lstm_embedding':data['train_embedding'].tolist() })
    train_df.head()

    # use name column as index 
    train_df.set_index('name', inplace=True)
    assert len(train_df.loc[train_df['ys'].isna()]) == 0


    #* Get the columns that contain the validation set data
    val_cols = [k for k in data.keys() if 'val' in k and 'input' not in k]

    #TODO_DELETE: assert all first dimension is the same
    assert all([data[k].shape[0] == data[val_cols[0]].shape[0] for k in val_cols])

    #TODO_CHANGE
    val_df = pd.DataFrame({ 'name':data['val_name'], 'ys':data['val_ys'].tolist(), 'lstm_embedding':data['val_embedding'].tolist() })
    val_df.head 
    val_df.set_index('name', inplace=True)


    #* Get the columns that contain the test set data
    test_cols = [k for k in data.keys() if 'test' in k and 'input' not in k]
    
    #TODO_DELETE: assert all first dimension is the same
    assert all([data[k].shape[0] == data[test_cols[0]].shape[0] for k in test_cols])

    #TODO_CHANGE
    test_df = pd.DataFrame({ 'name':data['test_name'], 'ys':data['test_ys'].tolist(), 'lstm_embedding':data['test_embedding'].tolist() })
    test_df.head() 
    test_df.set_index('name', inplace=True)


    return train_df, val_df, test_df


def get_features():

    # Read the statistical features and lstm embeddings
    stat_train_df, stat_val_df, stat_test_df = get_statistical_features()
    lstm_train_df, lstm_val_df, lstm_test_df = read_lstm_embeddings("data/BCE_LSTM_2024-11-30/")
    # mgrn_train_df, mgrn_val_df, mgrn_test_df = read_mgrnn_embeddings()

    # Concatenate the statistical and lstm embeddings
    train_df = pd.concat([stat_train_df, lstm_train_df], axis=1)
    val_df = pd.concat([stat_val_df, lstm_val_df], axis=1)
    test_df = pd.concat([stat_test_df, lstm_test_df], axis=1)

    #TODO_DELETE: assert the dataframes are the same size
    assert len(train_df) == len(stat_train_df) == len(lstm_train_df)
    assert len(val_df) == len(stat_val_df) == len(lstm_val_df)
    assert len(test_df) == len(stat_test_df) == len(lstm_test_df)
    # assert len(train_df) > len(val_df) and len(val_df) > len(test_df), "Train, val, test dataframes are not Correct sizes (train > val > test) but instead: train: {}, val: {}, test: {}".format(len(train_df), len(val_df), len(test_df))
    assert all([df[col].isna().sum() == 0 for df in [train_df, val_df, test_df] for col in df.columns]), "There are NaNs in the dataframes"
    

    return train_df, val_df, test_df
