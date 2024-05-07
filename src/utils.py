import os
import json
import pandas as pd
import paths
from pathlib import Path
from typing import List, Tuple
from scoring.forecasting import process_time_cols_in_dfs
import paths


def get_models_list():
    models_df = pd.read_csv(paths.MODELS_FPATH, encoding="iso-8859-1")
    models = models_df["model_id"].tolist()
    return models

def get_datasets_list():
    datasets_df = pd.read_csv(paths.DATASETS_FPATH, encoding="iso-8859-1")
    datasets = datasets_df["dataset_id"].tolist()
    return datasets

def get_dataset_files(dataset:str):
    dataset_dir_path = os.path.join(paths.DATASETS_DIR, dataset)
    data_schema_path = os.path.join(dataset_dir_path, f"{dataset}_schema.json")
    with open(data_schema_path, "r") as f:
        data_schema = json.load(f)

    time_col = data_schema["timeField"]["name"]
    time_dtype = data_schema["timeField"]["dataType"]
    train_data = pd.read_csv(
        os.path.join(dataset_dir_path, f"{dataset}_train.csv.gz")
    )
    test_key = pd.read_csv(
        os.path.join(dataset_dir_path, f"{dataset}_test_key.csv.gz")
    )
    train_data, test_key = process_time_cols_in_dfs(
        time_col, time_dtype, train_data, test_key
    )
    data = {
        "data_schema": data_schema,
        "train_data": train_data,
        "test_key": test_key,
    }
    return data

def get_prediction_file(model:str, dataset:str, data_schema:dict):
    predictions_file_path = os.path.join(
        paths.PREDICTIONS_DIR, model, dataset, "predictions.csv.gz"
    )
    time_col = data_schema["timeField"]["name"]
    time_dtype = data_schema["timeField"]["dataType"]
    try:
        predictions = pd.read_csv(
            predictions_file_path,
            # parse_dates=[time_col],
        )
        predictions = process_time_cols_in_dfs(time_col, time_dtype, predictions)
        return predictions[0]
    except Exception as e:
        return None

