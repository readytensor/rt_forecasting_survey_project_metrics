
import os
import paths
import pandas as pd
from tabulate import tabulate
from scoring.forecasting import get_forecasting_scores
from utils import (
    get_models_list,
    get_datasets_list,
    get_dataset_files,
    get_prediction_file
)
from tqdm import tqdm  # Import tqdm for the progress bar


def calculate_and_save_scores():
    datasets = get_datasets_list()
    models = get_models_list()
    
    all_scores = []
    num_records_by_model = {model: 0 for model in models}
    # Dictionary to cache dataset files
    dataset_cache = {}
    for i, model in enumerate(models):
        print(f"Scoring model {i+1} of {len(models)}: {model}")
        for dataset in tqdm(datasets, desc="Scoring datasets", total=len(datasets)):
            if dataset not in dataset_cache:
                data = get_dataset_files(dataset)
                if data is None:
                    # print(f"Skipping calculation for dataset: {dataset}.")
                    dataset_cache[dataset] = -1
                    continue
                dataset_cache[dataset] = data
            else:
                data = dataset_cache[dataset]
            
            if data == -1:
                # no data found
                # print(f"Skipping calculation for dataset: {dataset}.")
                continue
            predictions = get_prediction_file(
                model=model,
                dataset=dataset,
                data_schema=data["data_schema"],
            )
            if predictions is None:
                # print(
                #     f"Error finding/reading prediction files for model {model} "
                #     f"and dataset {dataset}. Skipping calculation...")
                continue
            else:
                scores = get_forecasting_scores(
                    train_data=data["train_data"],
                    test_key=data["test_key"],
                    data_schema=data["data_schema"],
                    predictions=predictions,
                )
                scores["dataset"] = dataset
                scores["model"] = model
                all_scores.append(scores)
                num_records_by_model[model] += 1

    all_scores = pd.DataFrame(all_scores)

    cols_to_front = ["dataset", "model"]
    new_col_order = cols_to_front + \
        [col for col in all_scores.columns if col not in cols_to_front]
    output_file_path = os.path.join(paths.OUTPUTS_DIR, "accuracy_scores.csv")
    all_scores[new_col_order].to_csv(output_file_path, index=False)

    # Prepare the data as a list of lists
    table = [[model, num_files] for model, num_files in num_records_by_model.items()]
    
    # Print using tabulate for a nicely formatted table
    print("-"*80)
    print("Scores calculated:")
    print(tabulate(table, headers=["Dataset", "Num Models Scored"], tablefmt="fancy_grid"))
    print("-"*80)
    print("Scoring complete.")

