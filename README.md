# Forecasting Survey Project Metrics

This repository contains the scripts used to analyze the data generated from the forecasting survey project.

## Overview

The repository contains the following input data used in the analysis:

- **Models**: List of forecasting models used in the project.
- **Datasets**: List of forecasting datasets used in the project. There are 20 datasets with 6 variations each, making a total of 120 datasets. For each dataset, this repository contains the following:
  - Schema file: This file contains the schema of the dataset.
  - Test-key file: This file contains the ground truth values for the forecast horizon.
- **Forecasting artifacts**: These are the outputs of the forecasting models when the experiments are run. The following artifacts are stored for each model and dataset:
  - Prediction files: These files contain the forecasts for the forecast horizon for each combination of model and dataset.
  - Train and prediction log files: These files contain the printed logs during training and inference steps. The execution times, and CPU/GPU memory usage are reported in the log files. These are generated by the forecasting models during training and inference steps.

There are scripts provided to analyze the data as follows:

- **Calculate forecast metrics**: Script to calculate forecasting metrics such as rmse, mae, smape, etc. for each model and dataset.
- **Extract execution times and memory usage**: Script to extract execution times and memory usage from the log files. This includes the training and inference times, and CPU/GPU memory usage.

## Usage

1. Create virtual environment and install dependencies in `requirements.txt`.
2. Run the script `src/main.py` to calculate the forecasting metrics. This will create the file `accuracy_scores.csv` containing the forecasting metrics for each model and dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

TBD.
