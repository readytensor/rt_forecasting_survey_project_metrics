import os

ROOT_DIR = os.path.dirname(__file__)

INPUTS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "inputs")

PREDICTIONS_DIR = os.path.join(INPUTS_DIR, "predictions")
DATASETS_DIR = os.path.join(INPUTS_DIR, "datasets")
LOGS_DIR = os.path.join(INPUTS_DIR, "logs")
MODELS_DIR = os.path.join(INPUTS_DIR, "models")

MODELS_FPATH = os.path.join(MODELS_DIR, "models.csv")
DATASETS_FPATH = os.path.join(DATASETS_DIR, "datasets.csv")

OUTPUTS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "outputs")
