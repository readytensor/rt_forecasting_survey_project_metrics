

import numpy as np
import pandas as pd


def get_smape(y_true: np.ndarray, y_hat: np.ndarray):
    """
    Compute Symmetric Mean Absolute Percentage Error
    """
    denominator = np.abs(y_hat) + np.abs(y_true)

    # Mask for elements where the denominator is not zero
    non_zero_mask = denominator != 0

    # Initialize an array for SMAPE values, filled with zeros
    smape_values = np.zeros_like(y_true, dtype=float)

    # Apply SMAPE calculation where the denominator is not zero
    smape_values[non_zero_mask] = (
        2
        * np.abs(y_hat[non_zero_mask] - y_true[non_zero_mask])
        / denominator[non_zero_mask]
    )

    # Compute the mean of the SMAPE values
    mean_smape = 100 * np.mean(smape_values)

    return mean_smape


def get_mean_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute Mean Error
    """
    return np.mean(y_true - y_pred)


def get_wape(y_true: np.ndarray, y_hat: np.ndarray):
    """
    Calculate weighted absolute percent error (WAPE). 
    Weighting is by actuals (y_true).
    """
    abs_errors = np.abs(y_true - y_hat)
    if np.sum(abs_errors) == 0:
        return 0.0
    if np.sum(np.abs(y_true)) == 0:
        return 1.0
    return 100 * np.sum(abs_errors) / np.sum(np.abs(y_true))
