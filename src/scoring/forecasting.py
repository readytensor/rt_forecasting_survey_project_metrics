import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from scipy.stats import linregress

from scoring.metrics import get_smape, get_wape


def get_forecasting_scores(train_data, test_key, predictions, data_schema):
    """
    Calculates various metrics given the test_key, predictions, and schema file.
    Returns:
        - scores: json object with metric names as keys, and metric values as values
    """
    id_col = data_schema["idField"]["name"]
    time_col = data_schema["timeField"]["name"]
    target_col = data_schema["forecastTarget"]["name"]
    time_dtype = data_schema["timeField"]["dataType"]

    train_data, predictions, test_key = process_time_cols_in_dfs(
        time_col, time_dtype, train_data, predictions, test_key
    )

    predictions[time_col] = predictions[time_col].astype(str)
    test_key[time_col] = test_key[time_col].astype(str)
    predictions = predictions[[id_col, time_col, "prediction"]]
    # print(predictions)
    predictions = predictions.merge(
        test_key[[id_col, time_col, target_col]], on=[id_col, time_col]
    )

    predictions.sort_values(by=[id_col, time_col], inplace=True)

    perc_pred_missing = np.round(
        100 * (1 - predictions.shape[0] / test_key.shape[0]), 4
    )

    rmse = []
    rmsse = []
    mae = []
    mase_ = []
    smape_ = []
    wape_ = []
    r2 = []

    unique_series = test_key[id_col].unique().tolist()
    for ser in unique_series:
        predictions_filtered = predictions[predictions[id_col] == ser]
        train_data_filtered = train_data[train_data[id_col] == ser]

        # get naive forecast
        y_hat_naive = get_y_hat_naive(
            train_data_filtered=train_data_filtered,
            forecast_len=predictions_filtered.shape[0],
            target_col=target_col,
        )

        # rmse and rmsse
        rmse_ser, rmsse_ser = get_rmse_and_rmsse(
            y_true=predictions_filtered[target_col].values,
            y_hat=predictions_filtered["prediction"].values,
            y_hat_naive=y_hat_naive,
        )
        rmse.append(rmse_ser)
        rmsse.append(rmsse_ser)

        # Calculate MAE and MASE
        mae_ser, mase_ser = get_mae_and_mase(
            y_true=predictions_filtered[target_col].values,
            y_hat=predictions_filtered["prediction"].values,
            y_hat_naive=y_hat_naive,
        )
        mae.append(mae_ser)
        mase_.append(mase_ser)

        # Calculate sMAPE
        smape_ser = get_smape(
            predictions_filtered[target_col], predictions_filtered["prediction"]
        )
        smape_.append(smape_ser)

        # Calculate WAPE
        wape_ser = get_wape(
            predictions_filtered[target_col].values,
            predictions_filtered["prediction"].values,
        )
        wape_.append(wape_ser)

        # Calculate r-squared
        try:
            _, _, r_value, _, _ = linregress(
                predictions_filtered[target_col], predictions_filtered["prediction"]
            )
        except ValueError:
            r_value = 0.0

        r2_ser = r_value * r_value
        r2.append(r2_ser)

    scores = {
        "rmse": np.round(np.mean(rmse), 4),
        "rmsse": np.round(np.mean(rmsse), 4),
        "mae": np.round(np.mean(mae), 4),
        "mase": np.round(np.mean(mase_), 4),
        "smape": np.round(np.mean(smape_), 4),
        "wape": np.round(np.mean(wape_), 4),
        "r2": np.round(np.mean(r2), 4),
        "perc_pred_missing": perc_pred_missing,
    }

    scores = replace_nans_with_none(scores)

    return scores


def get_y_hat_naive(
    train_data_filtered: pd.DataFrame, forecast_len: int, target_col: str
) -> np.ndarray:
    """
    Calculates the mean of the last `forecast_len` values for the target
    series and returns it as the forecast for each epoch in the forecast
    window.

    Args:
        train_data_filtered (pd.DataFrame): Dataframe for the train
            (historical) values for the given series.
        forecast_len (int): Length of the forecast window.
        target_col (str): Name of the target column in the data.

    Returns:
        np.ndarray: An array of length `forecast_len` which contains the
            mean value of the last `forecast_len` in history as forecast
            for each time step.

    """
    # Calculate mean of last `forecast_len` values
    mean_ = train_data_filtered[target_col].values[-forecast_len:].mean()
    # generrate the np array of naive forecast
    y_hat_naive = np.ones(forecast_len) * mean_
    return y_hat_naive


def replace_nans_with_none(scores: dict):
    """
    Replace any NaN values for metrics with None.
    This is needed before returning the metrics' dictionary
    as json. Otherwise, there is a Json parsing error.

    Args:
        scores (dict): A dictionary of metrics names and their values.

    Returns:
        dict: Dictionary of metrics with NaN (if any) replaced by None
    """
    scores_updated = {**scores}
    # replace NaN with None so json can interpret it as null
    for key, value in scores_updated.items():
        if isinstance(value, float) and np.isnan(value):
            scores_updated[key] = None
    return scores_updated


def process_time_cols_in_dfs(
    time_col: str, time_dtype: str, *dfs: pd.DataFrame
) -> List[pd.DataFrame]:
    """
    Converts the time column in multiple dataframes to datetime format and
    removes time zone.

    Parameters:
    time_col (str): The name of the column containing time data.
    time_dtype (str): The specific data type of time col.
    dfs (variable number of DataFrame): The dataframes to process.

    Returns:
    List[DataFrame]: A list of dataframes with the time column processed.
    """
    processed_dfs = []

    # Define datetime formats based on time_dtype
    datetime_formats = []
    if time_dtype == "INT":
        datetime_formats = []
    elif time_dtype == "DATE":
        datetime_formats = ["%Y-%m-%d"]
    elif time_dtype == "DATETIME":
        datetime_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]

    # Process each dataframe
    for df in dfs:
        for datetime_format in datetime_formats:
            try:
                df[time_col] = pd.to_datetime(df[time_col], format=datetime_format)
                df[time_col] = df[time_col].dt.tz_localize(None)
                break
            except:
                print(f"format didnt work: {datetime_format}")
                # raise ValueError
        processed_dfs.append(df)
    return processed_dfs


def get_mae_and_mase(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    y_hat_naive: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate Mean Absolute Error (MAE) and Mean Absolute Scaled Error (MASE).

    This function computes the MAE and MASE for given actual and forecasted values.
    MASE is calculated using the given naive forecast.

    Args:
        y_true (np.ndarray): The actual values.
        y_hat (np.ndarray): The forecasted values.
        y_hat_naive (np.ndarray): Naive forecast for the series.

    Returns:
        Tuple[float, float]: A tuple containing the MAE and MASE values.

    Raises:
        ValueError: If the scale used for MASE calculation is zero or too close
    """
    abs_errors = np.abs(y_true - y_hat)
    mae = np.mean(abs_errors)
    naive_abs_errors = np.abs(y_true - y_hat_naive)
    scale = np.mean(naive_abs_errors)
    # raise_if_scale_is_zero(scale)
    if scale < 1e-4:
        # Cannot calculate mase. Setting value to 1.0
        mase = 1.0
    else:
        mase = mae / scale
    return mae, mase


def get_rmse_and_rmsse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    y_hat_naive: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate Root Mean Squared Error (RMSE) and Root Mean Squared Scaled Error (RMSSE).

    This function computes the RMSE and RMSSE for given actual and forecasted values.
    RMSSE is calculated using the given naive forecast.

    Args:
        y_true (np.ndarray): The actual values.
        y_hat (np.ndarray): The forecasted values.
        y_hat_naive (np.ndarray): Naive forecast for the series.

    Returns:
        Tuple[float, float]: A tuple containing the RMSE and RMSSE values.

    Raises:
        ValueError: If the scale used for RMSSE calculation is zero or too close to
                    zero.
    """
    sq_errors = np.square(y_true - y_hat)
    rmse = np.sqrt(np.mean(sq_errors))
    naive_sq_errors = np.square(y_true - y_hat_naive)
    # naive_sq_errors = np.square(x_t[m:] - x_t[:-m])
    scale = np.sqrt(np.mean(naive_sq_errors))
    # raise_if_scale_is_zero(scale)
    if scale < 1e-4:
        # Cannot calculate rmsse. Setting value to 1.0
        rmsse = 1.0
    else:
        rmsse = rmse / scale
        # print(round(rmse, 4), round(scale, 4), round(rmsse, 4))
    return rmse, rmsse


def raise_if_scale_is_zero(scale: float):
    """
    Raise a ValueError if the scale value is zero or very close to zero.

    This function checks if the given scale value is effectively zero, which is used
    to avoid division by zero or very small numbers in error calculations like MASE
    or RMSSE.

    Args:
        scale (float): The scale value to check.

    Raises:
        ValueError: If the scale is zero or too close to zero.
    """
    if np.isclose(scale, 0, atol=1e-5):
        raise ValueError(
            "Scale value is zero or too close to zero;"
            " cannot compute MASE for a perfectly periodic series."
        )
