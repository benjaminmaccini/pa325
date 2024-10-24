import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Callable

def guassian_function(x, a, b, c):
    return a * np.exp(-b * x) + c

def base_salinity_function(X, c_local):
    """
    Model function where c_local is the parameter to fit.

    This function gives c_out

    Parameters:
    - c_local: The parameter to fit (salinity for local inflows).
    - flow_local, flow_in, c_in, flow_out: Fixed known parameters.
    """
    flow_local, flow_in, flow_out, c_in = X
    return (c_in * flow_in + c_local * flow_local) / (-1 * flow_out)


def remove_outliers(df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to check for outliers.
        multiplier (float): IQR multiplier for determining outliers. Default is 1.5.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df.reset_index(drop=True)

def keep_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Keep only outliers from a pandas Series.

    Args:
        series (pd.Series): Input Series.
        multiplier (float): IQR multiplier for determining outliers. Default is 1.5.

    Returns:
        pd.Series: Series with only outliers kept.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    return series[(series < lower_bound) | (series > upper_bound)]

def regression_error(model_function, x_data, y_data, p0, bounds: tuple, check_finite: bool = False, nan_policy="omit") -> float:
    popt, pcov = curve_fit(model_function, x_data, y_data, p0=p0, bounds=bounds, check_finite=check_finite, nan_policy=nan_policy)
    print(f"Optimized parameter: {popt[0]}")

    # Compute fitted values
    y_fit = model_function(x_data, *popt)

    # Remove NaN values before calculating error metrics
    valid_mask = ~pd.isna(y_data) & ~pd.isna(y_fit)
    y_data_valid = y_data[valid_mask]
    y_fit_valid = y_fit[valid_mask]

    print(f"Length of y_data_valid: {len(y_data_valid)}")
    print(f"Length of y_fit_valid: {len(y_fit_valid)}")
    print(f"Unique values in y_data_valid: {pd.Series(y_data_valid).nunique()}")
    print(f"Unique values in y_fit_valid: {pd.Series(y_fit_valid).nunique()}")

    print(f"Mean Squared Error: {mean_squared_error(y_data_valid, y_fit_valid)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_data_valid, y_fit_valid)}")
    print(f"Pearson Correlation: {np.corrcoef(y_data_valid, y_fit_valid)[0, 1]}")

    # Ensure no constant or all NaN values exist
    if len(y_data_valid) > 1 and pd.Series(y_data_valid).nunique() > 1 and pd.Series(y_fit_valid).nunique() > 1:
        # Calculate Spearman correlation only if valid
        spearman_corr = pd.Series(y_data_valid).corr(pd.Series(y_fit_valid), method='spearman')
        print(f"Spearman Correlation: {spearman_corr}")
    else:
        print("Spearman Correlation could not be calculated due to insufficient data variability.")

    return popt[0]

def moving_average(data: pd.Series, window: int = 5) -> pd.Series:
    """
    Calculate the moving average of a given data series.

    Args:
        data (pd.Series): Input data series.
        window (int): Size of the moving window.

    Returns:
        pd.Series: Series containing the moving average.
    """
    return data.rolling(window=window).mean()

def add_moving_average_column(data: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """
    Add a column with the x-day moving average to the input data.

    Args:
        data (pd.DataFrame): Input data frame.
        column (str): Name of the column to calculate moving average for.
        window (int): Size of the moving window.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the moving average.
    """
    ma_column_name = f'{column}_{window}day_MA'
    data[ma_column_name] = moving_average(data[column], window)
    return data

def add_lag(df: pd.DataFrame, n: int) -> pd.Series:
    """
    Transform a pandas series so that it is lagged by n days.

    Args:
        series (pd.Series): Input series with datetime index.
        n (int): Number of days to lag the series by.

    Returns:
        pd.Series: Lagged series.
    """
    return df.shift(periods=n, freq='D')

def segments_fit(X, Y, count):
    """
    Credit: https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
    """
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)
