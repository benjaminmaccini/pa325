import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional

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


def base_salt_mass_balance(X, salt_local):
    """
    Based on:
        flow_out(salt_out) = flow_in(salt_in) + flow_local(salt_local)

    Gives flow_local
    """
    flow_out, flow_in, salt_out, salt_in = X
    return (flow_out * salt_out - flow_in * salt_in) / salt_local


def log_transform_sal_diff(X, c_local):
    """
    Take the log of the salinity differences

    Gives np.log(delta_c_out)

    14) $$ \log ( \Delta sal_{out} ) = \log ( \Delta sal_{local} ) + \log ( flow_{local} ) - \log ( flow_{out} )$$
    """
    flow_local, flow_out, c_in = X

    # Ensure positive values for log
    safe_sal_diff = c_local - c_in

    return np.log(safe_sal_diff) + np.log(flow_local) - np.log(flow_out)


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

def regression_error(
    model_function,
    x_data,
    y_data,
    p0,
    bounds: tuple,
    check_finite: bool = False,
    nan_policy="omit"
) -> float:
    """
    For unknown parameters, fit a curve and output the regression error

    Usage:
        x_data = np.array([flow_local, flow_in, flow_out, c_in])

        # Fit the curve on c_local, adjusting it to minimize the error
        # Compare against smoothed salinity
        # Median Salinity For Non-Point Sources in stormwater runoff (Dirrigl et al, 2016)
        optimal_sal = analysis.regression_error(
            model_function=analysis.base_salinity_function,
            x_data=x_data,
            y_data=analysis.moving_average(c_out,2),
            p0=[671.],
            bounds=((0,10000)),
        )

    Params:
        model_function: A callable function to be fit
        x_data: Parameters for the function
        y_data: Function output to fit on
        p0: A list of initial parameter values
        bounds: Range for the parameters to be fit in
        check_finite:
        nan_policy: Whether or not to omit null values
    """
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

def get_weekly_maxima(
    df: pd.DataFrame,
    columns: list[str],
    drop_if_max: Optional[dict] = None
) -> pd.DataFrame:
    """
    Calculate weekly maxima for specified columns in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of column names to calculate maxima for
        drop_if_max: Optional dictionary with column name as key and value to check for dropping
                    Example: {'salinity': 0} will drop weeks where salinity maxima is 0

    Returns:
        DataFrame with weekly maxima for specified columns
    """
    df_weekly = df.copy()

    # Calculate weekly maxima
    weekly_maxima = df_weekly[columns].resample('W').max()

    # Drop weeks based on specific maxima if requested
    if drop_if_max is not None:
        for col, value in drop_if_max.items():
            if col in columns:
                weekly_maxima = weekly_maxima[weekly_maxima[col] != value]

    return weekly_maxima
