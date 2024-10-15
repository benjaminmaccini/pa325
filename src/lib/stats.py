import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


def check_missing_values(df):
    """Check for missing values in the dataset."""
    missing = df.isnull().sum()
    return missing[missing > 0]

def detect_outliers(df, column, threshold=3):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

def plot_boxplot(df, columns):
    """Create boxplots to visualize distribution and potential outliers for multiple columns."""
    n_cols = 2
    n_rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    ii = 0
    for i, column in enumerate(columns):
    	if column in df.columns:
            sns.boxplot(x=df[column], ax=axes[i])
            axes[i].set_title(f'Boxplot: {column}')
            ii += 1

    # Remove any unused subplots
    for j in range(ii+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_histogram(df, column, bins=30):
    """Plot a histogram of the data."""
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(f'Histogram: {column}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def check_stationarity(df, column, target: str = "stationary") -> bool:
    """
    Check stationarity using Augmented Dickey-Fuller test.

    Times series with trends and seasonality are non-stationary

    https://otexts.com/fpp2/stationarity.html

    If p-value > 0.05 data has unit root and is non-stationary
    If p-value <= 0.05 Reject null hypothesis and data is stationary
    """
    try:
        result = adfuller(df[column].dropna())
        is_stationary = float(result[1]) <= 0.05
        if target == "stationary" and is_stationary:
            print(f"\n{column}\n")
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print(f"Critical Values: \n{'\n'.join(f'\t {k}: {str(v)}' for k, v in result[4].items())}")
        elif target == "nonstationary" and not is_stationary:
            print(f"\n{column} (Nonstationary)\n")
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print(f"Critical Values: \n{'\n'.join(f'\t {k}: {str(v)}' for k, v in result[4].items())}")
        return is_stationary
    except Exception as err:
        return False

def plot_acf_pacf(df, columns, lags=40):
    """
    Plot ACF and PACF for multiple columns.

    https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
    """
    n_cols = 2
    n_rows = len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7*n_rows), squeeze=False)

    for i, column in enumerate(columns):
        plot_acf(df[column].dropna(), lags=lags, ax=axes[i, 0])
        axes[i, 0].set_title(f'ACF: {column}')
        plot_pacf(df[column].dropna(), lags=lags, ax=axes[i, 1])
        axes[i, 1].set_title(f'PACF: {column}')

    plt.tight_layout()
    plt.show()

def decompose_time_series(df, column, model='additive'):
    """Decompose time series into trend, seasonal, and residual components."""
    result = seasonal_decompose(df[column], model=model)
    result.plot()
    plt.show()

def check_for_sudden_changes(df, column, window=7):
    """Check for sudden changes or jumps in the data."""
    rolling_mean = df[column].rolling(window=window).mean()
    diff = df[column] - rolling_mean
    threshold = diff.std() * 3
    sudden_changes = df[abs(diff) > threshold]
    return sudden_changes

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for all numeric columns.

    https://ademos.people.uic.edu/Chapter22.html

    Using Spearman here because most of this data is not normal
    """
    corr = df.corr(method="spearman")
    plt.figure(figsize=(20, 16))  # Increased figure size
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.xticks(rotation=90)  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
    plt.tight_layout()  # Adjust layout to show all labels
    plt.show()

def check_data_consistency(df, column):
    """Check for data consistency issues like negative values where inappropriate."""
    if df[column].min() < 0:
        print(f"Warning: Negative values found in {column}")
        return df[df[column] < 0]
    return pd.DataFrame()