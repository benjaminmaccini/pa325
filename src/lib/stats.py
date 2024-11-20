import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
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
    n = len(columns)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
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

def plot_histogram(df, columns, bins=30):
    """Plot histograms of the data for multiple columns."""
    n = len(columns)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if i < n:
            axes[i].hist(df[column].dropna(), bins=bins)
            axes[i].set_title(f'Histogram: {column}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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

def get_stats(df, columns):
    """
    Print the mean, median, std dev, variance for all the columns
    """
    stats_dict = {}
    for column in columns:
        if column in df.columns:
            stats_dict[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std_dev': df[column].std(),
                'variance': df[column].var()
            }

    return pd.DataFrame(stats_dict).T

def gumbel_corr(df, col1, col2, sample_size: int = 7, size: int = 1000):
    """
    Computes the empirical Pearson and Spearman correlation between two Gumbel-distributed variables. As
    well as the Mutual Information score

    Parameters:
        sample_size: Number of days to take the peak from
        size: int, number of samples to generate.

    Returns:
    - pearson_corr: float, Pearson correlation coefficient.
    - spearman_corr: float, Spearman correlation coefficient.
    - mutual_info_score: float
    """
    # Generate samples from two independent Gumbel distributions
    loc1, scale1, gumbel1 = gumbel_fit(df, col1, sample_size=sample_size, size=size)
    loc2, scale2, gumbel2 = gumbel_fit(df, col2, sample_size=sample_size, size=size)

    # Compute Pearson correlation
    pearson_corr = np.corrcoef(gumbel1, gumbel2)[0, 1]

    # Compute Spearman correlation (rank-based, can handle non-linear relationships)
    spearman_corr = np.corrcoef(np.argsort(gumbel1), np.argsort(gumbel2))[0, 1]

    def discretize_data(data, bins=50):
        return np.digitize(data, np.histogram(data, bins=bins)[1])

    # Compute mutual information
    mutual_info = mutual_info_score(discretize_data(gumbel1), discretize_data(gumbel2))

    return pearson_corr, spearman_corr, mutual_info

def gumbel_fit(df, col, sample_size: int = 7, size: int = 1000) -> tuple:
    """
    Fit a column to the Gumbel distribution. Output the loc and scale and random variate.

    - Need at least 30 samples
    """
    weekly_max = df.resample(f"{sample_size}d").max()
    loc, scale = stats.gumbel_r.fit(weekly_max[col].dropna().astype(int))

    gumbel = stats.gumbel_r.rvs(loc=loc, scale=scale, size=size)

    return loc, scale, gumbel
