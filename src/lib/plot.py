import sqlite3
import pandas as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.lib.analysis import segments_fit

def plot_two_columns(df, col1, col2, x=None, smooth=False, y_lim=2000):
    if x is None:
        x = df.index

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Salinity (mg/L)')
    ax.set_ylim(0,y_lim)

    if smooth:
        df = df.copy()
        df[col1] = df[col1].interpolate()
        df[col2] = df[col2].interpolate()

    sns.lineplot(x='timestamp', y=col1, data=df, ax=ax, label=col1)
    sns.lineplot(x='timestamp', y=col2, data=df, ax=ax, label=col2)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'{col1} vs {col2}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_series(df, column, ylim: tuple[int, int] = (0, 2000)):
    """Plot a time series for a specific column."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column])
    plt.title(f'Time Series Plot: {column}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    ax = plt.gca()
    ax.set_ylim(*ylim)
    plt.show()

def scatter(df, x_cols, y_cols, grid_width=5, title="Scatter Plot Matrix", plot_piecewise=False, piecewise_count=8):
    num_plots = len(x_cols) * len(y_cols)
    grid_height = (num_plots - 1) // grid_width + 1

    fig, axes = plt.subplots(grid_height, grid_width, figsize=(4*grid_width, 4*grid_height))
    fig.suptitle(title, fontsize=16)

    for i, (x_col, y_col) in enumerate([(x, y) for x in x_cols for y in y_cols]):
        row = i // grid_width
        col = i % grid_width
        ax = axes[row, col] if grid_height > 1 else axes[col]

        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        # Call segments_fit function
        if plot_piecewise:
            px, py = segments_fit(df[x_col], df[y_col], count=piecewise_count)
            ax.plot(px, py, color='red', linestyle='--', linewidth=2)

    # Remove any unused subplots
    for i in range(num_plots, grid_width * grid_height):
        row = i // grid_width
        col = i % grid_width
        fig.delaxes(axes[row, col] if grid_height > 1 else axes[col])

    plt.tight_layout()
    plt.show()
