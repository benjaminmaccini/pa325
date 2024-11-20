import matplotlib.pyplot as plt
import seaborn as sns

from src.lib.analysis import segments_fit

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
