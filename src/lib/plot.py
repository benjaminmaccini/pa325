import sqlite3
import pandas as pl
import matplotlib.pyplot as plt
import seaborn as sns

def get_inflow_salinity(conn: sqlite3.Connection):
    # Query to fetch the required data
    query = """
    SELECT
        timestamp,
        MassBalanceBalance300 as MassBalanceBalance,
        "Upstream200.250" as Upstream,
        ("DrenIndiosPuertecitos235.250" + "DrenHuizaches285.300" + "DrenMorrillo295.300") as OtherInflow,
        "Precipitation257.250" as Precipitation,
        Salinity300 as Salinity
    FROM mass_balance
    """

    # Read the data into a pandas DataFrame
    df = pl.read_sql_query(query, conn)
    df['timestamp'] = pl.to_datetime(df['timestamp'])

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create the first y-axis
    ax1 = plt.gca()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Flow (m³/s)')
    ax1.set_ylim(-150, 150)

    # Plot lines for the left y-axis
    sns.lineplot(x='timestamp', y='MassBalanceBalance', data=df, ax=ax1, label='Mass Balance Balance')
    sns.lineplot(x='timestamp', y='Upstream', data=df, ax=ax1, label='Upstream')
    sns.lineplot(x='timestamp', y='OtherInflow', data=df, ax=ax1, label='Other Inflow')
    sns.lineplot(x='timestamp', y='Precipitation', data=df, ax=ax1, label='Precipitation')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Salinity (mg/L)')
    ax2.set_ylim(0, 2000)

    # Plot the Salinity line on the right y-axis
    sns.lineplot(x='timestamp', y='Salinity', data=df, ax=ax2, color='red', label='Salinity')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set title
    plt.title('Salinity Inflows Over Time')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def get_outflow_salinity(conn: sqlite3.Connection):
    # Query to fetch the required data
    query = """
    SELECT
        timestamp,
        MassBalanceBalance300 as MassBalanceBalance,
        "Downstream250.300" as Downstream,
        ("GDOrdaz250.230" + "UpperDemandTX250.240" + "Evaporation250.255" + "Phreatophyte250.245" + "LowerDemandTX300.275") as OtherOutflow,
        "Anzaldues300.280" as Anzalduas,
        Salinity300 as Salinity
    FROM mass_balance
    """

    # Read the data into a pandas DataFrame
    df = pl.read_sql_query(query, conn)
    df['timestamp'] = pl.to_datetime(df['timestamp'])

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create the first y-axis
    ax1 = plt.gca()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Flow (m³/s)')
    ax1.set_ylim(-150,150)

    # Plot lines for the left y-axis
    sns.lineplot(x='timestamp', y='MassBalanceBalance', data=df, ax=ax1, label='Mass Balance Balance')
    sns.lineplot(x='timestamp', y='Downstream', data=df, ax=ax1, label='Downstream')
    sns.lineplot(x='timestamp', y='OtherOutflow', data=df, ax=ax1, label='Other Outflow')
    sns.lineplot(x='timestamp', y='Anzalduas', data=df, ax=ax1, label='Anzalduas')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Salinity (mg/L)')
    ax2.set_ylim(0,2000)

    # Plot the Salinity line on the right y-axis
    sns.lineplot(x='timestamp', y='Salinity', data=df, ax=ax2, color='red', label='Salinity')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Set title
    plt.title('Salinity Outflows Over Time')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def mass_balance(conn: sqlite3.Connection):
    # Query to fetch the required data
    query = """
    SELECT
        timestamp,
        "Upstream200.250" as Upstream,
        "Downstream250.300" as Downstream,
        ("DrenIndiosPuertecitos235.250" + "DrenHuizaches285.300" + "DrenMorrillo295.300") as OtherInflow,
        ("GDOrdaz250.230" + "UpperDemandTX250.240" + "Evaporation250.255" + "Phreatophyte250.245" + "Anzaldues300.280" + "LowerDemandTX300.275") as OtherOutflow,
        "Precipitation257.250" as Precipitation,
        MassBalanceBalance300 as MassBalanceBalance
    FROM mass_balance
    """

    # Read the data into a Polars DataFrame
    df = pl.read_sql(query, conn)
    df["timestamp"] = pl.to_datetime(df["timestamp"])

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create the axis
    ax = plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Flow (m³/s)')
    ax.set_ylim(-150,150)

    # Plot lines
    for column in ['Upstream', 'Downstream', 'OtherInflow', 'OtherOutflow', 'Precipitation', 'MassBalanceBalance']:
        sns.lineplot(x='timestamp', y=column, data=df, ax=ax, label=column)

    # Set title
    plt.title('Mass Balance Components Over Time')

    # Adjust legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_two_columns(df, col1, col2, x=None, smooth=False):
    if x is None:
        x = df.index

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlabel('Date')
    ax.set_ylabel('Salinity (mg/L)')
    ax.set_ylim(0,2000)

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

def plot_salinity_mass_balance(df: pl.DataFrame):
    """
    Compute the actual salt content (not the ratio)

    Flows are CMS, salinity is mg/L, flow*salinity is in kg/s
    """
    # Compute new columns
    df["Downstream"] = df["Salinity300"] * df["Downstream250.300"]
    df["Upstream"] = df["Salinity200"] * df["Upstream200.250"]
    df["Puertecitos"] = df["SalinityDrenPuertecitos"] * df["DrenIndiosPuertecitos235.250"]
    df["Morillos"] = df["SalinityDrenMorillos"] * df["DrenMorrillo295.300"]
    df["Huizaches"] = df["SalinityDrenHuizaches"] * df["DrenHuizaches285.300"]
    df["SalinityMassBalance"] = df["Puertecitos"] + df["Morillos"] + df["Huizaches"] + df["Upstream"] + df["Downstream"]

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create the first y-axis for salinities
    ax1 = plt.gca()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Salinity (kg/s)')
    ax1.set_ylim(-100000, 100000)

    # Plot salinity lines
    sns.lineplot(x='timestamp', y='Downstream', data=df, ax=ax1, label='Salinity Out (300)')
    sns.lineplot(x='timestamp', y='Upstream', data=df, ax=ax1, label='Salinity Upstream (200)')
    sns.lineplot(x='timestamp', y='Puertecitos', data=df, ax=ax1, label='Salinity Dren Puertecitos')
    sns.lineplot(x='timestamp', y='Morillos', data=df, ax=ax1, label='Salinity Dren Morillos')
    sns.lineplot(x='timestamp', y='Huizaches', data=df, ax=ax1, label='Salinity Dren Huizaches')
    sns.lineplot(x='timestamp', y='SalinityMassBalance', data=df, ax=ax1, color='black', label='Mass Balance')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')

    # Set title
    plt.title('Salinity Mass Balance Over Time')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
