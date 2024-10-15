import argparse
import csv
import sqlite3
from pathlib import Path


def oasis2sqlite(db_file: Path, csv_file: Path):
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Read CSV file
    with csv_file.open('r', newline='') as f:
        csv_reader = csv.reader(f)

        # Read the first three rows
        header_parts = [next(csv_reader) for _ in range(3)]

        # Combine header parts, strip whitespace, and replace spaces with underscores
        headers = [f"{a.strip()}_{b.strip()}_{c.strip()}".replace(' ', '_')
                for a, b, c in zip(*header_parts)]

        cursor.execute(f"""
        CREATE TABLE mass_balance (
            timestamp TEXT,
            "Upstream200.250" REAL,
            "DrenIndiosPuertecitos235.250" REAL,
            "Precipitation257.250" REAL,
            "DrenHuizaches285.300" REAL,
            "DrenMorrillo295.300" REAL,
            "GDOrdaz250.230" REAL,
            "UpperDemandTX250.240" REAL,
            "Evaporation250.255" REAL,
            "Phreatophyte250.245" REAL,
            "Downstream250.300" REAL,
            "Anzaldues300.280" REAL,
            "LowerDemandTX300.275" REAL,
            "MassBalance299.300" REAL,
            "MassBalanceOut300.298" REAL,
            "MassBalanceBalance300" REAL,
            "Salinity300" REAL,
            "Salinity200" REAL,
            "SalinityDrenPuertecitos" REAL,
            "SalinityDrenMorillos" REAL,
            "SalinityDrenHuizaches" REAL,
            OtherInflow REAL GENERATED ALWAYS AS (
                "DrenIndiosPuertecitos235.250" + "DrenHuizaches285.300" + "DrenMorrillo295.300"
            ) VIRTUAL,
            OtherOutflow REAL GENERATED ALWAYS AS (
                "Phreatophyte250.245" + "UpperDemandTX250.240" + "LowerDemandTX300.275" + "Evaporation250.255"
            ) VIRTUAL,
            FlowRetentionRate REAL GENERATED ALWAYS AS (
                (CAST("Downstream250.300" AS REAL) + CAST("Upstream200.250" AS REAL)) /
                CAST("Upstream200.250" AS REAL) * 100
            ) VIRTUAL,
            MassBalanceError REAL GENERATED ALWAYS AS (
                "MassBalance299.300" - "MassBalanceOut300.298" - "MassBalanceBalance300"
            ) VIRTUAL
        );
        """)

        # Insert data
        for row in csv_reader:
            if row and not row[0].strip().startswith('AVG'):  # Skip summary rows
                cursor.execute(f"INSERT INTO mass_balance VALUES ({', '.join(['?' for _ in row])})", row)

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print(f"SQLite database created: {db_file}")

# CLI entrypoint
if __name__ == "__main__":
    # Input and output file paths
    csv_file = Path('data/mass_balance_data.csv')
    db_file = Path('mass_balance.sqlite')

    parser = argparse.ArgumentParser(description="Convert CSV to SQLite database.")
    parser.add_argument("--csv", type=Path, default=csv_file, help="Input CSV file path")
    parser.add_argument("--db", type=Path, default=db_file, help="Output SQLite database file path")

    args = parser.parse_args()

    oasis2sqlite(db_file=args.db, csv_file=args.csv)
