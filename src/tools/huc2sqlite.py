import argparse
import csv
import sqlite3
from pathlib import Path


def huc2sqlite(db_file: Path, csv_file: Path):
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
        CREATE TABLE IF NOT EXISTS huc (
            "Join_ID" INTEGER,
            "Date_Text" TEXT,
            "mean_precip" REAL,
            "Point_Count" TEXT,
            "AOI" INTEGER,
            "timestamp" TEXT GENERATED ALWAYS AS (
                substr("Date_Text", 5, 2) || '/' ||
                substr("Date_Text", 7, 2) || '/' ||
                substr("Date_Text", 1, 4)
            ) STORED
        );
        """)

        # Insert data
        for row in csv_reader:
            if row and not row[0].strip().startswith('AVG'):  # Skip summary rows
                cursor.execute(f"INSERT INTO huc VALUES ({', '.join(['?' for _ in row])})", row)

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print(f"SQLite database created: {db_file}")

# CLI entrypoint
if __name__ == "__main__":
    # Input and output file paths
    csv_file = Path('data/Rain_Ave_Date_2N.csv')
    db_file = Path('huc.sqlite')

    parser = argparse.ArgumentParser(description="Convert CSV to SQLite database.")
    parser.add_argument("--csv", type=Path, default=csv_file, help="Input CSV file path")
    parser.add_argument("--db", type=Path, default=db_file, help="Output SQLite database file path")

    args = parser.parse_args()

    huc2sqlite(db_file=args.db, csv_file=args.csv)
