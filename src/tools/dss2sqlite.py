import argparse
import sqlite3
import re

from datetime import datetime
from pathlib import Path

def parse_dss_text_to_sqlite(input_file: str, output_db: str):
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                      (id INTEGER PRIMARY KEY,
                       pathname TEXT,
                       start_date TEXT,
                       end_date TEXT,
                       num_values INTEGER,
                       units TEXT,
                       data_type TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS time_series_data
                      (id INTEGER PRIMARY KEY,
                       metadata_id INTEGER,
                       date TEXT,
                       value REAL,
                       FOREIGN KEY (metadata_id) REFERENCES metadata(id))''')

    with open(input_file, 'r') as file:
        metadata = {}
        current_metadata_id = None

        for line in file:
            line = line.strip()

            if line.startswith('/'):
                # New dataset
                pathname = line
                metadata = {'pathname': pathname}
                cursor.execute("INSERT INTO metadata (pathname) VALUES (?)", (pathname,))
                current_metadata_id = cursor.lastrowid

            elif line.startswith('Start:'):
                # Parse metadata
                start_match = re.search(r'Start: (\d{2}\w{3}\d{4}).*End: (\d{2}\w{3}\d{4}).*Number: (\d+)', line)
                if start_match:
                    metadata['start_date'] = start_match.group(1)
                    metadata['end_date'] = start_match.group(2)
                    metadata['num_values'] = int(start_match.group(3))
                    cursor.execute('''UPDATE metadata
                                      SET start_date = ?, end_date = ?, num_values = ?
                                      WHERE id = ?''',
                                   (metadata['start_date'], metadata['end_date'],
                                    metadata['num_values'], current_metadata_id))

            elif line.startswith('Units:'):
                # Parse units and data type
                units_match = re.search(r'Units: (\S+)\s+Type: (\S+)', line)
                if units_match:
                    metadata['units'] = units_match.group(1)
                    metadata['data_type'] = units_match.group(2)
                    cursor.execute('''UPDATE metadata
                                      SET units = ?, data_type = ?
                                      WHERE id = ?''',
                                   (metadata['units'], metadata['data_type'], current_metadata_id))

            elif re.match(r'\d{2}\w{3}\d{4}, \d{4};\s+', line):
                # Parse data
                data_match = re.match(r'(\d{2}\w{3}\d{4}), (\d{4});\s+(.+)', line)
                if data_match:
                    date = data_match.group(1) + ' ' + data_match.group(2)
                    value = float(data_match.group(3))
                    cursor.execute('''INSERT INTO time_series_data
                                      (metadata_id, date, value)
                                      VALUES (?, ?, ?)''',
                                   (current_metadata_id, date, value))

    conn.commit()
    conn.close()
    print(f"Data has been successfully imported into {output_db}")


if __name__ == "__main__":
    # Input and output file paths
    dss_file = Path('data/dss_out')
    db_file = Path('r2_mass_balance.sqlite')

    parser = argparse.ArgumentParser(description="Parse DSS text file to SQLite database")
    parser.add_argument("--dss", type=Path, default=dss_file, help="Path to the input DSS text(!) file, export this from Hec DSS VUE")
    parser.add_argument("--db", type=Path, default=db_file, help="Path to the output SQLite database")
    args = parser.parse_args()

    parse_dss_text_to_sqlite(args.dss, args.db)
