# data-exploration

A collection of tools and analysis for the data project, focusing on Reach 2

## Setup

This project is managed with [Rye](https://rye.astral.sh/), follow the instructions there to get up and running with all the
dependencies.

Also install the following depending on your system:
- [sqlite3](https://www.sqlite.org/download.html)

> Why SQLite? Well, I like having structured data beyond a CSV, plus SQLite allows me to plug into the
[datasette](https://datasette.io/) ecosystem which allows for rapid data exploration. Just run `rye run datasette mass_balance.sqlite`
for a working example!

## Tools

- Convert OASIS output to SQLite
```
rye run oasis2sqlite --csv data/mass_balance_data.csv --db mass_balance.sqlite
```
- Convert a HEC-DSS file to SQLite
```
rye run dss2sqlite --dss=input_dss_file.dss --db=output_database.sqlite
```
- Convert a HUC precipitation file to SQLite
```
rye run huc2sqlite --csv="data/Rain_Ave_Date_2N.csv" --db="huc.sqlite"
```

## Workflows

- All analysis below can be accessed via the Jupyter server with `rye run jupyter notebook` then navigating to `Analysis.ipynb`.
- One-off queries
  - Run whatever tool to get

Example (Get the max of the max for all DSS salinity sites):
```
WITH monthly_maxes AS (
    SELECT
        m.pathname as pathname,
        substr(tsd.date, 3, 3) || '-' || substr(tsd.date, 7, 4) AS month,
        MAX(tsd.value) AS monthly_max
    FROM
        metadata m
    JOIN
        time_series_data tsd ON m.id = tsd.metadata_id
    WHERE
        pathname LIKE '%SALINITY%'
    GROUP BY
        pathname,
        month
)
SELECT
    pathname,
    month,
    MAX(monthly_max) AS max_across_pathnames
FROM
    monthly_maxes
WHERE
    month LIKE '%23%'
GROUP BY
    month
ORDER BY
    substr(month, 5, 4), -- Year
    CASE substr(month, 1, 3)
        WHEN 'JAN' THEN 1
        WHEN 'FEB' THEN 2
        WHEN 'MAR' THEN 3
        WHEN 'APR' THEN 4
        WHEN 'MAY' THEN 5
        WHEN 'JUN' THEN 6
        WHEN 'JUL' THEN 7
        WHEN 'AUG' THEN 8
        WHEN 'SEP' THEN 9
        WHEN 'OCT' THEN 10
        WHEN 'NOV' THEN 11
        WHEN 'DEC' THEN 12
    END;
```
