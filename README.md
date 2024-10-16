# LRG/RB Analysis

This project is the backing library and notebook for graphical, regression, event and statistical analysis for 
analyzing salinity data in the Lower Rio Grande/Rio Bravo (LRG/RB) as part of UT Austin's PA325 course.

The project contains functions for transforming CSV sources from a binational coalition of agencies (CILA, CONAGUA, TCEQ, EPA, IBWC). Sources include flow rates and salinity levels (mg/L) at gauges south of the Falcon International Reservior until the Gulf of Mexico.

## Setup

This project is managed with [Rye](https://rye.astral.sh/), follow the instructions there to get up and running with all the dependencies.

Also install the following depending on your system:
- [sqlite3](https://www.sqlite.org/download.html)

> Why SQLite? Well, I like having structured data beyond a CSV, plus SQLite allows me to plug into the
[datasette](https://datasette.io/) ecosystem which allows for rapid data exploration.

## Tools

There are several tools for transforming the data from a target source to SQLite.

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


## Datasette

A variety of methods are great for cursory data exploration, I prefer datasette. Just run `rye run datasette mass_balance.sqlite` for a working example!

Example Query -- Get the max of the max for all salinity sites in the study:
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
