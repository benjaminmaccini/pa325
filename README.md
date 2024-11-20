# LRG/RB Analysis

This project is the backing library and notebook for graphical, regression, event and statistical analysis for
analyzing salinity data in the Lower Rio Grande/Rio Bravo (LRG/RB) as part of UT Austin's PA325 course.

The project contains functions for transforming CSV sources from a binational coalition of agencies (CILA, CONAGUA, TCEQ, EPA, IBWC).
Sources include flow rates and salinity levels at gauges south of the Falcon International Reservior until the Gulf of Mexico.

All supplemental documentation can be found in `pa325/docs/`.

## Setup

This project is managed with [Rye](https://rye.astral.sh/), follow the instructions there to get up and running with all the dependencies.

## Workflows

- OASIS data
  - Take the `all_data.1v` file and put in OASIS under `Tables > Simulation`.
  - Then from OASIS select `Output > TABLES` for the latest run and `all_data.1v`, then view the ouput.
  - Save the file and convert to `.csv` in the `pa325/data/all_data.csv` directory.
- HUC data
  - Data is available in the Box and should be downloaded, without modification, to `pa325/data/`.
- All analysis below can be accessed via the Jupyter server with `rye run jupyter notebook` then navigating
to any of the notebook files (`.ipynb`). Update the notebook parameters for the data file names.
