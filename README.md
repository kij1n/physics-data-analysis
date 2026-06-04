# Coupled pendulums

## Introduction

The experiment investigates resonance and energy transfer in coupled oscillating systems (Barton's Pendulums). The project covers the analysis of the raw data files extracted from measurements.
The analysis for each session is inside its appropriate notebook. The missing session files are just an indication that no measurements were taken, hence there was no data to analyze.
The measurements were taken as videos of the setup and analyzed frame-by-frame in [Tracker](https://opensourcephysics.github.io/tracker-website/) and exported as `csv`.
It's also important to note that many of the values might've changed as during the analysis we have reached many conclusions/breakthroughs. All is reflected in the final report.

## Project structure

- `data/` - Raw data files from each measurement session.
- `figures/` - All the figures shown in the notebooks exported as `svg`. This directory is excluded from git. The files, as well as the directory, are generated after running the notebooks.
- `sessions/` - The notebook files for each session.
- `src/utils.py` - A utility module with classes for storing and analyzing the data.

**Note:** There are notebooks with `-single` annotation. This means it covers calculations of the damping coefficient for each pendulum separately.
**Note:** The notebook `sessions/salvaging-project.ipynb` contains FFT analysis of the project.

## Data formatting

Each file consists of 5 columns: `t`, `Mass A`, `Mass B`, `Mass C`, `Mass D`. The lengths of the pendulums are the same for every measurement, except where noted below. Where `Mass A` is the closest to the driver and `Mass D` is the farthest.
**Note:** The unit of `t` is seconds, the remaining columns are dimensionless (the calculations use normalized values, so units are irrelevant).

### Lengths

|Name|Length|
|---|---|
|`Mass A`|97 cm|
|`Mass B`|99 cm|
|`Mass C`|99.5 cm|
|`Mass D`|100 cm|

### Exceptions

The only exception is `data/session-2/data2.csv` file from the second session. It's the only one that has driver included in the data. The column names are the same, with the difference in the order: `Mass D`, `Mass B`, `Mass A`, `Mass C` in the physical setup.

#### Values

|Name|Length|
|---|---|
|`Mass D` (driver)|50 cm|
|`Mass B`|65 cm|
|`Mass A`|50 cm|
|`Mass C`|40 cm|

## Installation

The project is using `matplotlib`, `scipy`, `numpy`, `pandas` and `jupyterlab` libraries. It's recommended to create a virtual environment and run the code with it.

The easiest way to install is to use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

This will create a virtual environment and automatically install the dependencies.

Should you prefer not to install with uv, it's possible to do it manually.

### Manual installation

#### Create virtual environment

##### Linux

```bash
python3.13 -m venv .venv 
```

and enter the virtual environment

```bash
source .venv/bin/activate
```

##### Windows

```cmd
py -3.13 -m venv .venv
```

enter the venv (cmd)

```cmd
.venv\Scripts\activate.bat
```

or with powershell

```powershell
.venv\Scripts\activate.ps1
```

**Note:** Powershell might block you with its strict execution policy. If this happens, open powershell as administrator and type:

```powershell
Set-ExecutionPolicy Unrestricted -Scope CurrentUser
```

#### Install the dependencies

And finally install the dependencies:

```bash
python -m pip install -e .
```

## Running the code

The recommended way is to use Visual Studio Code's [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) which will enable you to run the code and see the results from inside the editor.

However, if you don't want to install the extension, it's possible to use JupyterLab's web interface.

### Web interface

**Note:** This section is only relevant for you if you're not using VS Code's Jupyter extension.

#### Starting the server

##### If using uv

```bash
uv run jupyter lab
```

##### If created manually

**Note:** make sure you are inside virtual environment, otherwise the command will crash

```bash
jupyter lab
```

#### Accessing it

The web interface should open automatically in your default browser, from there you can access all the files and run the notebooks.
If not, look in the output of the command you used to start the server, the url should be written there.
