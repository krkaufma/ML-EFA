# Machine learning for entropy forming ability (EFA)

A lightweight training and evaluation framework for reliable determination of synthesizability (EFA) of high entropy materials.

See the associated publication for more information about EFA: https://www.nature.com/articles/s41524-020-0317-6

## Setup

This project targets a CPU-enabled Linux workstation. Additional work may be required for testing on other operating systems.

We target Python 3.6+. 

1. Create/Activate a virtual environment (via [anaconda](https://docs.conda.io/en/latest/miniconda.html), [virtualenv](https://virtualenv.pypa.io/en/latest/), or [pyenv](https://github.com/pyenv/pyenv)) Recommended: Anaconda 
2. `pip install -e .`

## How to Run Experiments

## Train a model

This project includes a `cli.py` script for training a Random Forest Regressor with the following assumptions:

- All of your data is in an excel document.
- Your labels are the last column of the data file.
- The path to the data file is passed in as an argument.
- The output file names are currently generic plus a time stamp, but can easily be modified to contain more experimental details.

The `cli.py` script returns several artifacts including the fit model, a table of feature importances, the hyperparamter search results and optimal hyperparameters found, and a transform mask (if feature selection is performed).

## Predicting EFA for new materials

This project includes a `predict.py` script for predicting the EFA of new data. The following assumptions are made:

- All of your data is in an excel document.
- The last column of the data file contains a header, but at least one of the cells in the column is blank. Celss with values will cause the material to be skipped.
- The path to the data file is passed in as an argument.
-The path to the transform mask (if applicable) and the fit model must be updated within the script.
- The output file names are currently generic plus a time stamp, but can easily be modified to contain more experimental details.

The `predict.py` script returns a .csv file containing the material name and the predicted EFA.


## Visualizing the decision trees

This project includes a `predict.py` script for exporting all of the decision trees in a random forest as png files. 

This script assumes:

- The path to the model is updated 
```
model_path = '../model_checkpoints/model_checkpoint_wCalphad_gs_2019-09-17-02-09.joblib'
```
- The data file is passed in as an argument.
- The transform mask (if applicable) is updated.
```
transform_mask_init = pd.read_csv('../transform_mask/FILENAME_HERE.csv')
```

## Data provided in this repo:

- The 56 high entropy carbides and their EFA values from density functional theory (DFT). 
- The 70 new Cr-containing carbides
- Pre-fit models using the 56 data points with DFT-based labels.
- Definitions of the predictor variable names `Predictor-Variable-Definitions.pdf`

