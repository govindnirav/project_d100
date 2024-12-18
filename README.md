# Final Project for Fundamentals of Data Science (D100) and Research Computing (D400):
Predicting Number of Bikes Rented in Washington D.C. (USA)

**Author:** *3381A*

The `project_d100` package uses GLM and LGBM models to predict the number of bikes rented in a bike-sharing rental process.

This project/package is created for the joint assessment for the D100 and D400 modules, from the MPhil in Economics and Data Science at the University of Cambridge.

** All outputs (data, model, visualisations) are provided in the .zip file. All outputs are also prerendered in`scripts/eda_cleaning.ipynb`.

## Report

The report can be found here: `project_d100/report.pdf`

## Installation

Run the following commands to install the package. Ensure you are in the `project_d100` (parent) directory.
```
conda env create -f environment.yml
```
```
conda activate final_d100
```
```
pip install -e .
```

This installation requires conda and pip to be pre-installed.

To use pre-commit hooks:
```
pre-commit install
```

## Run the package

The package can be run from the command line interface.

To load and clean the raw data, as well as produce exploratory and explanatory visualisations:
```
jupyter nbconvert --to notebook --execute scripts/eda_cleaning.ipynb --inplace
```

To load the clean data, split it, train the GLM and LGBM models, tune the respective hyperparameters, and save the tuned models:
```
python scripts/model_training.py
```

To obtain evaluation metrics for both the tuned models:
```
python scripts/evaluation.py
```

To obtain visualisations used in the report:
```
python scripts/visualisation.py
```

To run the test:
```
pytest tests/test_standardscaler.py
```

## Outputs

### Package:
Source files from package: `project_d100/project_d100`

### Scripts:
EDA and Cleaning: `project_d100/scripts/eda_cleaning.ipynb`

Model Training: `project_d100/scripts/model_training.py`

Model Evaluation: `project_d100/scripts/evaluation.py`


### Data:
Raw (along with two other files (`Readme.txt` and `additional_info.md`) with more details about the raw data): `project_d100/data/raw`

Cleaned (after EDA and cleaning): `project_d100/data/cleaned`

Test and train sets (split of clean data into train and test datasets, before preprocessing): `project_d100/data/processed`

Cleaned data is stored as a .parquet file.
Test and train sets are stored as .pkl files.

### Models:
Tuned model pipelines for GLM and LGBM (after training) to be used to predict, or for evaluation: `project_d100/models`

Tuned model pipelines are stored as .pkl files.


### Visualisations:
Visualisations to be used in the report from exploratory and explanatory data analysis, and evaluation: `project_d100/visualisation`
