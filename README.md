# Diabetes Risk Prediction

Diabetes is a long-term metabolic condition in which the body has difficulty regulating blood glucose levels. If it is not identified early, it can lead to complications that affect the heart, kidneys, nerves, and eyes. A basic prediction model can be trained from historical patient records to estimate whether a person is likely to be at higher or lower risk based on measurements such as glucose level, BMI, age, blood pressure, and family history.


## Dataset

The notebook uses the Pima Indians Diabetes dataset. It contains diagnostic measurements for female patients and a binary target column called `Outcome`.

Features used:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

Target:
- `Outcome`
  - `1` = diabetes
  - `0` = no diabetes

Source:
- Original dataset: UCI Machine Learning Repository
- Common hosted version: `uciml/pima-indians-diabetes-database`


## Why This Method

Logistic Regression was chosen because it is one of the clearest starting points for binary classification:

- it is simple to train and explain
- it works well as a baseline model
- its coefficients are easy to inspect

## Methodology

The notebook applies a straightforward workflow:

- check class balance, missing values, and medically invalid zero values
- visualize the glucose distribution and the feature correlation matrix
- replace invalid zero values in selected columns with `NaN`
- fill those missing values with the median of each column
- split the data into training and test sets
- standardize the feature values with `StandardScaler`
- train a `LogisticRegression` model
- evaluate the model with accuracy, confusion matrix, classification report, and a confusion matrix heatmap
- inspect feature coefficients to understand how each variable affects the prediction
- allow a single custom input to be passed into a prediction helper

## What The Output Means

The model returns:

- a probability for class `1`
- a simple label: `High Risk` or `Low Risk`

This output is only a statistical estimate based on patterns in the dataset. It is useful for learning how classification works, but it should not be treated as clinical advice or a real diagnostic tool.

## Current Result

With the current notebook setup, the Logistic Regression baseline reaches an accuracy of about `75.32%` on the test split.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install notebook ipykernel
python3 -m ipykernel install --user --name=.venv
jupyter notebook
```

Open `diabetes_risk_prediction.ipynb` and run the cells top to bottom.
