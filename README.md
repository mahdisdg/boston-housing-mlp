# Housing Data Analysis & Regression Modeling
This project performs end-to-end data analysis and predictive modeling on the HousingData.csv dataset.
All analysis, preprocessing steps, and visualizations are contained inside the notebook.

## Project Summary

The notebook walks through a full machine-learning pipeline:

### 1. Load & Inspect Data

Load HousingData.csv

Display dataset dimensions, column names, and first rows

Check for missing values

### 2. Exploratory Data Analysis (EDA)

Summary statistics

Visualizing distributions

Correlation heatmap

Scatterplots of key features (e.g., RM, LSTAT) vs. target (MEDV)

### 3. Data Cleaning & Preprocessing

Handle missing data using SimpleImputer

Remove or adjust problematic rows

Train/Test split (80% / 20%)

Feature scaling

### 4. Feature Selection for 2D Regression Surface

The two most influential features (RM and LSTAT) are extracted and used to generate:

2D scatterplots

A predicted regression surface plot across a (RM, LSTAT) grid

Contour visualization showing how predicted MEDV changes with features

### 5. Machine Learning Model — MLP Regressor

A Neural Network Regression Model (MLPRegressor) is trained:

hidden_layer_sizes=(64, 8, 4)

activation='relu'

learning_rate_init=0.001

max_iter=1000

Then:

Model is fitted

Predictions are generated on a dense grid

Results are plotted as surface + contour maps

### 6. Visualization

The notebook generates multiple figures:

Feature scatter plots

Heatmaps

## Technologies Used

Python 3

Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

MLPRegressor (Neural Network)
Regression surface (3D)

Contour plots of model predictions
