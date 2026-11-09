# Wine Quality Prediction System

## Problem
A small wine-production business needs a consistent preliminary way to screen red-wine samples using recorded physicochemical measurements.

## Dataset
This prototype uses the UCI Machine Learning Repository Red Wine Quality CSV:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

## Prediction rule
- Quality score greater than or equal to 6: GOOD QUALITY (class 1)
- Quality score below 6: NEEDS IMPROVEMENT (class 0)

The original `quality` column is used only to create the target. It is removed from the model input features to prevent target leakage.

## Technologies
Python, pandas, scikit-learn, Random Forest, joblib, Streamlit, and GitHub.

## Project files
- `train_model.py`: downloads and inspects the dataset, preprocesses it, trains and evaluates the model, and saves `wine_model.pkl`.
- `app.py`: Streamlit interface for dataset preview, CSV upload, prediction, and result download.
- `requirements.txt`: required Python packages.
- `wine_quality.csv`: dataset file.
- `wine_model.pkl`: trained model.

## Installation and running
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Train the model and generate the CSV and model files:
   `python train_model.py`
5. Start the application:
   `streamlit run app.py`
6. Open the local URL printed in the terminal, normally `http://localhost:8501`.
7. In the app, download the sample input CSV, upload it, review predictions, and download the results.

## Limitations
The model is trained on a particular public red-wine dataset. It may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening results and are not substitutes for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
