# Cardiovascular Disease Predictor

A Streamlit-based web application that predicts the risk of cardiovascular disease using a tuned XGBoost model. Built with Python, this project includes data preprocessing, model training, and evaluation, leveraging datasets like cardio_train.csv. Explore the interactive interface to assess patient risk based on health metrics. Ideal for healthcare enthusiasts and data science learners!

## Features
- Predicts cardiovascular disease risk with a tuned XGBoost model.
- Interactive Streamlit UI for user input and results.
- Handles preprocessing and feature engineering automatically.

## Setup
1. Clone the repository:
   ``n   git clone https://github.com/faizan1343/Cardiovascular-Disease-Predictor.git
   cd Cardiovascular-Disease-Predictor
   ``n2. Install dependencies:
   ``n   pip install -r requirements.txt
   ``n3. Ensure the model file files.Cardio/week4_tuned_xgboost_balanced.pkl is in place.

## Usage
1. Run the Streamlit app:
   ``n   streamlit run app/app.py
   ``n2. Enter patient details in the interface.
3. Click Predict Health Risk to see the result.

## Files
- app/app.py: Streamlit application.
- files.Cardio/week4_tuned_xgboost_balanced.pkl: Trained model.
- Notebooks/*.ipynb: Jupyter notebooks for the workflow.
- predict_cardio.py: Prediction script.
- 
equirements.txt: Dependencies.
- README.md: This file.


## Live Demo
Access the deployed app here: [Cardiovascular Disease Predictor](https://cardiovascular-disease-predictor-luwdacpohstdj8vuinmq4t.streamlit.app/)
