Cardiovascular Disease Predictor
A Streamlit-based web application that predicts the risk of cardiovascular disease using a tuned XGBoost model. Built with Python, this project includes data preprocessing, model training, and evaluation, leveraging datasets like cardio_train.csv. Explore the interactive interface to assess patient risk based on health metrics. Ideal for healthcare enthusiasts and data science learners!

Features
Predicts cardiovascular disease risk with a tuned XGBoost model.
Interactive Streamlit UI for user input and results.
Handles preprocessing and feature engineering automatically.

Setup

Clone the repository:
git clone https://github.com/faizan1343/Cardiovascular-Disease-Predictor.git
cd Cardiovascular-Disease-Predictor


Install dependencies:
pip install -r requirements.txt

Usage
Run the Streamlit app:

streamlit run app/appfiles/app.py

Enter patient details in the interface.
Click "Predict Health Risk" to see the result.

Files
app/appfiles/app.py: Streamlit application.
files.Cardio/week4_tuned_xgboost_balanced.pkl: Trained model.
Notebooks/*.ipynb: Jupyter notebooks for the workflow.
predict_cardio.py: Prediction script.
requirements.txt: Dependencies.
README.md: This file.

Notes
This README.md content reflects the current project structure and includes instructions for running the app locally or deploying it.
The live demo section (e.g., with the Streamlit URL) will be added after deployment, as outlined in the previous steps.
If you’d like to modify this content (e.g., add more sections, update descriptions), let me know!