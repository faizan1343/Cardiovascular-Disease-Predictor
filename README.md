# Cardiovascular-Disease-Predictor
Cardio Predictor: Streamlit app with tuned XGBoost for heart risk. Includes preprocessing &amp; training. Great for healthcare &amp; data learners!

## Features
- Predicts cardiovascular disease risk with a tuned XGBoost model.
- Interactive Streamlit UI for user input and results.
- Handles preprocessing and feature engineering automatically.

2. Install dependencies:
3. Ensure the model file `Tuned_XGBoost_balanced.pkl` is in the root directory (download from [link] if not included).

## Usage
1. Run the Streamlit app:

2. Enter patient details (age, height, weight, blood pressure, etc.) in the interface.
3. Click "Predict Health Risk" to see the result (e.g., "No Cardio" with probability).

## Files
- `app.py`: Streamlit application with prediction logic.
- `Tuned_XGBoost_balanced.pkl`: Trained XGBoost model (if <100MB; otherwise, download externally).
- `requirements.txt`: List of Python dependencies.
- `Section1.ipynb`, etc.: Jupyter notebooks for data processing and modeling.
- `README.md`: This file.

## Dataset
- Source: `cardio_train.csv` (not included; available from Kaggle or similar).
- Cleaned version: Use `cleaned_cardio_train.csv` if provided.

## Contributing
Feel free to fork and submit pull requests. Report issues via the Issues tab.
