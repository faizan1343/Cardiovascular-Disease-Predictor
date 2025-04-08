import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = "C:\\Users\\Faizan\\OneDrive - BENNETT UNIVERSITY\\blood disease\\cardio_train.csv\\Tuned_XGBoost_balanced.pkl"
try:
    best_model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

best_threshold = 0.5

# Match the training feature columns
X_columns = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi', 'bp_interaction',
             'age_group_Middle-aged', 'age_group_Senior', 'age_group_Young',
             'bp_category_Elevated', 'bp_category_High Stage 1', 'bp_category_High Stage 2']

def preprocess_data(patient_data):
    df = pd.DataFrame([patient_data])
    
    # Generate age groups
    age_bins = [0, 35, 60, 120]  # Adjust bins based on your training logic
    age_labels = ['Young', 'Middle-aged', 'Senior']
    df['age_group'] = pd.cut(df['age_years'], bins=age_bins, labels=age_labels, include_lowest=True)
    df = pd.get_dummies(df, columns=['age_group'], drop_first=False)  # Keep all categories
    
    # Generate bp_category
    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 130, 140, float('inf')], 
                              labels=['Normal', 'Elevated', 'High Stage 1', 'High Stage 2'])
    df['bp_interaction'] = df['ap_hi'] * df['ap_lo']
    df = pd.get_dummies(df, columns=['bp_category'], drop_first=False)  # Keep all categories
    
    # Reindex to match training features, filling missing with 0
    df = df.reindex(columns=X_columns, fill_value=0)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'bp_interaction']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def predict_cardio(patient_data):
    processed_data = preprocess_data(patient_data)
    probability = best_model.predict_proba(processed_data)[:, 1][0]
    prediction = 'Cardio' if probability >= best_threshold else 'No Cardio'
    return prediction, probability

# Streamlit UI
st.title("Cardiovascular Disease Predictor")
st.write("Enter patient details to predict the risk of cardiovascular disease.")

# Input fields
age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
height = st.number_input("Height (cm)", min_value=0, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=300, value=140)
ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=200, value=90)
cholesterol = st.selectbox("Cholesterol Level (1=Normal, 2=Above Normal, 3=Well Above Normal)", [1, 2, 3], index=1)
gluc = st.selectbox("Glucose Level (1=Normal, 2=Above Normal, 3=Well Above Normal)", [1, 2, 3], index=0)
smoke = st.selectbox("Smoker (0=No, 1=Yes)", [0, 1], index=0)
alco = st.selectbox("Alcohol Intake (0=No, 1=Yes)", [0, 1], index=0)
active = st.selectbox("Physically Active (0=No, 1=Yes)", [0, 1], index=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.7, step=0.1)
gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1], index=0)  # Add gender input

# Calculate bp_interaction on the fly
bp_interaction = ap_hi * ap_lo

# Collect input data
patient_data = {
    'age_years': age,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': gluc,
    'smoke': smoke,
    'alco': alco,
    'active': active,
    'bmi': bmi,
    'bp_interaction': bp_interaction,
    'gender': gender  # Add gender to input
}

# Prediction button
if st.button("Predict"):
    prediction, probability = predict_cardio(patient_data)
    st.write(f"**Prediction**: {prediction}")
    st.write(f"**Probability of Cardio**: {probability:.4f}")
    if prediction == "Cardio":
        st.warning("This patient may be at risk of cardiovascular disease. Consult a healthcare professional!")
    else:
        st.success("No immediate risk of cardiovascular disease detected.")

# Optional: Display model info
st.write("**Model Details**: Tuned XGBoost (Balanced) with threshold 0.5")