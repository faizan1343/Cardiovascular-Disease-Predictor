import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

best_model = joblib.load('C:\Users\Faizan\OneDrive - BENNETT UNIVERSITY\blood disease\cardio_train.csv\Tuned_XGBoost_balanced.pkl')
best_threshold = 0.5
X_columns = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi', 'bp_interaction', 'age_group_Middle-aged', 'age_group_Senior', 'age_group_Young', 'bp_category_Elevated', 'bp_category_High Stage 1', 'bp_category_High Stage 2']

def predict_cardio(patient_data):
    if not all(key in patient_data for key in ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_interaction']):
        return 'Error: Missing required patient data fields.', 0.0
    df = pd.DataFrame([patient_data])
    df['bp_category'] = pd.cut(df['ap_hi'], bins=[0, 120, 130, 140, float('inf')], labels=['Normal', 'Elevated', 'High Stage 1', 'High Stage 2'])
    df['bp_interaction'] = df['ap_hi'] * df['ap_lo']
    df = pd.get_dummies(df, columns=['bp_category'], drop_first=True)
    df = df.reindex(columns=X_columns, fill_value=0)
    scaler = StandardScaler()
    numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'bp_interaction']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    probability = best_model.predict_proba(df)[:, 1]
    prediction = (probability >= best_threshold).astype(int)[0]
    return 'Cardio' if prediction == 1 else 'No Cardio', probability[0]
