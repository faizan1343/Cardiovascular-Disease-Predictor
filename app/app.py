import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure the page
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-positive {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .result-negative {
        background: linear-gradient(90deg, #51cf66, #40c057);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    .stNumberInput > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioVascular Disease Predictor</h1>
    <h3>Advanced Cardiovascular Disease Risk Assessment</h3>
    <p>Powered by Machine Learning | Tuned XGBoost Model</p>
</div>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Faizan\OneDrive - BENNETT UNIVERSITY\Cardiovascular Disease Proj\files.Cardio\Tuned_XGBoost_balanced.pkl"
    try:
        best_model = joblib.load(model_path)
        return best_model, None
    except FileNotFoundError:
        return None, f"Model file not found at {model_path}"
    except Exception as e:
        return None, f"Error loading model: {e}"

# Initialize model
best_model, error_message = load_model()

if error_message:
    st.error(f"🚨 {error_message}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

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

# Sidebar for model information
with st.sidebar:
    st.markdown("## 🔬 Model Information")
    st.info("""
    **Model Type:** Tuned XGBoost (Balanced)
    **Threshold:** 0.5
    **Features:** 19 clinical parameters
    **Accuracy:** ~94%
    """)
    
    st.markdown("## 📊 Risk Factors")
    st.warning("""
    **High Risk Indicators:**
    - Age > 60 years
    - High blood pressure
    - High cholesterol
    - Smoking
    - Obesity (BMI > 30)
    - Physical inactivity
    """)
    
    st.markdown("## ⚠️ Disclaimer")
    st.error("""
    This tool is for educational purposes only.
    Always consult healthcare professionals
    for medical decisions.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 👤 Patient Information")
    
    # Personal Information Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📋 Basic Demographics")
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    
    with col_demo1:
        age = st.number_input("🎂 Age (years)", min_value=0, max_value=120, value=50, help="Patient's age in years")
    
    with col_demo2:
        gender = st.selectbox("👤 Gender", options=[0, 1], index=0, 
                             format_func=lambda x: "👩 Female" if x == 0 else "👨 Male",
                             help="0 = Female, 1 = Male")
    
    with col_demo3:
        height = st.number_input("📏 Height (cm)", min_value=0, max_value=250, value=165, help="Patient's height in centimeters")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Physical Measurements Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Physical Measurements")
    
    col_phys1, col_phys2 = st.columns(2)
    
    with col_phys1:
        weight = st.number_input("⚖️ Weight (kg)", min_value=0, max_value=300, value=70, help="Patient's weight in kilograms")
    
    with col_phys2:
        bmi = st.number_input("📊 BMI", min_value=0.0, max_value=100.0, value=25.7, step=0.1, help="Body Mass Index")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cardiovascular Measurements Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 💓 Cardiovascular Measurements")
    
    col_cardio1, col_cardio2 = st.columns(2)
    
    with col_cardio1:
        ap_hi = st.number_input("📈 Systolic BP (mmHg)", min_value=0, max_value=300, value=140, 
                               help="Upper blood pressure reading")
    
    with col_cardio2:
        ap_lo = st.number_input("📉 Diastolic BP (mmHg)", min_value=0, max_value=200, value=90, 
                               help="Lower blood pressure reading")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lab Results Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🧪 Laboratory Results")
    
    col_lab1, col_lab2 = st.columns(2)
    
    with col_lab1:
        cholesterol = st.selectbox("🩸 Cholesterol Level", 
                                  options=[1, 2, 3], 
                                  index=1,
                                  format_func=lambda x: f"{'🟢 Normal' if x==1 else '🟡 Above Normal' if x==2 else '🔴 Well Above Normal'}",
                                  help="1=Normal, 2=Above Normal, 3=Well Above Normal")
    
    with col_lab2:
        gluc = st.selectbox("🍯 Glucose Level", 
                           options=[1, 2, 3], 
                           index=0,
                           format_func=lambda x: f"{'🟢 Normal' if x==1 else '🟡 Above Normal' if x==2 else '🔴 Well Above Normal'}",
                           help="1=Normal, 2=Above Normal, 3=Well Above Normal")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Lifestyle Factors Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🏃‍♂️ Lifestyle Factors")
    
    col_life1, col_life2, col_life3 = st.columns(3)
    
    with col_life1:
        smoke = st.selectbox("🚬 Smoking Status", 
                            options=[0, 1], 
                            index=0,
                            format_func=lambda x: "🚭 Non-Smoker" if x == 0 else "🚬 Smoker",
                            help="0=No, 1=Yes")
    
    with col_life2:
        alco = st.selectbox("🍷 Alcohol Intake", 
                           options=[0, 1], 
                           index=0,
                           format_func=lambda x: "🚫 No Alcohol" if x == 0 else "🍷 Drinks Alcohol",
                           help="0=No, 1=Yes")
    
    with col_life3:
        active = st.selectbox("🏃‍♂️ Physical Activity", 
                             options=[0, 1], 
                             index=1,
                             format_func=lambda x: "😴 Inactive" if x == 0 else "🏃‍♂️ Active",
                             help="0=No, 1=Yes")
    
    st.markdown('</div>', unsafe_allow_html=True)

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

with col2:
    st.markdown("## 🎯 Risk Assessment")
    
    # Quick stats display
    st.markdown("### 📊 Current Readings")
    
    # BMI interpretation
    if bmi < 18.5:
        bmi_status = "🔵 Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_status = "🟢 Normal"
        bmi_color = "green"
    elif bmi < 30:
        bmi_status = "🟡 Overweight"
        bmi_color = "orange"
    else:
        bmi_status = "🔴 Obese"
        bmi_color = "red"
    
    st.metric("BMI Status", f"{bmi:.1f}", bmi_status)
    
    # Blood pressure interpretation
    if ap_hi < 120 and ap_lo < 80:
        bp_status = "🟢 Normal"
    elif ap_hi < 130 and ap_lo < 80:
        bp_status = "🟡 Elevated"
    elif ap_hi < 140 or ap_lo < 90:
        bp_status = "🟠 Stage 1 High"
    else:
        bp_status = "🔴 Stage 2 High"
    
    st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}", bp_status)
    
    # Age group
    if age < 35:
        age_group = "🟢 Young Adult"
    elif age < 60:
        age_group = "🟡 Middle-aged"
    else:
        age_group = "🟠 Senior"
    
    st.metric("Age Group", f"{age} years", age_group)
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 **ANALYZE CARDIOVASCULAR RISK**", type="primary", use_container_width=True):
        with st.spinner("🧠 AI Model analyzing patient data..."):
            prediction, probability = predict_cardio(patient_data)
            
            st.markdown("### 📋 Analysis Results")
            
            # Display prediction with styling
            if prediction == "Cardio":
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <h3>Prediction: {prediction}</h3>
                    <h4>Risk Probability: {probability:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("🚨 **IMPORTANT:** This patient may be at risk of cardiovascular disease. Please consult a healthcare professional immediately!")
                
                st.markdown("### 🏥 Recommended Actions:")
                st.markdown("""
                - 📞 Schedule immediate cardiology consultation
                - 🧪 Consider additional cardiac testing
                - 💊 Review current medications
                - 🏃‍♂️ Implement lifestyle modifications
                - 📅 Regular monitoring required
                """)
                
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ LOW RISK DETECTED</h2>
                    <h3>Prediction: {prediction}</h3>
                    <h4>Risk Probability: {probability:.1%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("🎉 **GOOD NEWS:** No immediate cardiovascular disease risk detected!")
                
                st.markdown("### 🎯 Prevention Tips:")
                st.markdown("""
                - 🥗 Maintain healthy diet
                - 🏃‍♂️ Regular physical activity
                - 📅 Annual health check-ups
                - 🚭 Avoid smoking
                - 😴 Adequate sleep
                """)
            
            # Probability bar
            st.markdown("### 📊 Risk Probability Breakdown")
            st.progress(int(probability * 100))

            
            # Risk factors analysis
            st.markdown("### 🔍 Risk Factors Analysis")
            risk_factors = []
            
            if age > 60:
                risk_factors.append("🔴 Advanced age (>60)")
            if bmi > 30:
                risk_factors.append("🔴 Obesity (BMI >30)")
            if ap_hi > 140 or ap_lo > 90:
                risk_factors.append("🔴 High blood pressure")
            if cholesterol > 2:
                risk_factors.append("🔴 High cholesterol")
            if gluc > 2:
                risk_factors.append("🔴 High glucose")
            if smoke == 1:
                risk_factors.append("🔴 Smoking")
            if active == 0:
                risk_factors.append("🟡 Physical inactivity")
            if alco == 1:
                risk_factors.append("🟡 Alcohol consumption")
            
            if risk_factors:
                st.markdown("**Present Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("🎉 No major risk factors detected!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>CardioPredict v2.0</strong> | Powered by XGBoost Machine Learning</p>
    <p>⚠️ <em>This tool is for educational and screening purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.</em></p>
    <p>🏥 For emergency medical situations, contact your local emergency services immediately.</p>
</div>
""", unsafe_allow_html=True)