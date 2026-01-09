import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load('heart_disease_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_assets()

# INPUT SECTION
st.title("❤️ Heart Disease Prediction App")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 54)
    sex = st.selectbox("Sex", [(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
    cp = st.selectbox("Chest Pain Type", [(1, "Typical"), (2, "Atypical"), (3, "Non-anginal"), (4, "Asymptomatic")], index=0)[0]
    bp = st.number_input("Resting BP", 50, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", [(1, "True"), (0, "False")], index=1)[0]
    restecg = st.selectbox("Resting EKG", [(0, "Normal"), (1, "Abnormality"), (2, "Hypertrophy")], index=0)[0]
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [(1, "Yes"), (0, "No")], index=1)[0]
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, 0.1)
    
st.write("---")
slope = st.selectbox("ST Slope", [(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], index=0)[0]
ca = st.selectbox("Major Vessels", [0, 1, 2, 3], index=0)
thal = st.selectbox("Thallium", [(3, "Normal"), (6, "Fixed"), (7, "Reversible")], index=0)[0]

if st.button("Predict"):
    # 1. Manual Scaling (Using original dataset stats to fix the broken scaler.pkl)
    # Stats from the Heart_Disease_Prediction.csv:
    # Age: mean=54.4, std=9.1 | BP: mean=131.3, std=17.8 | Chol: mean=249.6, std=51.6 
    # MaxHR: mean=149.6, std=23.1 | STDep: mean=1.05, std=1.14
    
    s_age = (age - 54.43) / 9.10
    s_bp = (bp - 131.34) / 17.86
    s_chol = (chol - 249.65) / 51.68
    s_hr = (max_hr - 149.60) / 23.16
    s_peak = (oldpeak - 1.05) / 1.14

    # 2. Build the exact feature vector
    data = {
        'Age': s_age, 'Sex': sex, 'BP': s_bp, 'Cholesterol': s_chol, 
        'FBS over 120': fbs, 'Max HR': s_hr, 'Exercise angina': exang, 'ST depression': s_peak,
        'Chest pain type_2': 1 if cp == 2 else 0,
        'Chest pain type_3': 1 if cp == 3 else 0,
        'Chest pain type_4': 1 if cp == 4 else 0,
        'EKG results_1': 1 if restecg == 1 else 0,
        'EKG results_2': 1 if restecg == 2 else 0,
        'Slope of ST_2': 1 if slope == 2 else 0,
        'Slope of ST_3': 1 if slope == 3 else 0,
        'Thallium_6': 1 if thal == 6 else 0,
        'Thallium_7': 1 if thal == 7 else 0,
        'Number of vessels fluro_1': 1 if ca == 1 else 0,
        'Number of vessels fluro_2': 1 if ca == 2 else 0,
        'Number of vessels fluro_3': 1 if ca == 3 else 0
    }
    
    df_final = pd.DataFrame([data])[model_columns]
    
    # 3. Predict
    prediction = model.predict(df_final)[0]
    prob = model.predict_proba(df_final)[0][1]

    if prediction == 1:
        st.error(f"### Result: Heart Disease Present")
    else:
        st.success(f"### Result: Heart Disease Absent")
    
    st.write(f"**Probability Score:** {prob:.2%}")