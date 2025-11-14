import streamlit as st
import joblib
import numpy as np

# Load model ans scaler

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Diabetes Prediction App')

# User Inputs
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=0, max_value=100, value=30)

# Predict Button
if st.button('Predict'):
    # Prepare Input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probablity = model.predit_proba(input_scaled)[0][1]

# Output

if prediction == 1:
    st.error(f'High risk of diabetes (Probablity:{probablity:.4f})')
else:
    st.succes(f'Low risk of diabetes (Probablity: {probablity:.2f})')


