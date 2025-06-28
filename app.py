import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load("KNN_heart_model.pkl")
scalar = jb.load("heart_scalar.pkl")
expected_columns = jb.load("heart_columns.pkl")

st.title("Heart Stroke Prediction By Haris ❤")
st.markdown("Provide The Following Details")

age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex",["Male", "Female"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 88, 200, 120)

cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

maxhr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Slope", ["UP", "FLAT", "DOWN"])

if st.button("Check"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingbs,
        'MaxHR': maxhr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType' + chest_pain: 1,
        'RestingECG' + resting_ecg: 1,
        'Exercise_Angina' + exercise_angina: 1,
        'ST_Slope' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])


    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scalar.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✔ Low Risk of Heart Disease")
