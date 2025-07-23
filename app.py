import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('salary_model.pkl')
encoders = joblib.load('encoders.pkl')

st.set_page_config(page_title="💼 Salary Predictor", page_icon="💸", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2E8B57;'>💼 Advanced Employee Salary Predictor</h1>
    <hr style='border: 1px solid #2E8B57;'>
""", unsafe_allow_html=True)

st.sidebar.header("📋 Enter Employee Details")

# Function to get label options
def get_label_options(col):
    return list(encoders[col].classes_)

# Sidebar Inputs (Text Labels)
workclass = st.sidebar.selectbox("Workclass", get_label_options("workclass"))
education = st.sidebar.selectbox("Education", get_label_options("education"))
marital_status = st.sidebar.selectbox("Marital Status", get_label_options("marital-status"))
occupation = st.sidebar.selectbox("Occupation", get_label_options("occupation"))
relationship = st.sidebar.selectbox("Relationship", get_label_options("relationship"))
race = st.sidebar.selectbox("Race", get_label_options("race"))
sex = st.sidebar.selectbox("Sex", get_label_options("sex"))
native_country = st.sidebar.selectbox("Native Country", get_label_options("native-country"))

# Numeric Inputs in Main Area
st.subheader("🧮 Numerical Inputs")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 17, 90)
    education_num = st.slider("Education Number", 1, 16)
    capital_gain = st.number_input("Capital Gain", value=0)
with col2:
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=100000)
    hours_per_week = st.slider("Hours per Week", 1, 99)
    capital_loss = st.number_input("Capital Loss", value=0)

if st.button("🎯 Predict Salary Category"):
    # Encode all label columns
    input_dict = {
        'age': age,
        'workclass': encoders['workclass'].transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education': encoders['education'].transform([education])[0],
        'education-num': education_num,
        'marital-status': encoders['marital-status'].transform([marital_status])[0],
        'occupation': encoders['occupation'].transform([occupation])[0],
        'relationship': encoders['relationship'].transform([relationship])[0],
        'race': encoders['race'].transform([race])[0],
        'sex': encoders['sex'].transform([sex])[0],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': encoders['native-country'].transform([native_country])[0]
    }

    input_df = pd.DataFrame([input_dict])

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    label = ">50K" if pred == 1 else "<=50K"

    st.success(f"💰 Predicted Salary Category: **{label}**")
    st.info(f"📊 Confidence: **{round(proba[pred]*100, 2)}%**")

    # Chart
    st.subheader("🔍 Salary Probability")
    st.bar_chart({"<=50K": proba[0], ">50K": proba[1]})
