# 💼 Employee Salary Predictor - Streamlit App

This project is a machine learning web application built using **Streamlit** that predicts whether a person's income exceeds $50K/year based on demographic data. The model is trained on the **UCI Adult Dataset**.

---

## 🚀 Features

- 📊 Predict salary category (≤50K or >50K) using user inputs
- 🧠 Built with a trained machine learning model (Random Forest, etc.)
- 💻 Easy-to-use Streamlit interface
- 🔐 Includes preprocessing with encoders
- 🗃️ Based on real-world data (UCI Adult dataset)

---

## 📁 Project Structure

employee-salary-predictor/
│
├── app.py # Streamlit app code
├── model_building.py # Model training and preprocessing script
├── adult.csv # Dataset
├── adult.names # Dataset feature info
├── salary_model.pkl # Trained ML model
├── encoders.pkl # Saved encoders
├── requirements.txt # Python dependencies
└── README.md # This file
