import streamlit as st
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Disease Predictor", layout="centered")

# =========================
# BACKGROUND IMAGE
# =========================
def add_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .block-container {
            background-color: rgba(255,255,255,0.92);
            padding: 25px;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# =========================
# LOAD MODELS
# =========================
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("heart_model.pkl", "rb"))

# =========================
# RISK FUNCTION
# =========================
def get_risk(prob):
    if prob < 0.40:
        return "🟢 LOW RISK"
    elif prob < 0.70:
        return "🟡 MEDIUM RISK"
    else:
        return "🔴 HIGH RISK"

# =========================
# TITLE
# =========================
st.title("🩺 Multi-Disease Prediction System")

option = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

# =========================
# DIABETES SECTION
# =========================
if option == "Diabetes":

    st.subheader("🧬 Diabetes Prediction")

    preg = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

        prob = diabetes_model.predict_proba(input_data)[0][1]
        pred = diabetes_model.predict(input_data)[0]

        st.success(f"Result: {'Diabetic' if pred==1 else 'Not Diabetic'}")
        st.info(f"Probability: {prob*100:.2f}%")
        st.warning(f"Risk: {get_risk(prob)}")

# =========================
# HEART DISEASE SECTION
# =========================
else:

    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    thalach = st.number_input("Max Heart Rate (thalach)")
    oldpeak = st.number_input("Oldpeak")

    # categorical dropdowns
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
    )
    cp = {
        "typical angina": 0,
        "atypical angina": 1,
        "non-anginal pain": 2,
        "asymptomatic": 3
    }[cp]

    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

    restecg = st.selectbox(
        "Rest ECG",
        ["normal", "st-t abnormal", "lv hypertrophy"]
    )
    restecg = {
        "normal": 0,
        "st-t abnormal": 1,
        "lv hypertrophy": 2
    }[restecg]

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):

        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang, oldpeak]])

        prob = heart_model.predict_proba(input_data)[0][1]
        pred = heart_model.predict(input_data)[0]

        st.success(f"Result: {'Heart Disease' if pred==1 else 'No Disease'}")
        st.info(f"Probability: {prob*100:.2f}%")
        st.warning(f"Risk: {get_risk(prob)}")