# Streamlit app to load best_model.joblib and serve predictions
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Visit with Us — Wellness Predictor", layout="centered")
st.title("Wellness Tourism Purchase Predictor")

@st.cache_resource
def load_model(path="best_model.joblib"):
    return joblib.load(path)

try:
    model = load_model("best_model.joblib")
except Exception as e:
    st.error("Model not found. Place best_model.joblib in the repo root or upload it to the Space.")
    st.stop()

st.sidebar.header("Customer features")
Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.sidebar.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
CityTier = st.sidebar.selectbox("CityTier", ["1", "2", "3"])
Occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.sidebar.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.sidebar.number_input("PreferredPropertyStar", min_value=1, max_value=7, value=3)
MaritalStatus = st.sidebar.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
NumberOfTrips = st.sidebar.number_input("NumberOfTrips", min_value=0, max_value=50, value=2)
Passport = st.sidebar.selectbox("Passport", [0,1])
OwnCar = st.sidebar.selectbox("OwnCar", [0,1])
NumberOfChildrenVisiting = st.sidebar.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
Designation = st.sidebar.text_input("Designation", value="Executive")
MonthlyIncome = st.sidebar.number_input("MonthlyIncome", min_value=0, value=50000)
PitchSatisfactionScore = st.sidebar.number_input("PitchSatisfactionScore", min_value=0, max_value=10, value=7)
ProductPitched = st.sidebar.text_input("ProductPitched", value="Wellness")
NumberOfFollowups = st.sidebar.number_input("NumberOfFollowups", min_value=0, value=1)
DurationOfPitch = st.sidebar.number_input("DurationOfPitch", min_value=0, value=10)

input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

st.write("Input preview")
st.dataframe(input_df.T)

if st.button("Predict purchase probability"):
    proba = model.predict_proba(input_df)[:,1][0]
    st.metric("Purchase probability", f"{proba:.2%}")
    if proba > 0.5:
        st.success("High likelihood to purchase the Wellness package.")
    else:
        st.info("Low likelihood to purchase the Wellness package.")
