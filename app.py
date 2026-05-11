import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and features
model = pickle.load(open('noshow_model.pkl', 'rb'))

st.set_page_config(page_title="No-Show Predictor", page_icon="🏥", layout="wide")
st.title("🏥 No-Show Appointment Predictor")
st.write("Predict whether a patient will show up for their appointment")

# Sidebar inputs
st.sidebar.header("Patient Details")
age = st.sidebar.slider("Patient Age", 0, 100, 30)
waiting_days = st.sidebar.slider("Waiting Days", 0, 180, 7)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
sms = st.sidebar.selectbox("SMS Reminder Sent?", ["No", "Yes"])
day = st.sidebar.selectbox("Appointment Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
scholarship = st.sidebar.selectbox("Has Scholarship?", ["No", "Yes"])
hypertension = st.sidebar.selectbox("Has Hypertension?", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Has Diabetes?", ["No", "Yes"])
alcoholism = st.sidebar.selectbox("Has Alcoholism?", ["No", "Yes"])
handicap = st.sidebar.selectbox("Has Handicap?", ["No", "Yes"])

# Convert inputs
gender_encoded = 1 if gender == "Male" else 0
sms_received = 1 if sms == "Yes" else 0
appt_dayofweek = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"].index(day)
scholarship_v = 1 if scholarship == "Yes" else 0
hypertension_v = 1 if hypertension == "Yes" else 0
diabetes_v = 1 if diabetes == "Yes" else 0
alcoholism_v = 1 if alcoholism == "Yes" else 0
handicap_v = 1 if handicap == "Yes" else 0

# Predict
input_data = np.array([[age, scholarship_v, hypertension_v, diabetes_v,
                         alcoholism_v, handicap_v, sms_received,
                         waiting_days, gender_encoded, appt_dayofweek]])

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1] * 100

# Main panel
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("No-show Probability", f"{probability:.1f}%")

with col2:
    if probability >= 60:
        st.error("🔴 HIGH RISK")
    elif probability >= 35:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

with col3:
    st.metric("Patient Age", age)

st.markdown("---")

# Risk gauge chart
fig, ax = plt.subplots(figsize=(8, 2))
ax.barh(["Risk"], [probability], color='#E24B4A' if probability >= 60 else '#EF9F27' if probability >= 35 else '#1D9E75')
ax.barh(["Risk"], [100 - probability], left=[probability], color='#e0e0e0')
ax.set_xlim(0, 100)
ax.set_xlabel("No-show Probability %")
ax.set_title(f"Risk Level: {probability:.1f}%")
st.pyplot(fig)

st.markdown("---")
st.subheader("Key Insights from Analysis")
st.write("- 19-30 year olds have highest no-show rate (35.2%)")
st.write("- Waiting 31-60 days increases no-show risk to 34.1%")
st.write("- Monday appointments have highest no-show rate (30.2%)")
st.write("- Age and waiting days are the top 2 predictors")

#footer
st.markdown("---")
st.write("Developed by Prabhkirat Singh | BCA 6th Semester | Major Project")