import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
import os
from datetime import datetime

# ----------------- Load Model & Scaler -----------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Early Alzheimer's Prediction", layout="centered")
st.title("🧠 Early Alzheimer's Disease Prediction")
st.markdown("Fill the details to predict Alzheimer's risk.")

# ----------------- Patient Name -----------------
patient_name = st.text_input("Patient Name", placeholder="Enter patient's name")

# ----------------- Input Fields -----------------
age = st.number_input("Age", min_value=40, max_value=100, value=65)
education = st.number_input("Education (Years)", min_value=0, max_value=30, value=12)
ses = st.number_input("SES (Socioeconomic Status)", min_value=1, max_value=5, value=2)
mmse = st.number_input("MMSE Score", min_value=0.0, max_value=30.0, value=27.0)
cdr = st.number_input("CDR Score", min_value=0.0, max_value=3.0, value=0.0)
etiv = st.number_input("eTIV (Brain Volume)", min_value=1100, max_value=2000, value=1500)
nwbv = st.number_input("nWBV (Normalized Volume)", min_value=0.5, max_value=1.0, value=0.70)
asf = st.number_input("ASF (Atlas Scaling Factor)", min_value=0.8, max_value=1.5, value=1.00)
delay = st.number_input("Delay (Months)", min_value=0, max_value=100, value=0)
gender = st.radio("Gender", ("M", "F"))

# ----------------- Prepare Input Data -----------------
input_data = np.array([[age, education, ses, mmse, cdr, etiv, nwbv, asf, delay, 1 if gender == "M" else 0]])

# ----------------- Prediction Button -----------------
if st.button("🔍 Predict"):
    if patient_name.strip() == "":
        st.warning("⚠️ Please enter the patient's name!")
    else:
        # Scale the input data before prediction
        input_scaled = scaler.transform(input_data)

        # Predict class & probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Display Prediction Result
        result_text = "AT RISK of Alzheimer's disease" if prediction == 1 else "NOT AT RISK of Alzheimer's disease"
        st.success(f"**{patient_name}** is **{result_text}**")
        st.write(f"**Demented Probability:** {probability[1]*100:.2f}%")
        st.write(f"**Non-Demented Probability:** {probability[0]*100:.2f}%")

        # ----------------- Generate Professional PDF Report -----------------
        pdf = FPDF()
        pdf.add_page()

        # College Logo + Project Title
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=30)
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(0, 15, " Early Alzheimer's Disease Prediction Report", ln=True, align="C")
        pdf.ln(10)

        # Subtitle
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, "This report is generated using a Machine Learning-based prediction model "
                             "developed for early Alzheimer's risk detection.", align="C")
        pdf.ln(10)

        # Patient Details Table
        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(0, 10, "Patient Details", ln=True)
        pdf.set_font("Arial", size=12)

        pdf.cell(90, 8, "Patient Name:", 1)
        pdf.cell(90, 8, str(patient_name), 1, ln=True)

        pdf.cell(90, 8, "Age:", 1)
        pdf.cell(90, 8, str(age), 1, ln=True)

        pdf.cell(90, 8, "Gender:", 1)
        pdf.cell(90, 8, gender, 1, ln=True)

        pdf.cell(90, 8, "MMSE Score:", 1)
        pdf.cell(90, 8, str(mmse), 1, ln=True)

        pdf.cell(90, 8, "CDR Score:", 1)
        pdf.cell(90, 8, str(cdr), 1, ln=True)

        pdf.cell(90, 8, "Education (Years):", 1)
        pdf.cell(90, 8, str(education), 1, ln=True)

        # Prediction Results Section
        pdf.ln(10)
        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(0, 10, "Prediction Results", ln=True)

        pdf.set_font("Arial", size=12)
        if prediction == 1:
            pdf.set_text_color(255, 0, 0)  # Red for AT RISK
            pdf.cell(0, 10, f"Result: AT RISK of Alzheimer's Disease", ln=True)
        else:
            pdf.set_text_color(0, 128, 0)  # Green for SAFE
            pdf.cell(0, 10, f"Result: NOT AT RISK of Alzheimer's Disease", ln=True)

        # Reset text color for probabilities
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Demented Probability: {probability[1]*100:.2f}%", ln=True)
        pdf.cell(0, 8, f"Non-Demented Probability: {probability[0]*100:.2f}%", ln=True)

        # ----------------- Authorized By + Signature -----------------
        pdf.ln(15)  # Add spacing before signature
        pdf.set_font("Arial", style="B", size=12)
        pdf.set_x(150)  # Move to right side
        pdf.cell(40, 10, "Authorized By", 0, ln=True, align="C")

        # Add Signature Below Text
        signature_path = "signature.jpg"
        if os.path.exists(signature_path):
            pdf.set_x(150)
            pdf.image(signature_path, x=150, w=40)

        # ----------------- Date of Issue (Blue & Bold) -----------------
        pdf.set_font("Arial", style="B", size=12)
        pdf.set_text_color(0, 0, 255)  # Blue color
        pdf.set_x(10)  # Keep on the same page, bottom-left
        pdf.cell(0, 10, f"Date of Issue: {datetime.now().strftime('%d-%m-%Y %I:%M %p')}", ln=True)

        # Reset text color for footer back to black
        pdf.set_text_color(0, 0, 0)

        # Footer Section
        pdf.set_font("Arial", style="I", size=10)
        pdf.cell(0, 10, "This report is generated automatically using an ML-based Alzheimer's prediction system.", ln=True, align="C")

        # Save PDF
        pdf_filename = f"{patient_name}_Alzheimers_Report.pdf"
        pdf.output(pdf_filename)

        # ----------------- Download PDF Button -----------------
        with open(pdf_filename, "rb") as file:
            st.download_button(
                label="📄 Download PDF Report",
                data=file,
                file_name=pdf_filename,
                mime="application/pdf"
            )
