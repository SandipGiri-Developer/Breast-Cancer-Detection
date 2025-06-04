import streamlit as st
from joblib import load
import numpy as np

model = load('model.joblib')

st.set_page_config(page_title="Breast Cancer Risk Assessment", layout="wide")
st.markdown("""
<div style="
    background-color: #f9f9f9;
    padding: 15px 20px;
    border-left: 6px solid #2c3e50;
    border-radius: 5px;
    margin-bottom: 25px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
">
    <h3 style="
        color: #2c3e50;
        text-align: center;
        margin-top: 0;
    ">Disclaimer</h3>
    <p style="
        font-size: 15px;
        color: #333;
        text-align: center;
        line-height: 1.6;
        margin-bottom: 0;
    ">
    This tool provides a preliminary risk assessment only. It is not a substitute for professional medical advice. 
    Always consult with a qualified healthcare provider for diagnosis and treatment.
    </p>
</div>
""", unsafe_allow_html=True)


st.title("Breast Cancer Risk Prediction Tool")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tumor Characteristics (Mean Values)")
    radius_mean = st.number_input("Mean Radius (mm)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    perimeter_mean = st.number_input("Mean Perimeter (mm)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    area_mean = st.number_input("Mean Area (micro meter²)", min_value=0.0, max_value=2500.0, value=500.0, step=1.0)
    concavity_mean = st.number_input("Mean Concavity", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    concave_points_mean = st.number_input("Mean Concave Points", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

with col2:
    st.subheader("Tumor Characteristics (Worst Values)")
    radius_worst = st.number_input("Worst Radius (mm)", min_value=0.0, max_value=40.0, value=15.0, step=0.1)
    perimeter_worst = st.number_input("Worst Perimeter (mm)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
    area_worst = st.number_input("Worst Area (mm²)", min_value=0.0, max_value=5000.0, value=1000.0, step=1.0)
    concavity_worst = st.number_input("Worst Concavity", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    concave_points_worst = st.number_input("Worst Concave Points", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

if st.button("Assess Risk", type="primary"):
    input_data = np.array([[
        concave_points_worst, perimeter_worst, concavity_worst,
        concavity_mean, area_mean, radius_mean, area_worst,
        perimeter_mean, radius_worst, concave_points_mean
    ]])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display results
    st.divider()
    if prediction == 1:
        st.error(f"🚨 High Risk Detected: {probability*100:.2f}% probability of malignancy")
        st.markdown("""
        <div style="background-color:#f8d7da;color:#721c24;padding:10px;border-radius:5px;">
        <b>Recommendation:</b> Please consult with an oncologist immediately for further evaluation.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Low Risk: {probability*100:.1f}% probability of malignancy")
        st.markdown("""
        <div style="background-color:#d4edda;color:#155724;padding:10px;border-radius:5px;">
        <b>Recommendation:</b> Continue regular screenings as recommended by your physician.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;font-size:12px;color:#7f8c8d;">
    <i>This tool uses machine learning models trained on the Wisconsin Breast Cancer Dataset.</i>
</div>
""", unsafe_allow_html=True)