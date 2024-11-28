# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:19:48 2024

@author: manma
"""

import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and label encoder
with open('corrosion_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Streamlit user interface
st.title("Corrosion Rate Prediction")

st.write("""
### Input Features:
- Material (e.g., Steel, Copper, Aluminum)
- Temperature (°C)
- pH level
- Flow rate (L/min)
""")

# User input
material = st.selectbox('Select Material', ['Steel', 'Copper', 'Aluminum'])
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, value=25.0)
ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0)
flow_rate = st.number_input('Flow Rate (L/min)', min_value=0.0, value=1.5)

# Prediction button
if st.button('Predict Corrosion Rate'):
    # Prepare the input data for prediction
    new_data = {
        'Material': material,
        'Temperature': temperature,
        'pH': ph,
        'FlowRate': flow_rate
    }
    
    # Convert input data into DataFrame
    df_new = pd.DataFrame([new_data])

    # Encode the 'Material' column
    df_new['Material'] = le.transform(df_new['Material'])

    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(df_new)

    # Make prediction
    predicted_corrosion_rate = model.predict(scaled_input)

    # Display the prediction result
    st.write(f"Predicted Corrosion Rate: {predicted_corrosion_rate[0]:.4f} units")
