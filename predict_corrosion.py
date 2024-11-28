# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:51:45 2024

@author: manma
"""

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('corrosion_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Simulate or take real-time input data (for example, from a user)
# Replace this with actual real-time data gathering if necessary
new_data = {
    'Material': 'Steel',  # Can be 'Steel', 'Copper', 'Aluminum', etc.
    'Temperature': 45.0,  # Example temperature in Celsius
    'pH': 7.0,  # Example pH level
    'FlowRate': 1.5  # Example flow rate in L/min
}

# Convert the new data into a DataFrame (same format as during training)
df_new = pd.DataFrame([new_data])

# Encode the 'Material' column (using the same label encoder as before)
with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)
df_new['Material'] = le.transform(df_new['Material'])

# Scale the input data using the same scaler as before
scaled_input = scaler.transform(df_new)

# Make a prediction using the trained model
predicted_corrosion_rate = model.predict(scaled_input)

# Output the predicted corrosion rate
print(f"Predicted corrosion rate: {predicted_corrosion_rate[0]}")
