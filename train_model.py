import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Simulated dataset with correct column names
data = {
    'Material': ['Steel', 'Copper', 'Aluminum', 'Steel', 'Copper', 'Aluminum'],
    'Temperature': [25.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    'pH': [7.5, 6.8, 7.0, 8.0, 5.5, 6.0],
    'FlowRate': [1.5, 2.0, 1.0, 3.0, 1.2, 2.5],
    'CorrosionRate': [0.5, 0.3, 0.2, 0.7, 0.4, 0.6]
}

# Creating the dataframe
df = pd.DataFrame(data)

# Initialize LabelEncoder for 'Material' column
le = LabelEncoder()

# Encoding 'Material' column
df['Material'] = le.fit_transform(df['Material'])

# Save the label encoder
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# Features and target variable
X = df[['Material', 'Temperature', 'pH', 'FlowRate']]  # Features
y = df['CorrosionRate']  # Target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing and training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Saving the trained model and scaler
with open('corrosion_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model, scaler, and label encoder saved successfully!")

