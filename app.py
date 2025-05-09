# app.py

import streamlit as st
from data_loader import load_and_preprocess
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

st.set_page_config(page_title="AI Energy Predictor", layout="centered")

st.title("⚡ AI-Based Energy Consumption Predictor")

# Load data
st.info("Loading and training model, please wait...")
df = load_and_preprocess()

# Features and target
features = ['hour', 'day', 'month', 'lag_1']
target = 'Global_active_power'
X = df[features]
y = df[target]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar inputs
st.sidebar.header("Input Parameters")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.selectbox("Day of Week (0=Mon)", list(range(7)))
month = st.sidebar.selectbox("Month", list(range(1, 13)))
lag_1 = st.sidebar.number_input("Previous Hour's Power (kW)", min_value=0.0, value=1.0, step=0.1)

# Make prediction
input_data = pd.DataFrame([[hour, day, month, lag_1]], columns=features)
prediction = model.predict(input_data)[0]

# Show result
st.subheader("🔍 Prediction Result")
st.metric(label="Estimated Energy Consumption (kW)", value=round(prediction, 3))
