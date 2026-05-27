import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved files
model = joblib.load('food_price_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title('Food Price Prediction - Nigeria')
st.write('Fill in the details to predict food price')

# Actual values from your dataset
commodity_list = ['Rice', 'Tomatoes', 'Yam', 'Beans', 'Maize', 'Plantain', 'Cassava']
location_list  = ['Lagos', 'Abuja', 'Kano', 'Ibadan', 'Port Harcourt', 'Enugu', 'Kaduna']

commodity = st.selectbox(' Select Commodity', commodity_list)
location  = st.selectbox(' Select Location',  location_list)

temperature = st.slider('🌡️ Temperature (°C)', 20, 45, 30)
rainfall    = st.slider('🌧️ Rainfall (mm)',     0,  20,  5)
humidity    = st.slider('💧 Humidity (%)',      40, 100, 65)

# Encode inputs same way as training
commodity_encoded = commodity_list.index(commodity)
location_encoded  = location_list.index(location)

if st.button('Predict Price'):
    input_data = pd.DataFrame(
        [[location_encoded, commodity_encoded, temperature, rainfall, humidity]],
        columns=columns
    )
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)

    st.success(f'💰 Predicted Price of **{commodity}** in **{location}**: ₦ {prediction[0]:,.2f}')