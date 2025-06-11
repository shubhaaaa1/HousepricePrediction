# app.py

import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="House Price Prediction Model", layout="centered")

st.title("House Price Prediction using XGBRegressor")
st.markdown("Provide the 8 features of the house to predict its price.")

st.subheader("🔍 Enter the 8 features of your house:")

input_1 = st.number_input("Feature 1 Med Inc", min_value=0.0)
input_2 = st.number_input("House Age", min_value=0.0)
input_3 = st.number_input("Average number of rooms", min_value=0.0)
input_4 = st.number_input("Average number of bedrooms", min_value=0.0)
input_5 = st.number_input("Local Population", min_value=0.0)
input_6 = st.number_input("Average Occupancy", min_value=0.0)
input_7 = st.number_input("Latitude", format="%.6f")
input_8 = st.number_input("Longitude", format="%.6f")

# Load model safely
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

if st.button("Predict"):
    try:
        features = np.array([[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted House Price (in USD): $ {int(prediction * 100000)}")
    except Exception as e:
        st.error(f"The prediction failed: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn")
