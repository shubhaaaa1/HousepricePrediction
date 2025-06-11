# app.py

import streamlit as st
import pickle
import numpy as np

# Streamlit UI
st.set_page_config(page_title="House Price Prediction Model", layout="centered")

st.title("House Price Prediction using XGBRegressor")
st.markdown("Provide Exact 8 features Of the House to get an Prediction on the Prices of your house .")

st.subheader("🔍 Enter the 8 features of Your house as asked :")
input_1 = st.number_input("Medical Insurance",min_value=0.0)
input_2 = st.number_input("Your House Age",min_value=0.0)
input_3 = st.number_input("Average no. of rooms in ur house",min_value=0.0)
input_4 = st.number_input("Average no. of bedrooms in ur house",min_value=0.0)
input_5 = st.number_input("Population in Your Locality",min_value=0.0)
input_6 = st.number_input("Average Occupation",min_value=0.0)
input_7 = st.number_input("Your Latitude",format="%.6f")
input_8 = st.number_input("Your Longitude",format="%.6f")


try:
    with open("model.pkl","rb") as file:
        model=pickle.load(file)
except Exception as e:
    st.error("Error Occured while loading the Model. Make sure the model.pkl file exist")
    st.stop()

if st.button("Predict"):
    try:
        features=np.array([[input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8]])
        prediction = model.predict(features)[0]
        st.success(f"The predicted House Price (in USD) : $ {int(prediction*100000)}")
    except Exception as e:
        st.error(f"The prediction failed : {e}")



st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
