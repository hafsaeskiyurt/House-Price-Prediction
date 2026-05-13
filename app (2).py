import streamlit as st
import joblib
import numpy as np

scaler=joblib.load("scaler.pkl")

model=joblib.load("model.pkl")

st.title("House Price Prediction App")

st.divider()

st.write("""This app is for getting a price estimation for a house based on metrics such as house age, population, number of rooms, and average income in the area.""")

income=st.number_input("Average Area Income", min_value=5000, max_value=200000, value=40000, step=5000)

houseage=st.number_input("House Age",min_value=0, max_value=100, value=10,step=1)

room=st.number_input("Number of Rooms",min_value=1, max_value=15,value=3,step=1)
population=st.number_input("Population in the Area",min_value=0, max_value=100000,value=30000,step=2000)

x=[income, houseage, room, population]
calculatebutton=st.button("Calculate")
st.divider()

if calculatebutton:
    
    x_2=np.array(x)
    x_array=scaler.transform([x_2])
    
    prediction=model.predict(x_array)
    st.write("Predicted price of the house is:", prediction[0])

    
else:
    st.write("Please enter the values and press the calculate button")

