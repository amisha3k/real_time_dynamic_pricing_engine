import streamlit as st
import requests

st.title("Dynamic Pricing Engine")

product_id = st.number_input("Product ID", min_value=1, max_value=500, value=101)
units_sold = st.number_input("Units Sold", value=20)
competitor_price = st.number_input("Competitor Price", value=200)
stock_level = st.number_input("Stock Level", value=50)
day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=4)
holiday_flag = st.selectbox("Holiday Flag", [0,1], index=0)
views = st.number_input("Views", value=100)

if st.button("Predict Price"):
    data = {
        "product_id": product_id,
        "units_sold": units_sold,
        "competitor_price": competitor_price,
        "stock_level": stock_level,
        "day_of_week": day_of_week,
        "holiday_flag": holiday_flag,
        "views": views
    }
    response = requests.post("http://127.0.0.1:8000/predict-price", json=data)
    st.write(response.json())
