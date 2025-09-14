import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
st.title("📊 Dynamic Pricing Engine Dashboard")

# ---------------------------
# Project Overview
# ---------------------------
st.markdown("""
### 🚀 Dynamic Pricing Engine
This dashboard predicts **optimal product prices** using ML models (LightGBM & XGBoost) based on product data, competitor pricing, stock levels, and seasonal factors. 
It also calculates **expected revenue and profit**, helping businesses make **data-driven pricing decisions**.
""")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Product Input")
product_id = st.sidebar.number_input("Product ID", min_value=1, step=1)
units_sold = st.sidebar.number_input("Units Sold", min_value=0, step=1)
competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.0, step=0.1)
stock_level = st.sidebar.number_input("Stock Level", min_value=0, step=1)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 0)
holiday_flag = st.sidebar.selectbox("Holiday?", [0, 1])
views = st.sidebar.number_input("Views", min_value=0, step=1)

API_URL = "http://127.0.0.1:8000/predict-price"

# ---------------------------
# Predict Button
# ---------------------------
if st.sidebar.button("Predict Price"):
    payload = {
        "product_id": product_id,
        "units_sold": units_sold,
        "competitor_price": competitor_price,
        "stock_level": stock_level,
        "day_of_week": day_of_week,
        "holiday_flag": holiday_flag,
        "views": views
    }

    # Call FastAPI
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        # ---------------------------
        # Price Prediction Display
        # ---------------------------
        st.subheader("🔮 Price Predictions")
        st.write(result)

        # ---------------------------
        # Revenue & Profit Metrics
        # ---------------------------
        avg_predicted_price = (result["lightgbm_price"] + result["xgboost_price"]) / 2
        revenue = units_sold * avg_predicted_price
        cost_per_unit = competitor_price * 0.7  # placeholder assumption
        profit = revenue - (units_sold * cost_per_unit)

        # Color-coded metrics
        def color_value(val, threshold=0):
            return "green" if val >= threshold else "red"

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Avg Predicted Price", 
            f"${avg_predicted_price:.2f}", 
            delta=None
        )
        col2.metric(
            "Expected Revenue", 
            f"${revenue:.2f}", 
            delta=None
        )
        col3.metric(
            "Expected Profit", 
            f"${profit:.2f}", 
            delta=None
        )

        # ---------------------------
        # Download Metrics CSV
        # ---------------------------
        st.download_button(
            label="Download Metrics CSV",
            data=pd.DataFrame([{
                "Avg Predicted Price": avg_predicted_price,
                "Expected Revenue": revenue,
                "Expected Profit": profit
            }]).to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # ---------------------------
        # Profit Visualization
        # ---------------------------
        st.subheader("💹 Profit Analysis")
        fig, ax = plt.subplots(figsize=(2, 2))  # very compact
        ax.bar(["Profit"], [profit], color="green" if profit >= 0 else "red")
        ax.set_ylabel("Profit ($)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        st.pyplot(fig)

        # ---------------------------
        # Model Performance (Static Example)
        # ---------------------------
        st.subheader("📈 Model Performance")
        st.write({
            "LightGBM MAE": 12.5,
            "XGBoost MAE": 13.2,
            "LightGBM R²": 0.87,
            "XGBoost R²": 0.85
        })

        # ---------------------------
        # Expandable "How it works"
        # ---------------------------
        with st.expander("How it works"):
            st.write("""
            1. User enters product details in the sidebar.
            2. Dashboard sends data to FastAPI ML backend.
            3. ML models predict optimal price (LightGBM & XGBoost).
            4. Dashboard calculates revenue & profit.
            5. Users can download metrics or visualize profit.
            """)

        # ---------------------------
        # Tech Stack
        # ---------------------------
        st.markdown("""
        **Tech Stack:** Python, FastAPI, Streamlit, LightGBM, XGBoost, MLflow
        """)

    else:
        st.error(f"API Error: {response.status_code}")

# import streamlit as st
# import requests
# import pandas as pd

# st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
# st.title("📊 Dynamic Pricing Engine Dashboard")

# # Sidebar input
# st.sidebar.header("Product Input")
# product_id = st.sidebar.number_input("Product ID", min_value=1, step=1)
# units_sold = st.sidebar.number_input("Units Sold", min_value=0, step=1)
# competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.0, step=0.1)
# stock_level = st.sidebar.number_input("Stock Level", min_value=0, step=1)
# day_of_week = st.sidebar.slider("Day of Week", 0, 6, 0)
# holiday_flag = st.sidebar.selectbox("Holiday?", [0, 1])
# views = st.sidebar.number_input("Views", min_value=0, step=1)

# if st.sidebar.button("Predict Price"):
#     payload = {
#         "product_id": product_id,
#         "units_sold": units_sold,
#         "competitor_price": competitor_price,
#         "stock_level": stock_level,
#         "day_of_week": day_of_week,
#         "holiday_flag": holiday_flag,
#         "views": views
#     }
    
#     # Call FastAPI
#     response = requests.post(API_URL, json=payload)
    
#     if response.status_code == 200:
#         result = response.json()
        
#         st.subheader("🔮 Price Predictions")
#         st.write(result)

#         # Revenue & Profit calc
#         avg_predicted_price = (result["lightgbm_price"] + result["xgboost_price"]) / 2
#         revenue = units_sold * avg_predicted_price
#         cost_per_unit = competitor_price * 0.7  # placeholder assumption
#         profit = revenue - (units_sold * cost_per_unit)

#         metrics = {
#             "Avg Predicted Price": avg_predicted_price,
#             "Expected Revenue": revenue,
#             "Expected Profit": profit
#         }
#         st.subheader("💰 Revenue & Profit Analysis")
#         st.dataframe(pd.DataFrame([metrics]))
        
#     else:
#         st.error(f"API Error: {response.status_code}")



# st.title("Dynamic Pricing Engine")

# product_id = st.number_input("Product ID", min_value=1, max_value=500, value=101)
# units_sold = st.number_input("Units Sold", value=20)
# competitor_price = st.number_input("Competitor Price", value=200)
# stock_level = st.number_input("Stock Level", value=50)
# day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=4)
# holiday_flag = st.selectbox("Holiday Flag", [0,1], index=0)
# views = st.number_input("Views", value=100)

# if st.button("Predict Price"):
#     data = {
#         "product_id": product_id,
#         "units_sold": units_sold,
#         "competitor_price": competitor_price,
#         "stock_level": stock_level,
#         "day_of_week": day_of_week,
#         "holiday_flag": holiday_flag,
#         "views": views
#     }
#     response = requests.post("http://127.0.0.1:8000/predict-price", json=data)
#     st.write(response.json())
# import streamlit as st
# import requests
# import pandas as pd

# st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
# st.title("📊 Dynamic Pricing Engine Dashboard")

# # Sidebar input
# st.sidebar.header("Product Input")
# product_id = st.sidebar.number_input("Product ID", min_value=1, step=1)
# units_sold = st.sidebar.number_input("Units Sold", min_value=0, step=1)
# competitor_price = st.sidebar.number_input("Competitor Price", min_value=0.0, step=0.1)
# stock_level = st.sidebar.number_input("Stock Level", min_value=0, step=1)
# day_of_week = st.sidebar.slider("Day of Week", 0, 6, 0)
# holiday_flag = st.sidebar.selectbox("Holiday?", [0, 1])
# views = st.sidebar.number_input("Views", min_value=0, step=1)

# API_URL = "http://127.0.0.1:8000/predict-price"

# if st.sidebar.button("Predict Price"):
#     payload = {
#         "product_id": product_id,
#         "units_sold": units_sold,
#         "competitor_price": competitor_price,
#         "stock_level": stock_level,
#         "day_of_week": day_of_week,
#         "holiday_flag": holiday_flag,
#         "views": views
#     }

#     # Call FastAPI
#     response = requests.post(API_URL, json=payload)

#     if response.status_code == 200:
#         result = response.json()

#         st.subheader("🔮 Price Predictions")
#         st.write(result)

#         # Revenue & Profit calc
#         avg_predicted_price = (result["lightgbm_price"] + result["xgboost_price"]) / 2
#         revenue = units_sold * avg_predicted_price
#         cost_per_unit = competitor_price * 0.7  # placeholder assumption
#         profit = revenue - (units_sold * cost_per_unit)

#         metrics = {
#             "Avg Predicted Price": avg_predicted_price,
#             "Expected Revenue": revenue,
#             "Expected Profit": profit
#         }
#         st.subheader("💰 Revenue & Profit Analysis")
#         st.dataframe(pd.DataFrame([metrics]))

#     else:
#         st.error(f"API Error: {response.status_code}")
