import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎")
st.title("💎 Diamond Price Predictor")
st.caption("Machine Learning Fundamentals (AIC354) – Assignment 3")

@st.cache_resource
def load_model():
    return joblib.load("diamond_price_model.pkl")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat", 0.2, 5.0, 0.9, 0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D best – J worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)
with col2:
    depth = st.number_input("Depth %", 40.0, 80.0, 61.5, 0.1)
    table = st.number_input("Table %", 40.0, 100.0, 57.0, 0.1)
    x = st.number_input("Length x (mm)", 0.1, 12.0, 6.2, 0.01)
    y = st.number_input("Width y (mm)", 0.1, 60.0, 6.25, 0.01)
    z = st.number_input("Depth z (mm)", 0.1, 35.0, 3.8, 0.01)

if st.button("Predict Price", type="primary"):
    row = pd.DataFrame([{"cut": cut, "color": color, "clarity": clarity, "carat": carat,
                         "depth": depth, "table": table, "x": x, "y": y, "z": z}])
    price = float(model.predict(row)[0])
    st.success(f"Estimated price: **${max(price, 0):,.2f}**")
