import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="California Housing Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("california_housing_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(Path("datasets/housing/housing.csv"))

st.title("🏠 California Housing Price Prediction App")
st.write("This UI uses your trained Random Forest model to explore data and predict house prices.")

menu = st.sidebar.selectbox("Navigation", ["Home", "Dataset Explorer", "Make Prediction"])

# ---------------- HOME ----------------
if menu == "Home":
    st.header("Project Overview")
    st.markdown("""
    This application is built on your end-to-end machine learning pipeline:

    - California housing dataset
    - Data preprocessing with pipelines
    - RandomForestRegressor model
    - Model saved as `california_housing_model.pkl`

    You can explore the dataset and make live predictions.
    """)

# ---------------- DATASET EXPLORER ----------------
elif menu == "Dataset Explorer":
    st.header("Dataset Explorer")

    data = load_data()

    st.subheader("Raw Data")
    st.dataframe(data.head(50))

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

    st.subheader("Ocean Proximity Distribution")
    st.bar_chart(data["ocean_proximity"].value_counts())

# ---------------- PREDICTION ----------------
elif menu == "Make Prediction":
    st.header("Predict Median House Value")

    model = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23)
        latitude = st.number_input("Latitude", value=37.88)
        housing_median_age = st.number_input("Housing Median Age", value=41)

    with col2:
        total_rooms = st.number_input("Total Rooms", value=880)
        total_bedrooms = st.number_input("Total Bedrooms", value=129)
        population = st.number_input("Population", value=322)

    with col3:
        households = st.number_input("Households", value=126)
        median_income = st.number_input("Median Income", value=8.3252)
        ocean_proximity = st.selectbox("Ocean Proximity", [
            "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"
        ])

    if st.button("Predict Price"):
        input_data = pd.DataFrame([
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity
            }
        ])

        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Median House Value: ${prediction:,.0f}")

st.sidebar.markdown("---")
st.sidebar.info("Run with: streamlit run california_housing_ui.py")
