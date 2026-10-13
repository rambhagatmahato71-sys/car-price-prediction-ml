import pickle as pk
import numpy as np
import pandas as pd
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Car Price Predictor", page_icon="🚗", layout="centered"
)

# Header Section
st.title("🚗 Car Price Prediction ML Model")
st.markdown(
    "Enter the details of the car below to predict its estimated selling price."
)
st.divider()

# 1. Load Trained Model
try:
  model = pk.load(open("model.pkl", "rb"))
except Exception as e:
  st.error(f"Error loading model.pkl file: {e}")
  st.stop()

# 2. Input UI Layout (2 Columns for Clean UI)
col1, col2 = st.columns(2)

with col1:
  # Car Brand Selection
  brand_list = [
      "Hyundai",
      "Maruti",
      "Tata",
      "Honda",
      "Kia",
      "Toyota",
      "Renault",
      "Ford",
      "Mahindra",
  ]
  brand_map = {brand: idx + 1 for idx, brand in enumerate(brand_list)}
  selected_brand = st.selectbox("Select Car Brand", brand_list)
  encoded_brand = brand_map[selected_brand]

  # Manufactured Year
  year = st.slider("Manufactured Year", 1990, 2026, 2017)

  # Kilometers Driven
  km_driven = st.number_input(
      "Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000
  )

  # Fuel Type
  fuel_map = {"CNG": 0, "Diesel": 1, "LPG": 2, "Petrol": 3}
  selected_fuel = st.selectbox("Fuel Type", list(fuel_map.keys()))
  encoded_fuel = fuel_map[selected_fuel]

  # Seller Type
  seller_map = {
      "Dealer": 0,
      "Individual": 1,
      "Trustmark Dealer": 2,
  }
  selected_seller = st.selectbox("Seller Type", list(seller_map.keys()))
  encoded_seller = seller_map[selected_seller]

with col2:
  # Transmission Type
  transmission_map = {"Automatic": 0, "Manual": 1}
  selected_transmission = st.selectbox(
      "Transmission Type", list(transmission_map.keys())
  )
  encoded_transmission = transmission_map[selected_transmission]

  # Owner Type
  owner_map = {
      "First Owner": 0,
      "Second Owner": 1,
      "Third Owner": 2,
      "Fourth & Above Owner": 3,
      "Test Drive Car": 4,
  }
  selected_owner = st.selectbox("Owner Type", list(owner_map.keys()))
  encoded_owner = owner_map[selected_owner]

  # Mileage
  mileage = st.slider("Mileage (kmpl)", 5.0, 50.0, 20.0, step=0.1)

  # Engine CC
  engine = st.number_input("Engine Capacity (CC)", 500, 5000, 1200, step=50)

  # Max Power (bhp)
  max_power = st.number_input(
      "Max Power (bhp)", 30.0, 500.0, 85.0, step=0.5
  )

  # Seats
  seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)

st.divider()

# 3. Prediction Action
if st.button("🚀 Predict Car Price", use_container_width=True):
  # Prepare Input DataFrame matching model's exact features & order
  input_data = pd.DataFrame(
      [[
          encoded_brand,
          year,
          km_driven,
          encoded_fuel,
          encoded_seller,
          encoded_transmission,
          encoded_owner,
          mileage,
          engine,
          max_power,
          seats,
      ]],
      columns=[
          "name",
          "year",
          "km_driven",
          "fuel",
          "seller_type",
          "transmission",
          "owner",
          "mileage",
          "engine",
          "max_power",
          "seats",
      ],
  )

  # Perform Prediction
  predicted_price = model.predict(input_data)[0]

  # Display Output
  st.success("### Estimated Car Price:")
  st.metric(
      label="Predicted Price (INR)", value=f"₹ {predicted_price:,.2f}"
  )