import streamlit as st
import requests
import pandas as pd
import numpy as np

st.title("SuperKart Sales Prediction")

st.write("Enter the product and store details to get a sales prediction.")

# Input fields for product and store details
product_id = st.text_input("Product ID")
product_weight = st.number_input("Product Weight", min_value=0.0)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar', 'reg'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0)
product_type = st.selectbox("Product Type", ['Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Snack Foods', 'Household', 'Soft Drinks', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood', 'Meat', 'Fruits and Vegetables'])
product_mrp = st.number_input("Product MRP", min_value=0.0)
store_id = st.selectbox("Store ID", ['OUT004', 'OUT003', 'OUT001', 'OUT002']) # Based on unique values from EDA
store_establishment_year = st.selectbox("Store Establishment Year", [1987, 1998, 1999, 2009]) # Based on unique values from EDA
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 2', 'Tier 1', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2', 'Departmental Store', 'Supermarket Type1', 'Food Mart'])


# Create a dictionary with the input data
data = {
    'Product_Id': [product_id],
    'Product_Weight': [product_weight],
    'Product_Sugar_Content': [product_sugar_content],
    'Product_Allocated_Area': [product_allocated_area],
    'Product_Type': [product_type],
    'Product_MRP': [product_mrp],
    'Store_Id': [store_id],
    'Store_Establishment_Year': [store_establishment_year],
    'Store_Size': [store_size],
    'Store_Location_City_Type': [store_location_city_type],
    'Store_Type': [store_type]
}

# Convert the dictionary to a pandas DataFrame
input_df = pd.DataFrame(data)

# Button to trigger prediction
if st.button("Predict Sales"):

    api_url = "https://hellohatim-superkart-sales-prediction-backend.hf.space//predict"

    try:
        response = requests.post(api_url, json=input_df.to_dict(orient='records'))

        if response.status_code == 200:
            predictions = response.json()
            st.success(f"Predicted Sales Total: {predictions[0]:.2f}")
        else:
            st.error(f"Error predicting sales: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"✅ Predicted Sales Total: (Confidence: {probability:.2f})")



import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="JohnsonSAimlarge/superkart-prediction", filename="best_superkart_model_v1.joblib")
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("""Superkart""" + " - Sales Prediction ""superkart sales Prediction App")

st.write("""
This application predicts potential sales for superkart.
""")


# Input fields for product and store details
product_id = st.text_input("Product ID")
product_weight = st.number_input("Product Weight", min_value=0.0)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar', 'reg'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0)
product_type = st.selectbox("Product Type", ['Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Snack Foods', 'Household', 'Soft Drinks', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood', 'Meat', 'Fruits and Vegetables'])
product_mrp = st.number_input("Product MRP", min_value=0.0)
store_id = st.selectbox("Store ID", ['OUT004', 'OUT003', 'OUT001', 'OUT002']) # Based on unique values from EDA
store_establishment_year = st.selectbox("Store Establishment Year", [1987, 1998, 1999, 2009]) # Based on unique values from EDA
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 2', 'Tier 1', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2', 'Departmental Store', 'Supermarket Type1', 'Food Mart'])


# Create a dictionary with the input data
data = {
    'Product_Id': [product_id],
    'Product_Weight': [product_weight],
    'Product_Sugar_Content': [product_sugar_content],
    'Product_Allocated_Area': [product_allocated_area],
    'Product_Type': [product_type],
    'Product_MRP': [product_mrp],
    'Store_Id': [store_id],
    'Store_Establishment_Year': [store_establishment_year],
    'Store_Size': [store_size],
    'Store_Location_City_Type': [store_location_city_type],
    'Store_Type': [store_type]
}


input_df = pd.DataFrame([input_data])

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"✅ Predicted Sales Total: (Confidence: {probability:.2f})")
