import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load model (relative path — file sits at repo root)
model = joblib.load("lr_reg_model.pickle")

# 2. Extract feature names directly from the trained model
# (No json file required — guarantees exact column names & order!)
data_columns = list(model.feature_names_in_)

# Extract location options (skipping the first 3 numeric features: bath, bhk, total_sqft)
locations = data_columns[3:]

# 3. Streamlit Interface
st.title("Bangalore House Price Prediction")
st.text("By Sandilyan — Aspiring Data Analyst & Scientist")

total_sqft = st.number_input("Total Square Feet", value=1500, min_value=300, step=50)
bath = st.number_input("Bathroom", value=2, min_value=1, step=1)
bhk = st.number_input("BHK", value=2, min_value=1, step=1)
selected_location = st.selectbox("Location", locations)

st.write("")
button = st.button("Predict Price 🚀")

# 4. Prediction Logic
if button:
    # Find index position of chosen location
    loc_index = -1
    if selected_location in data_columns:
        loc_index = data_columns.index(selected_location)

    # Build zero-filled input array matching feature length
    x = np.zeros(len(data_columns))
    x[0] = bath
    x[1] = bhk
    x[2] = total_sqft

    # Set One-Hot Encoding flag for selected location
    if loc_index >= 0:
        x[loc_index] = 1

    # Convert row array into DataFrame with matching column names
    input_df = pd.DataFrame([x], columns=data_columns)

    # Predict & extract the single float output
    prediction = model.predict(input_df)[0]

    st.success(f"The Predicted Price for the House is: **{prediction:.2f} Lakhs**")

# 5. Background Styling (Scrolls along with page content)
st.markdown(
    """
<style>
	[data-testid="stAppViewContainer"] {
		background-image: url("https://img.freepik.com/premium-photo/luxurious-homes-upscale-neighborhood-stunning-aerial-view-concept-luxury-real-estate-aerial-photography-upscale-neighborhoods-beautiful-homes-premium-locations_918839-112196.jpg?w=2000");
		background-size: cover;
		background-position: center;
	}
</style>
""",
    unsafe_allow_html=True,
)
