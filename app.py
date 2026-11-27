
# Import the required libraries
import streamlit as st
import pandas as pd
import joblib

# Load the saved machine learning model
model = joblib.load("house_price_prediction_model.pkl")

# Load default house information
default_house = joblib.load("default_house.pkl")

# Configure the application page
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

# Display application heading
st.title("House Price Prediction System")

# Display project description
st.write("Enter house details to predict its estimated selling price.")

# Create an input form
with st.form("house_prediction_form"):

    # Collect house quality
    quality = st.slider(
        "House Quality",
        1, 10, 5
    )

    # Collect living area
    living_area = st.number_input(
        "Living Area (Square Feet)",
        min_value=200,
        max_value=10000,
        value=1500
    )

    # Collect garage capacity
    garage_cars = st.selectbox(
        "Garage Capacity",
        [0, 1, 2, 3, 4]
    )

    # Collect basement area
    basement = st.number_input(
        "Basement Area",
        min_value=0,
        max_value=5000,
        value=800
    )

    # Collect number of bathrooms
    bathrooms = st.selectbox(
        "Full Bathrooms",
        [1, 2, 3, 4]
    )

    # Collect year built
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    # Collect neighbourhood
    neighbourhoods = [
        "Blmngtn", "Blueste", "BrDale", "BrkSide",
        "ClearCr", "CollgCr", "Crawfor", "Edwards",
        "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
        "Names", "NoRidge", "NPkVill", "NridgHt",
        "NWAmes", "OldTown", "SWISU", "Sawyer",
        "SawyerW", "Somerst", "StoneBr", "Timber",
        "Veenker"
    ]

    neighbourhood = st.selectbox(
        "Neighbourhood",
        neighbourhoods
    )

    # Create prediction button
    submit = st.form_submit_button("Predict House Price")

# Process the prediction when submitted
if submit:

    # Create a DataFrame using the default house features
    house_data = pd.DataFrame([default_house])

    # Update the features selected by the user
    house_data["OverallQual"] = quality
    house_data["GrLivArea"] = living_area
    house_data["GarageCars"] = garage_cars
    house_data["TotalBsmtSF"] = basement
    house_data["FullBath"] = bathrooms
    house_data["YearBuilt"] = year_built
    house_data["Neighborhood"] = neighbourhood

    # Predict the house selling price
    prediction = model.predict(house_data)[0]

    # Display prediction result
    st.success("Prediction completed successfully!")

    st.metric(
        "Estimated House Price",
        f"${prediction:,.2f}"
    )

    # Explain the prediction
    st.info(
        "This prediction is based on the Kaggle Ames housing dataset."
    )
