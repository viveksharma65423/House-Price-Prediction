import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


model = joblib.load("house_price_model.pkl")
feature_columns = joblib.load("house_price_features.pkl")
df = pd.read_csv("house_price_india_2026.csv")
performance = joblib.load("model_performance.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")
st.write("Enter property details to predict the house price.")



City = st.selectbox(
    "City",
    sorted(df["City"].dropna().unique())
)



city_locations = df[df["City"] == City]["Location"].dropna().unique()

Location = st.selectbox(
    "Location",
    sorted(city_locations)
)

location_data=df[(df["City"] == City) & (df["Location"] == Location)]
#min_area=int(location_data["Area_sqft"].min())
#max_area=int(location_data["Area_sqft"].max())
available_areas=sorted(location_data["Area_sqft"].dropna().unique())
Area_sqft = st.slider(
    "Area (sqft)",
    min_value=500,
    max_value=5000,
    value=500,
    step=50
)
st.write(f"Selected Area: **{Area_sqft} sqft**")

Bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

Bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

Floor = st.number_input(
    "Floor",
    min_value=0,
    max_value=50,
    value=1
)

Property_Age_Years = st.number_input(
    "Property Age (Years)",
    min_value=0,
    max_value=100,
    value=5
)

Parking = st.number_input(
    "Parking",
    min_value=0,
    max_value=10,
    value=1
)



Furnished = st.selectbox(
    "Furnished",
    sorted(df["Furnished"].dropna().unique())
)
Property_Type = st.selectbox(
    "Property Type",
    sorted(df["Property_Type"].dropna().unique())
)


if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "City": [City],
        "Location": [Location],
        "Area_sqft": [Area_sqft],
        "Bedrooms": [Bedrooms],
        "Bathrooms": [Bathrooms],
        "Floor": [Floor],
        "Property_Age_Years": [Property_Age_Years],
        "Parking": [Parking],
        "Furnished": [Furnished],
        "Property_Type": [Property_Type]
    })


    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "City",
            "Location",
            "Furnished",
            "Property_Type"
        ],
        drop_first=True
    )

    
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    prediction = model.predict(input_encoded)[0]

    def format_price(price):
        if price >= 1_00_00_000:
            return f"₹ {price / 1_00_00_000:.2f} Cr"
        elif price >= 1_00_000:
            return f"₹ {price / 1_00_000:.2f} Lakh"
        else:
            return f"₹ {price:.2f}"

    st.success(f"Estimated House Price: {format_price(prediction)}")
    st.subheader("Property Details:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**City:** {City}")
        st.write(f"**Location:** {Location}")
        st.write(f"**Area (sqft):** {Area_sqft}")
        st.write(f"**Bedrooms:** {Bedrooms}")
        st.write(f"**Bathrooms:** {Bathrooms}")
    with col2:
        st.write(f"**Floor:** {Floor}")
        st.write(f"**Property Age (Years):** {Property_Age_Years}")
        st.write(f"**Parking:** {Parking}")
        st.write(f"**Furnished:** {Furnished}")
        st.write(f"**Property Type:** {Property_Type}")
    st.subheader("Model Performance Dashboard:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{performance['MAE']:.2f}")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{performance['RMSE']:.2f}")
    with col3:
        st.metric("R-squared (R2)", f"{performance['R2']:.4f}")

    actual_vs_predicted = pd.read_csv("actual_vs_predicted.csv")
    st.subheader("Actual vs Predicted Prices:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual_vs_predicted["Actual Price"], actual_vs_predicted["Predicted Price"], alpha=0.6)
    min_price = min(actual_vs_predicted["Actual Price"].min(), actual_vs_predicted["Predicted Price"].min())
    max_price = max(actual_vs_predicted["Actual Price"].max(), actual_vs_predicted["Predicted Price"].max())
    ax.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--', label='Perfect Prediction')
    ax.legend()
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)