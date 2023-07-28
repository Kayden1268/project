import streamlit as st
import pandas as pd
import joblib

# Function to load the trained model
def load_model():
    return joblib.load("trained_lightgbm_model.joblib")

# Function to make predictions
def predict_price(model, input_features):
    return model.predict(input_features)

def main():
    st.title("House Price Prediction")
    st.write("Enter the features of the house to get the estimated price.")

    # Load the trained model
    model = load_model()
    # Create input widgets for the user to enter the features of the house
    UNDER_CONSTRUCTION = st.slider("UNDER_CONSTRUCTION", min_value=0, max_value=1, value=0, key="uc")
    RERA = st.slider("RERA", min_value=0, max_value=1, value=0, key="rera")
    BHK_NO = st.slider("BHK_NO.", min_value=1, max_value=5, value=2, key="bhk")
    SQUARE_FT = st.slider("SQUARE_FT", min_value=500, max_value=5000, value=1500, step=100, key="sqft")
    READY_TO_MOVE = st.slider("READY_TO_MOVE", min_value=0, max_value=1, value=1, key="rtm")
    RESALE = st.slider("RESALE", min_value=0, max_value=1, value=1, key="resale")
    BHK_OR_RK_BHK = st.slider("BHK_OR_RK_BHK", min_value=0, max_value=1, value=1, key="bhk_or_rk")
    # Convert the input features to a DataFrame
    input_features = {
        "UNDER_CONSTRUCTION": UNDER_CONSTRUCTION,
        "RERA": RERA,
        "BHK_NO.": BHK_NO,
        "SQUARE_FT": SQUARE_FT,
        "READY_TO_MOVE": READY_TO_MOVE,
        "RESALE": RESALE,
        "BHK_OR_RK_BHK": BHK_OR_RK_BHK,
    }
    input_df = pd.DataFrame([input_features])

    # Make predictions using the model
    predicted_price = predict_price(model, input_df)

    # Display the predicted price to the user
    st.subheader("Predicted Price")
    st.write(f"${predicted_price[0]:,.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
