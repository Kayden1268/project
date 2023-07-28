import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

st.write("""
# House Price Prediction

""")

st.sidebar.header('User Input Features')
def user_input_features():
    under_construction = st.radio("Under Construction", ["0", "1"])
    rera_approved = st.radio("Rera Approved", ["0", "1"])
    ready_to_move = st.radio("Ready to Move", ["0", "1"])
    property_type = st.sidebar.radio('BHK Property Type', ["0", "1"])
    resale = st.radio("Being Re-Sold", ["0", "1"])
    number_of_rooms = st.sidebar.slider('Number of Rooms', 1, 5, 3)
    area = st.sidebar.slider('Area of House(SQFT)', 500, 2000, 1000 )
    data = {
        'UNDER_CONSTRUCTION': under_construction,
        'RERA': rera_approved,
        'READY_TO_MOVE': ready_to_move,
        'RESALE': resale,
        'BHK_OR_RK_BHK': property_type,
        'BHK_NO.': number_of_rooms,
        'SQUARE_FT': area
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


#combine user input features with entire dataset

housing = pd.read_csv('Housing.csv')


# Displays the user input features

st.subheader('User Input Features')


# Reads in saved classification model

load_clf = pickle.load(open('trained_model.pkl', 'rb'))


# Apply model to make predictions

prediction = load_clf.predict(input_df)


st.subheader('Prediction')

st.write(prediction)


