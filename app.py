# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
import lightgbm as lgb
# from lightgbm import LGBMRegressor 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import GridSearchCV
# st.write("""
# # House Price Prediction

# """)

# st.sidebar.header('User Input Features')
# def user_input_features():
#     under_construction = st.radio("Under Construction", ["0", "1"])
#     rera_approved = st.radio("Rera Approved", ["0", "1"])
#     ready_to_move = st.radio("Ready to Move", ["0", "1"])
#     property_type = st.sidebar.radio('BHK Property Type', ["0", "1"])
#     resale = st.radio("Being Re-Sold", ["0", "1"])
#     number_of_rooms = st.sidebar.slider('Number of Rooms', 1, 5, 3)
#     area = st.sidebar.slider('Area of House(SQFT)', 500, 2000, 1000 )
#     data = {
#         'UNDER_CONSTRUCTION': under_construction,
#         'RERA': rera_approved,
#         'READY_TO_MOVE': ready_to_move,
#         'RESALE': resale,
#         'BHK_OR_RK_BHK': property_type,
#         'BHK_NO.': number_of_rooms,
#         'SQUARE_FT': area
#     }
#     features = pd.DataFrame(data, index=[0])
#     return features

# input_df = user_input_features()


# #combine user input features with entire dataset

# housing = pd.read_csv('Housing.csv')


# # Displays the user input features

# st.subheader('User Input Features')


# # Reads in saved classification model

# load_clf = pickle.load(open('trained_model.pkl', 'rb'))


# # Apply model to make predictions

# prediction = load_clf.predict(input_df)


# st.subheader('Prediction')

# st.write(prediction)


import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import lightgbm as lgb
from lightgbm import LGBMRegressor 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


# App title and description
st.write("""
# House Price Prediction
""")
st.write("Welcome to the House Price Prediction App. Enter the details of the house and get the estimated price.")

def user_input_features():
    st.sidebar.header('User Input Features')
    
    under_construction = st.sidebar.radio("Under Construction", ["0", "1"])
    rera_approved = st.sidebar.radio("Rera Approved", ["0", "1"])
    ready_to_move = st.sidebar.radio("Ready to Move", ["0", "1"])
    property_type = st.sidebar.radio('BHK Property Type', ["0", "1"])
    resale = st.sidebar.radio("Being Re-Sold", ["0", "1"])
    
    number_of_rooms = st.slider('Number of Rooms', 1, 5, 3)
    area = st.slider('Area of House(SQFT)', 500, 2000, 1000)
    
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
    
    # Convert columns to integers or floats
    features['UNDER_CONSTRUCTION'] = features['UNDER_CONSTRUCTION'].astype(int)
    features['RERA'] = features['RERA'].astype(int)
    features['READY_TO_MOVE'] = features['READY_TO_MOVE'].astype(int)
    features['RESALE'] = features['RESALE'].astype(int)
    features['BHK_OR_RK_BHK'] = features['BHK_OR_RK_BHK'].astype(int)
    features['BHK_NO.'] = features['BHK_NO.'].astype(int)
    features['SQUARE_FT'] = features['SQUARE_FT'].astype(float)
    
    return features

input_df = user_input_features()

# Reads in saved regression model
housing = pd.read_csv("Cleaned_Housing.csv", encoding='cp1252')
features_df = housing.copy()
del features_df['TARGET(PRICE_IN_LACS)']
X = features_df.values
y = housing['TARGET(PRICE_IN_LACS)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create a MinMaxScaler object
scaler = MinMaxScaler()
# Fit and transform the scaler on the training data (normalizing in place)
X_train = scaler.fit_transform(X_train)
# Transform the test data using the same scaler (normalizing in place)
X_test = scaler.transform(X_test)

# Apply the same scaling to the user input data
input_df_scaled = scaler.transform(input_df)

# Convert the DataFrame to a 1-dimensional array (single row)
input_array = input_df_scaled[0].reshape(1, -1)

# Make sure the array has the correct shape for prediction
if input_array.shape[1] != X_train.shape[1]:
    st.write("Error: Input features do not match the trained model.")
else:
    clf = LGBMRegressor()
    # Define hyperparameters
    param_grid = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'max_depth': 4,
        'min_child_samples': 9,
        'n_estimators': 500,
        'objective': 'regression_l2'
    }

    # Train the LightGBM model
    clf.fit(X_train, y_train)

    # Make predictions using the scaled input data
    prediction = clf.predict(input_array)
    
    st.subheader('User Input Features')
    st.write(input_df)
    
    st.subheader('Prediction (IN LAKHS)')
    st.write(prediction[0])  # Display the scalar prediction instead of an array
