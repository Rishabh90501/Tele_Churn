import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
try:
    model = joblib.load(r"C:\Users\cereb\OneDrive\Documents\Rishabh\Rishabh\Projects\Tele_Churn\tele_churn.h5")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: tele_churn.h5 not found. Make sure the model file is in the correct directory.")
    st.stop()

# Add a title to the app
st.title("Telecom Customer Churn Prediction")

# Try to get feature names directly from the kernel's df_dummies if available
try:
    # Access df_dummies from the kernel state
    feature_names = [col for col in df_dummies.columns if col != 'Churn']
    st.info("Using feature names from the notebook kernel's processed training data.")
except NameError:
    # If df_dummies is not available, load sample data and preprocess
    st.warning("Could not access processed training data from the kernel. Loading sample data to determine feature names.")
    try:
        # Load data - make sure the path is correct for your environment
        sample_data = pd.read_csv(r"C:\Users\cereb\OneDrive\Documents\Rishabh\Rishabh\Projects\Tele_Churn\Data\Telco_Customer_Churn.csv")

        # Apply the same preprocessing steps as in the training notebook
        sample_data.TotalCharges = pd.to_numeric(sample_data.TotalCharges, errors="coerce")
        sample_data.dropna(inplace=True)
        sample_data["Churn"].replace(to_replace="Yes", value=1, inplace=True)
        sample_data["Churn"].replace(to_replace="No", value=0, inplace=True)

        # Drop the customerID column if it exists
        if 'customerID' in sample_data.columns:
            sample_data = sample_data.drop('customerID', axis=1)

        # Perform one-hot encoding to get the template for feature names, keeping all features except Churn
        df_dummies_template = pd.get_dummies(sample_data.drop('Churn', axis=1))

        # Get the list of features from the processed data
        feature_names = df_dummies_template.columns.tolist()

    except FileNotFoundError:
        st.error("Error: Telco_Customer_Churn.csv not found. Make sure the data file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading and preprocessing: {e}")
        st.stop()


# Define categorical features for dropdowns
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Add input fields for features
st.header("Enter Customer Details:")

input_data = {}
# Iterate through the columns of the original sample_data to create input fields
# Need to ensure sample_data is available here if df_dummies was used for feature_names
try:
    # If sample_data wasn't loaded in the except block, load it now
    if 'sample_data' not in locals():
        sample_data = pd.read_csv("Telco_Customer_Churn.csv")
        sample_data.TotalCharges = pd.to_numeric(sample_data.TotalCharges, errors="coerce")
        sample_data.dropna(inplace=True)
        sample_data["Churn"].replace(to_replace="Yes", value=1, inplace=True)
        sample_data["Churn"].replace(to_replace="No", value=0, inplace=True)
    if 'customerID' in sample_data.columns:
        sample_data = sample_data.drop('customerID', axis=1)
    if 'MonthlyCharges' in sample_data.columns:
        sample_data = sample_data.drop('MonthlyCharges', axis=1)
    if 'TotalCharges' in sample_data.columns:
        sample_data = sample_data.drop('TotalCharges', axis=1)

    for feature in sample_data.columns:
        if feature == 'Churn' or feature == 'customerID':
            continue # Skip the target variable and customerID

        if feature in categorical_features:
            if feature == 'SeniorCitizen':
                 # Display 'Yes'/'No' but store 1/0
                 senior_citizen_options = ['No', 'Yes']
                 selected_value = st.selectbox(f"Select {feature}", senior_citizen_options)
                 input_data[feature] = 1 if selected_value == 'Yes' else 0
            else:
                 options = sample_data[feature].unique().tolist()
                 input_data[feature] = st.selectbox(f"Select {feature}", options)
        elif sample_data[feature].dtype in ['int64', 'float64']:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(sample_data[feature].mean()))
        else:
            input_data[feature] = st.text_input(f"Enter {feature}")
except FileNotFoundError:
    st.error("Error: Telco_Customer_Churn.csv not found. Cannot display input fields.")
    st.stop()


# Add a button to make predictions
if st.button("Predict Churn"):
    # Prepare input data for prediction
    input_df = pd.DataFrame([input_data])

    # Apply the same preprocessing steps as in training, specifically one-hot encoding
    # We need to ensure the columns match the training data after one-hot encoding
    input_df_encoded = pd.get_dummies(input_df)

    # Align columns - add missing columns with 0 and reorder
    for col in feature_names:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[feature_names]

    # Check if the number of features matches before prediction
    if input_df_encoded.shape[1] != len(feature_names):
        st.error(f"Feature mismatch: Input data has {input_df_encoded.shape[1]} features, but the model expects {len(feature_names)}. Please check your data and preprocessing steps.")
        st.stop()


    # Scale the numerical features - identify numerical columns from the original sample_data
    # Ensure 'MonthlyCharges' and 'TotalCharges' are included here
    numerical_features = sample_data.select_dtypes(include=np.number).columns.tolist()
    if 'Churn' in numerical_features:
        numerical_features.remove('Churn')
    if 'SeniorCitizen' in numerical_features:
        numerical_features.remove('SeniorCitizen') # Assuming SeniorCitizen is treated as categorical


    # Need to fit the scaler on the training data's numerical columns and then transform the input
    # For simplicity here, we'll refit a scaler. In a real app, you'd save and load the fitted scaler.
    scaler = StandardScaler()
    # Fit on a subset of sample_data that reflects the training data's numerical columns
    # Use the original sample_data for fitting the scaler
    scaler.fit(sample_data[numerical_features])

    # Apply scaling to the numerical columns of the encoded input data
    for num_col in numerical_features:
        if num_col in input_df_encoded.columns:
             # Reshape the input for the scaler if it's a single sample
             input_df_encoded[num_col] = scaler.transform(input_df_encoded[[num_col]])


    # Make prediction
    prediction = model.predict(input_df_encoded)
    prediction_proba = model.predict_proba(input_df_encoded)

    # Display prediction
    if prediction[0] == 1:
        st.error("This customer is likely to Churn.")
        st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    else:
        st.success("This customer is likely to stay.")
        st.write(f"Probability of Staying: {prediction_proba[0][0]:.2f}")