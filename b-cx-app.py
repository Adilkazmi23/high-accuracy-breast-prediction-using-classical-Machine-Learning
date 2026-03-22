import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. PAGE SETUP (Must be the very first Streamlit command)
import streamlit as st


# 2. LOAD DATA & TRAIN MODEL (Defined once)
import streamlit as st
@st.cache_resource
def get_model_and_scaler():
    # Load the Wisconsin dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple model for the UI
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return data, model, scaler

# Call the function to create global variables
data, model, scaler = get_model_and_scaler()

# 3. SIDEBAR INPUTS
def user_input_features():
    st.sidebar.header('Patient Input Metrics')
    user_data = {}
    # Creating sliders for all 30 features automatically
    for i, name in enumerate(data.feature_names):
        # Set default to the mean value of the dataset
        default_val = float(data.data[:, i].mean())
        user_data[name] = st.sidebar.slider(name.capitalize(), 0.0, default_val*3, default_val)
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# 4. MAIN UI LOGIC
st.title("🩺 Breast Cancer Diagnostic System")
st.markdown("Adjust the patient metrics in the sidebar to generate a prediction.")

# Get inputs from sidebar
input_df = user_input_features()

# Display current inputs
with st.expander("View Input Data"):
    st.write(input_df)

# PREDICTION BUTTON
if st.button('Run Diagnostic Analysis'):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.error("🚨 The model predicts: **MALIGNANT**")
        else:
            st.success("✅ The model predicts: **BENIGN**")
            
    with col2:
        st.subheader("Prediction Confidence")
        confidence = np.max(prediction_proba) * 100
        st.metric("Probability", f"{confidence:.2f}%")

st.info("**Project Note:** This model uses a Random Forest Classifier trained on the Wisconsin Breast Cancer dataset.")