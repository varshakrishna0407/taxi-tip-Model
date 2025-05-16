
import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Streamlit UI configuration - Must be the first Streamlit command in your script
st.set_page_config(page_title=" Taxi Tip Predictor", layout="centered")

# Title of the app
st.title("Taxi Tip Predictor")
st.write("Enter the details below to predict the expected tip amount.")

# Load the trained model (should be a pipeline)
model = joblib.load("taximodel.pkl")

# Streamlit UI to take inputs
with st.form("tip_form"):
    total_bill = st.slider("Total Bill ($)", min_value=0.0, max_value=500.00,value=20.0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
    time = st.selectbox("Time", ["Lunch", "Dinner"])
    size = st.number_input("Party Size", min_value=1, value=2)

    # Submit button
    submitted = st.form_submit_button("Predict Tip")

# Prediction on form submission
if submitted:
    input_df = pd.DataFrame([{
        'total_bill': total_bill,
        'sex': sex,
        'smoker': smoker,
        'day': day,
        'time': time,
        'size': size
    }])

    # Print input data
    #st.write("Input Data:")
    #st.dataframe(input_df)

    # Check the model type again just before prediction
    #st.write(f"Model type before prediction: {type(model)}")  # Should show <class 'sklearn.pipeline.Pipeline'>

    try:
        # Predict the tip
        prediction = model.predict(input_df)

        # Ensure the output is a scalar value
        predicted_tip = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

        # Display the predicted tip
        st.success(f"Predicted Tip: ${predicted_tip:.2f}")
    except Exception as e:
        st.error(f" Error: {str(e)}")