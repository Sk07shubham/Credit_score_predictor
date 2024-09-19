import streamlit as st
import pickle
import pandas as pd

# Load the pickled model and encoders
enc = pickle.load(open("credit_score_multi_class_ord_encoder.pkl", 'rb'))
le = pickle.load(open("credit_score_multi_class_le.pkl", 'rb'))
dummy = pickle.load(open("credit_score_multi_class_dummy.pkl", 'rb'))
model = pickle.load(open("credit_score_model.pkl", 'rb'))

# Define the prediction function
def predict_credit(data):
    # Preprocess the data using the loaded encoders and transformer
    data_encoded = enc.transform(data)
    data_dummy = dummy.transform(data_encoded)
    prediction = model.predict(data_dummy)
    prediction_proba = model.predict_proba(data_dummy)
    return prediction, prediction_proba

# Streamlit app
st.title("Credit Score Prediction")

# Define the input fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Income", min_value=0, value=50000)
job_type = st.selectbox("Job Type", ["Scientist", "Engineer", "Artist", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Collect input data in a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'JobType': [job_type],
    'MaritalStatus': [marital_status]
})

# Make predictions when the button is clicked
if st.button("Predict Credit Score"):
    prediction, prediction_proba = predict_credit(input_data)
    st.write(f"Predicted Credit Score: {prediction[0]}")
    st.write("Prediction Probabilities:")
    st.write(prediction_proba)
