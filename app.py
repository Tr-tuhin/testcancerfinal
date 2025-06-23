import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('Random_Forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid scikit-learn model")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

# App UI
st.title('Lung Cancer Prediction System')
st.write("""
This app predicts the likelihood of lung cancer based on various health and lifestyle factors.
""")

st.header('Patient Information')
col1, col2 = st.columns(2)

with col1:
    gender = st.radio('Gender', ['Male', 'Female'], index=0)
    age = st.slider('Age', 20, 100, 60)
    smoking = st.selectbox('Smoking', ['No', 'Yes'], index=1)
    yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'], index=1)
    anxiety = st.selectbox('Anxiety', ['No', 'Yes'], index=0)
    peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'], index=0)
    chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'], index=1)

with col2:
    fatigue = st.selectbox('Fatigue', ['No', 'Yes'], index=1)
    allergy = st.selectbox('Allergy', ['No', 'Yes'], index=0)
    wheezing = st.selectbox('Wheezing', ['No', 'Yes'], index=1)
    alcohol = st.selectbox('Alcohol Consuming', ['No', 'Yes'], index=0)
    coughing = st.selectbox('Coughing', ['No', 'Yes'], index=1)
    shortness = st.selectbox('Shortness of Breath', ['No', 'Yes'], index=1)
    swallowing = st.selectbox('Swallowing Difficulty', ['No', 'Yes'], index=0)
    chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'], index=1)

# Prepare input
def prepare_input():
    input_data = {
        'GENDER': 1 if gender == 'Male' else 0,
        'AGE': age,
        'SMOKING': 1 if smoking == 'Yes' else 0,
        'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
        'ANXIETY': 1 if anxiety == 'Yes' else 0,
        'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
        'CHRONIC_DISEASE': 1 if chronic_disease == 'Yes' else 0,
        'FATIGUE': 1 if fatigue == 'Yes' else 0,
        'ALLERGY': 1 if allergy == 'Yes' else 0,
        'WHEEZING': 1 if wheezing == 'Yes' else 0,
        'ALCOHOL_CONSUMING': 1 if alcohol == 'Yes' else 0,
        'COUGHING': 1 if coughing == 'Yes' else 0,
        'SHORTNESS_OF_BREATH': 1 if shortness == 'Yes' else 0,
        'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
        'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
    }
    feature_order = list(input_data.keys())
    return np.array([[input_data[feat] for feat in feature_order]])

# Predict
if st.button('Predict Lung Cancer Risk'):
    try:
        input_data = prepare_input()
        # prediction = model.predict(input_data).flatten()

        st.subheader('Prediction Results')

        # # Show raw output
        # st.write(f"🔍 Model prediction (raw output): {prediction}")
        # print(f"Model prediction: {prediction}")  # For debug/console

        

        proba = model.predict_proba(input_data)
        if proba[0][1] > 0.3:  # lower threshold
            prediction = [1]
        else:
            prediction = [0]
        if prediction[0] == 1:
            st.error('High risk of lung cancer detected')
            risk_level = "High"
        else:
            st.success('Low risk of lung cancer detected')
            risk_level = "Low"


        # Confidence (if available)
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)
                st.write(f"Risk probability: {proba[0][1]:.1%}")
        except Exception as e:
            st.warning(f"Couldn't calculate confidence: {str(e)}")

        # Recommendation
        st.subheader('Recommendation')
        if risk_level == "High":
            st.warning("""
            **Consult a healthcare professional immediately**  
            • Schedule a doctor's appointment  
            • Consider diagnostic tests  
            • Review risk factors
            """)
        else:
            st.info("""
            **Maintain healthy habits**  
            • Regular check-ups recommended  
            • Avoid smoking  
            • Monitor for symptoms
            """)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.sidebar.header('About')
st.sidebar.info("This tool provides estimates only. Always consult a healthcare professional for medical advice.")
