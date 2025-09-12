import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Set page configuration for a modern look
st.set_page_config(
    page_title="Air Quality Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a modern, elegant UI ---
# This enhances the visual appeal with custom fonts, buttons, and layout.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    .stExpander {
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
        margin-top: 1rem;
    }
    
    .st-bu {
        color: #2c3e50;
    }
    
    .st-ck {
        background-color: #ecf0f1;
        border-radius: 12px;
        padding: 1rem;
    }
    .result-container {
        padding: 2rem;
        border-radius: 16px;
        background-color: #ecf0f1;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
    }
    .status-good { color: #2ecc71; font-weight: bold; }
    .status-moderate { color: #f1c40f; font-weight: bold; }
    .status-poor { color: #e74c3c; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# --- Main Page Content ---
st.markdown("<h1 class='main-header'>Air Quality Index (AQI) Predictor</h1>", unsafe_allow_html=True)
st.write("This application predicts the Air Quality Index (AQI) based on various pollutant levels. Enter the values below and get an instant prediction.")

# --- Load the pre-trained model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from a pkl file."""
    if not os.path.exists('best_model.pkl'):
        st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory.")
        st.stop()
        return None

    try:
        with open('best_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Explicitly check if the loaded object has a 'predict' method
        if hasattr(model, 'predict'):
            return model
        else:
            st.error("The file 'best_model.pkl' was loaded, but it does not contain a valid machine learning model. Please ensure you saved the correct object.")
            st.stop()
            return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()
        return None

model = load_model()

if model:
    # --- Input Fields for User Data ---
    st.subheader("Enter Pollutant Levels")

    col1, col2, col3 = st.columns(3)
    with col1:
        co = st.number_input("Carbon Monoxide (CO)", min_value=0.0, max_value=2000.0, value=500.0, step=10.0)
        no2 = st.number_input("Nitrogen Dioxide (NO₂)", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
    with col2:
        so2 = st.number_input("Sulfur Dioxide (SO₂)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
        o3 = st.number_input("Ozone (O₃)", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
    with col3:
        pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
        pm10 = st.number_input("PM10", min_value=0.0, max_value=600.0, value=100.0, step=1.0)

    # --- Prediction Button ---
    if st.button("Predict Air Quality"):
        # Create a DataFrame from user input with features in the correct order
        input_df = pd.DataFrame([[co, no2, so2, o3, pm25, pm10]],
                                columns=['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10'])
        
        # Convert the DataFrame to a NumPy array to avoid feature name mismatch
        input_array = input_df
        
        # Make a prediction
        prediction = model.predict(input_array)
        
        # Convert prediction to human-readable format, handling both floats and integers
        aqi_mapping = {0: 'Good', 1: 'Moderate', 2: 'Poor'}
        predicted_aqi = aqi_mapping.get(int(prediction[0]))
        
        # Determine color for the result text
        if predicted_aqi == 'Good':
            color_class = 'status-good'
        elif predicted_aqi == 'Moderate':
            color_class = 'status-moderate'
        else:
            color_class = 'status-poor'

        # Display the prediction result
        with st.container():
            # st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.markdown(f"<p class='result-text'>Predicted Air Quality Index: <span class='{color_class}'>{predicted_aqi}</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


